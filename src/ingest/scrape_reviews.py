"""Collect public UAE bank app reviews (Google Play + App Store) into one anonymized CSV.

Run order (from the repository root):
  python src/ingest/scrape_reviews.py resolve   # find candidate Google Play IDs for banks missing one
  python src/ingest/scrape_reviews.py verify    # confirm every play_id points to the right bank
  python src/ingest/scrape_reviews.py play      # scrape Google Play (English + Arabic)
  python src/ingest/scrape_reviews.py ios       # scrape the Apple App Store (optional, capped by Apple)
  python src/ingest/scrape_reviews.py merge     # clean, anonymize and write data/processed/uae_bank_reviews.csv
"""
import argparse, hashlib, re, sys, time
from datetime import date, datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BANKS = ROOT / "config" / "banks.csv"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "uae_bank_reviews.csv"
COUNTRY, LANGS = "ae", ["en", "ar"]
START, END = datetime(2024, 10, 1), datetime(2026, 9, 30, 23, 59, 59)
MAX_PER_LANG = 20000
COLS = ["review_id", "bank", "group", "platform", "review_text", "rating", "review_date",
        "app_version", "helpful_votes", "bank_reply", "reply_date", "collected_on"]

def load_banks():
    return pd.read_csv(BANKS, dtype=str).fillna("")

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def cmd_resolve(_):
    from google_play_scraper import search
    rows = []
    for _, b in load_banks().iterrows():
        if b.play_id:
            continue
        for h in search(f"{b.bank} mobile banking", lang="en", country=COUNTRY, n_hits=5):
            rows.append({"bank": b.bank, "candidate_id": h["appId"], "title": h["title"],
                         "developer": h["developer"], "score": h.get("score")})
        time.sleep(1)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "config" / "banks_candidates.csv", index=False)
    print(out.to_string(index=False))
    print("\nOpen each candidate on Google Play, check the developer is the bank, then paste the "
          "correct candidate_id into the play_id column of config/banks.csv.")

def cmd_verify(_):
    from google_play_scraper import app
    for _, b in load_banks().iterrows():
        if not b.play_id:
            print(f"{b.bank:34} -- no play_id yet"); continue
        try:
            d = app(b.play_id, lang="en", country=COUNTRY)
            print(f"{b.bank:34} {b.play_id:30} {d['title']} | {d['developer']} | "
                  f"score {d['score'] or 0:.1f} | ratings {d['ratings'] or 0:,}")
        except Exception as e:
            print(f"{b.bank:34} {b.play_id:30} NOT FOUND ({e})")

def fetch_play(app_id, lang):
    from google_play_scraper import Sort, reviews
    out, token = [], None
    while len(out) < MAX_PER_LANG:
        batch, token = reviews(app_id, lang=lang, country=COUNTRY, sort=Sort.NEWEST,
                               count=200, continuation_token=token)
        out.extend(batch)
        if not batch or token is None or min(r["at"] for r in batch) < START:
            break
        time.sleep(0.5)
    return out

def cmd_play(_):
    RAW.mkdir(parents=True, exist_ok=True)
    for _, b in load_banks().iterrows():
        if not b.play_id:
            print(f"skip {b.bank}: no play_id"); continue
        rows = []
        for lang in LANGS:
            data = fetch_play(b.play_id, lang)
            print(f"{b.bank:34} [{lang}] {len(data):6} reviews")
            rows += data
        if not rows:
            continue
        df = pd.DataFrame(rows).drop_duplicates(subset="reviewId")
        df = df.drop(columns=["userName", "userImage"], errors="ignore")  # anonymize at source
        df["bank"], df["group"], df["platform"] = b.bank, b.group, "google_play"
        df.to_csv(RAW / f"play_{slug(b.bank)}.csv", index=False, encoding="utf-8-sig")

def cmd_ios(_):
    from app_store_scraper import AppStore
    RAW.mkdir(parents=True, exist_ok=True)
    for _, b in load_banks().iterrows():
        if not b.ios_id:
            print(f"skip {b.bank}: no ios_id"); continue
        a = AppStore(country=COUNTRY, app_name=slug(b.bank), app_id=int(b.ios_id))
        a.review(how_many=5000)
        print(f"{b.bank:34} [ios] {len(a.reviews):6} reviews")
        if a.reviews:
            df = pd.DataFrame(a.reviews).drop(columns=["userName"], errors="ignore")
            df["bank"], df["group"], df["platform"] = b.bank, b.group, "app_store"
            df.to_csv(RAW / f"ios_{slug(b.bank)}.csv", index=False, encoding="utf-8-sig")

MASKS = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"), (re.compile(r"\d[\d\s-]{6,}\d"), "[NUMBER]")]
def mask(text):
    for pat, rep in MASKS:
        text = pat.sub(rep, str(text))
    return text

def cmd_merge(_):
    frames = []
    for f in sorted(RAW.glob("*.csv")):
        d = pd.read_csv(f, dtype=str).fillna("")
        if f.name.startswith("play_"):
            d = d.rename(columns={"reviewId": "review_id", "content": "review_text", "score": "rating",
                                  "at": "review_date", "reviewCreatedVersion": "app_version",
                                  "thumbsUpCount": "helpful_votes", "replyContent": "bank_reply",
                                  "repliedAt": "reply_date"})
        else:
            d = d.rename(columns={"review": "review_text", "date": "review_date"})
            d["review_id"] = [("ios-" + hashlib.md5((r.bank + r.review_date + r.review_text).encode()).hexdigest()[:16])
                              for r in d.itertuples()]
            d["app_version"], d["helpful_votes"], d["bank_reply"], d["reply_date"] = "", "", "", ""
        frames.append(d.reindex(columns=[c for c in COLS if c != "collected_on"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="review_id")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df[(df.review_date >= START) & (df.review_date <= END) & (df.review_text.str.strip() != "")]
    df["review_text"] = df["review_text"].map(mask)
    df["collected_on"] = date.today().isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df):,} reviews -> {OUT}")
    print(df.groupby(["group", "bank", "platform"]).size().to_string())

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["resolve", "verify", "play", "ios", "merge"])
    args = p.parse_args()
    globals()[f"cmd_{args.command}"](args)
