"""Reproduce the counts in the data cleaning log (Table 3 of the interim report).
Run from the repository root:  python src/analysis/cleaning_log.py
Reads data/raw/play_*.csv and data/processed/uae_bank_reviews.csv; prints the counts.
"""
import sys, glob
from pathlib import Path
import pandas as pd
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
raw = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "data" / "raw" / "play_*.csv"))])
raw["at"] = pd.to_datetime(raw["at"])
d = pd.read_csv(ROOT / "data" / "processed" / "uae_bank_reviews.csv", parse_dates=["review_date"])
out = ((raw["at"] < "2024-10-01") | (raw["at"] > "2026-09-30 23:59:59")).sum()
print(f"raw rows: {len(raw)}   duplicate review IDs: {raw.reviewId.duplicated().sum()}")
print(f"outside window: {out} ({out/len(raw):.1%});  earliest raw review: {raw['at'].min()}")
print(f"empty or missing review text: {raw.content.fillna('').str.strip().eq('').sum()}")
print(f"processed rows: {len(d)}")
print(f"missing app_version: {d.app_version.isna().sum()} ({d.app_version.isna().mean():.1%});  no bank reply: {d.bank_reply.isna().sum()} ({d.bank_reply.isna().mean():.1%})")
print(f"repeated text within the same bank: {d.duplicated(['bank','review_text']).sum()} ({d.duplicated(['bank','review_text']).mean():.1%})")
n = d.review_text.str.split().str.len()
print(f"fewer than 3 tokens: {(n<3).sum()}; length > p99 ({n.quantile(.99):.0f} tokens): {(n>n.quantile(.99)).sum()}")
print(f"helpful_votes > p99 ({d.helpful_votes.quantile(.99):.0f}): {(d.helpful_votes>d.helpful_votes.quantile(.99)).sum()}")
print(d.groupby("bank").size().sort_values().to_string())
