"""Mask reviewer names in bank replies before the data is published.

Bank replies often open with a salutation that names the reviewer ("Dear Mohammed Nadeem,", "Hi Hazem,").
This replaces the name with [NAME]. Generic salutations ("Dear Customer", "Dear Valued Customer") are kept.
Run from the repository root BEFORE committing data:  python src/clean/mask_reply_names.py
Edits in place: data/processed/uae_bank_reviews.csv (bank_reply) and data/raw/play_*.csv (replyContent).
"""
import re, sys, glob
from pathlib import Path
import pandas as pd
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
STOP = {"dear","valued","customer","customers","client","sir","madam","we","thank","thanks","your","sorry","apologies","please",
        "for","greetings","i","it","our","kindly","there","team","the","this","unfortunately","regarding","at","all","user","users",
        "esteemed","hope","you","hello","hi","hey","good","morning","afternoon","evening","mr","mrs","ms","miss","dr","and","to","is","are"}
HEAD = re.compile(r'^(\W*(?:Dear|Hi|Hello|Hey)\s+)([^\n,.!:;]{1,60})')
def mask(t):
    """Replace the name tokens right after the opening salutation with [NAME]."""
    if not isinstance(t, str): return t
    m = HEAD.match(t)
    if not m: return t
    toks = m.group(2).split()
    i = 0
    while i < len(toks) and re.sub(r'\W', '', toks[i]).lower() in STOP: i += 1   # skip "Dear", "Mr", "Valued"...
    j = i
    while j < len(toks) and j - i < 3 and re.sub(r'\W', '', toks[j]).lower() not in STOP and not toks[j].startswith(('[', '%')) and (toks[j][0].isupper() or ord(toks[j][0]) > 127): j += 1
    if j == i: return t
    toks[i:j] = ["[NAME]"]
    return t[:m.start(2)] + " ".join(toks) + t[m.end(2):]
total = 0
p = ROOT / "data" / "processed" / "uae_bank_reviews.csv"
d = pd.read_csv(p); before = d.bank_reply.copy(); d["bank_reply"] = d.bank_reply.map(mask)
n = int((before.fillna("") != d.bank_reply.fillna("")).sum()); total += n
d.to_csv(p, index=False, encoding="utf-8-sig"); print(f"{p.name}: masked {n} replies")
for f in glob.glob(str(ROOT / "data" / "raw" / "play_*.csv")):
    r = pd.read_csv(f); b = r.replyContent.copy(); r["replyContent"] = r.replyContent.map(mask)
    n = int((b.fillna("") != r.replyContent.fillna("")).sum()); total += n
    r.to_csv(f, index=False, encoding="utf-8-sig"); print(f"{Path(f).name}: masked {n}")
print("total masked:", total)
