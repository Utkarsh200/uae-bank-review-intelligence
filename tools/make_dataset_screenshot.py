"""Render the first rows of the collected dataset as a PNG for Figure 1 of the synopsis.

Usage:  python tools/make_dataset_screenshot.py [path/to/uae_bank_reviews.csv]
Output: reports/figures/figure1_dataset_sample.png
(Opening the CSV in Excel and taking a normal screenshot is equally acceptable.)
"""
import re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "processed" / "uae_bank_reviews.csv"
df = pd.read_csv(src, dtype=str).fillna("")
is_ar = df["review_text"].str.contains(r"[\u0600-\u06FF]")
sample = pd.concat([df[~is_ar].head(4), df[is_ar].head(4)]).reset_index(drop=True)
if sample.empty:
    sys.exit("No rows found in the CSV.")

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    shape = lambda t: get_display(arabic_reshaper.reshape(t)) if re.search(r"[\u0600-\u06FF]", t) else t
except ImportError:
    shape = lambda t: t

def cell(t, n=60):
    t = re.sub(r"\s+", " ", t).strip()
    return shape(t if len(t) <= n else t[:n - 1] + "…")

cols = ["review_id", "bank", "platform", "rating", "review_date", "app_version", "review_text"]
rows = [[cell(str(r[c]), 14 if c == "review_id" else 60 if c == "review_text" else 22) for c in cols]
        for _, r in sample.iterrows()]
fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.2), dpi=200)
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.6)
tbl.auto_set_column_width(list(range(len(cols))))
for (r, c), cl in tbl.get_celld().items():
    cl.set_edgecolor("#999999")
    if r == 0:
        cl.set_facecolor("#E6E6E6"); cl.set_text_props(weight="bold")
out = ROOT / "reports" / "figures" / "figure1_dataset_sample.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("Saved", out)
