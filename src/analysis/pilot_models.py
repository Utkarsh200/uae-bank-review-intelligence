"""Preliminary models for the interim report.

(1) RQ1 explanatory model: logistic regression of a low rating (1-2 stars) on theme, bank group, language and
    review length, with a majority-class baseline and a TF-IDF text baseline. The train/test split and the 5-fold
    cross-validation are grouped by identical review text, so a repeated text never sits in both train and test.
(2) RQ4 rolling-origin validation: train on earlier months, predict the next month's rating decline.
Run from the repository root AFTER src/analysis/pilot_eda.py:  python src/analysis/pilot_models.py
Writes reports/pilot_models.json
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
warnings.filterwarnings("ignore")
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
d = pd.read_csv(ROOT / "data" / "processed" / "analysis_dataset.csv")
d["low"] = (d.rating <= 2).astype(int)
d["loglen"] = np.log1p(d.review_length)
grp = d.review_text.str.lower().str.strip()
res = {}
def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(auc=roc_auc_score(y, p), accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), f1=f1_score(y, yh))
X_meta = pd.get_dummies(d[["theme", "group", "language"]], drop_first=False).astype(float)
X_meta_len = pd.concat([X_meta, d[["loglen"]]], axis=1)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(d, groups=grp))
y = d.low.values
res["split"] = dict(train=int(len(tr)), test=int(len(te)), low_share_test=float(y[te].mean()))
res["majority"] = metrics(y[te], DummyClassifier(strategy="prior").fit(X_meta.iloc[tr], y[tr]).predict_proba(X_meta.iloc[te])[:, 1])
res["majority"].update(accuracy=float(1 - y[te].mean()), precision=0.0, recall=0.0, f1=0.0)
def fit_eval(X, name, Xte=None):
    m = LogisticRegression(max_iter=2000, C=1.0, class_weight=None).fit(X[tr] if sps.issparse(X) else X.iloc[tr], y[tr])
    p = m.predict_proba(X[te] if sps.issparse(X) else X.iloc[te])[:, 1]
    res[name] = metrics(y[te], p)
    # grouped 5-fold CV AUC
    aucs = []
    for a, b in GroupKFold(5).split(d, groups=grp):
        mm = LogisticRegression(max_iter=2000).fit(X[a] if sps.issparse(X) else X.iloc[a], y[a])
        aucs.append(roc_auc_score(y[b], mm.predict_proba(X[b] if sps.issparse(X) else X.iloc[b])[:, 1]))
    res[name]["cv_auc_mean"] = float(np.mean(aucs)); res[name]["cv_auc_sd"] = float(np.std(aucs))
    return m
fit_eval(X_meta, "logit_theme_group_language")
fit_eval(X_meta_len, "logit_theme_group_language_length")
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
Xt = tf.fit_transform(d.review_text.fillna("")[tr]); Xall = sps.csr_matrix(tf.transform(d.review_text.fillna("")))
fit_eval(Xall, "tfidf_text_baseline")
# odds ratios for themes (statsmodels, train rows, reference = other / Conventional / en)
ds = d.iloc[tr]
f = smf.logit("low ~ C(theme, Treatment('other')) + C(group, Treatment('Conventional')) + C(language, Treatment('en')) + loglen", ds).fit(disp=0)
orr = np.exp(f.params); ci = np.exp(f.conf_int())
res["odds_ratios"] = {k: dict(OR=float(orr[k]), lo=float(ci.loc[k, 0]), hi=float(ci.loc[k, 1])) for k in orr.index if k != "Intercept"}
# RQ4 rolling-origin validation
pan = pd.read_csv(ROOT / "data" / "processed" / "rq4_panel.csv").sort_values("m")
months = sorted(pan.m.unique()); pred = {"mean": [], "m1": [], "m2": [], "y": []}
for t in months:
    trn, tst = pan[pan.m < t], pan[pan.m == t]
    if len(trn) < 60 or len(tst) == 0: continue
    m1 = smf.ols("decline ~ freq_share + rating_m", trn).fit(); m2 = smf.ols("decline ~ freq_share + rating_m + P", trn).fit()
    pred["mean"] += [trn.decline.mean()] * len(tst); pred["m1"] += list(m1.predict(tst)); pred["m2"] += list(m2.predict(tst)); pred["y"] += list(tst.decline)
y4 = np.array(pred["y"]); rm = lambda p: float(np.sqrt(np.mean((y4 - np.array(p)) ** 2)))
sse = lambda p: float(np.sum((y4 - np.array(p)) ** 2))
res["rq4_rolling"] = dict(n_test=int(len(y4)), rmse_mean=rm(pred["mean"]), rmse_m1=rm(pred["m1"]), rmse_m2=rm(pred["m2"]),
                          r2oos_m1=1 - sse(pred["m1"]) / sse(pred["mean"]), r2oos_m2=1 - sse(pred["m2"]) / sse(pred["mean"]))
json.dump(res, open(ROOT / "reports" / "pilot_models.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "odds_ratios"}, indent=1))
