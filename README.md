# UAE Bank App-Review Intelligence (QM640 capstone, Walsh College)

Evidence-grounded retrieval-augmented generation (RAG) for prioritizing mobile-banking app improvements,
using public Google Play reviews of 11 UAE banks (English, Arabic, Arabizi).
Author: Kumar Utkarsh. Mentor: Prof. Javad Katibai. Status: interim report (Oct 27, 2026); final report due Dec 13.

## Research questions
1. RQ1: Which complaint themes drive low ratings, and do they differ by bank group (conventional, Islamic, digital-only)?
2. RQ2: Does retrieval-grounded answering improve faithfulness over answering without retrieval?
3. RQ3: Does Arabic retrieval match English retrieval quality (dense vs. hybrid BGE-M3 + BM25)?
4. RQ4: Does a priority score (frequency, severity, recency, rating impact) predict next-month rating decline?

## Data
- Public Google Play reviews only (no Kaggle, no synthetic data); 46,685 reviews, Oct 2024 - Sep 2026. iOS was not collected (the App Store endpoint rejected the scraper).
- Usernames, user IDs and profile images are dropped at collection; e-mails and long digit strings are masked.
- Bank replies greet reviewers by name, so run `python src/clean/mask_reply_names.py` before committing data.
- Anonymized dataset: `data/processed/uae_bank_reviews.csv`. App IDs: `config/banks.csv`. Domain profile (8 themes, severity and priority weights): `config/uae_banking.yaml`.

## Setup
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`; also install `scipy statsmodels scikit-learn scikit-posthocs numpy` (add them to `requirements.txt`).

## Collect data
1. `python src/ingest/scrape_reviews.py resolve`, then paste the correct IDs into `config/banks.csv`
2. `python src/ingest/scrape_reviews.py verify` (each row must show the right bank as developer)
3. `python src/ingest/scrape_reviews.py play`, then `python src/ingest/scrape_reviews.py merge`
4. `python src/clean/mask_reply_names.py`

## Reproduce the interim results
| Command | Output |
|---|---|
| `python src/analysis/cleaning_log.py` | Counts for the cleaning log (Table 3) |
| `python src/analysis/pilot_eda.py` | Figures 4-8, RQ1/RQ4 pilot tests, `reports/pilot_results.json`, `data/processed/analysis_dataset.csv`, `rq4_panel.csv` |
| `python src/analysis/pilot_models.py` | Logistic model, majority and TF-IDF baselines, rolling-origin RQ4 check, `reports/pilot_models.json` |
| `python tools/make_dataset_screenshot.py` | Dataset preview figure |

Themes in the interim results come from a keyword labeler (exploratory); the LLM classifier replaces it in the final report.

## Folder structure
- `config/`: domain profile and bank list
- `data/raw/`: scraper output (committed, names masked)
- `data/processed/`: anonymized dataset and analysis tables
- `data/eval/`: analyst questions and relevance judgments (planned)
- `src/ingest`, `src/clean`, `src/analysis`: pipeline stages
- `tools/`: helpers
- `reports/figures/`, `reports/*.json`: figures and result files

## Reproducibility
Pin versions with `pip freeze > requirements.lock`; record model name/version, temperature 0 and random seed for every LLM run.
Scraping dates are in the `collected_on` column. Train/test splits are grouped by identical review text to prevent leakage.

## Responsible use
Reviews are public but personal; do not attempt to re-identify reviewers. Check each store's terms before reuse.
