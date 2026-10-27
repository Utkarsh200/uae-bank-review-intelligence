# UAE Bank App-Review Intelligence (QM640 capstone, Walsh College)

Evidence-grounded retrieval-augmented generation (RAG) for prioritizing mobile-banking improvements,
using public Google Play and App Store reviews of 11 UAE banks (English, Arabic, Arabizi).
Author: Kumar Utkarsh. Mentor: Prof. Javad Katibai.

## Data
Public reviews only, collected with `google-play-scraper` and `app-store-scraper` (no Kaggle, no synthetic data).
Usernames, user IDs and profile images are dropped at collection time; e-mails and long digit strings are masked.
The anonymized dataset is `data/processed/uae_bank_reviews.csv`; variables are defined in Table 5 of the synopsis.
App identifiers for every bank are in `config/banks.csv`.

## Quick start
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `python src/ingest/scrape_reviews.py resolve`  then paste the correct IDs into `config/banks.csv`
4. `python src/ingest/scrape_reviews.py verify`   (every row must show the right bank as developer)
5. `python src/ingest/scrape_reviews.py play`, then optionally `ios`, then `merge`
6. `python tools/make_dataset_screenshot.py`  -> `reports/figures/figure1_dataset_sample.png`

## Folder structure
`config/` domain profile and bank list | `data/raw` scraper output (git-ignored) | `data/processed` anonymized dataset |
`data/eval` analyst questions and relevance judgments | `src/` pipeline stages | `tools/` helpers | `reports/figures/` figures

## Reproducibility
Pin versions with `pip freeze > requirements.lock`; record the model name/version, temperature (0) and random seed
for every LLM run. Scraping dates are stored in the `collected_on` column.

## Responsible use
Reviews are public but personal; do not attempt to re-identify reviewers. Check each store's terms before reuse.
