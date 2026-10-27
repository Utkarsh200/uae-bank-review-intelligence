# UAE Bank App-Review Intelligence (QM640 capstone, Walsh College)

Evidence-grounded retrieval-augmented generation (RAG) for prioritizing mobile-banking improvements,
using public Google Play reviews of 11 UAE banks (English, Arabic, Arabizi).
Author: Kumar Utkarsh. Mentor: Prof. Javad Katibai.

## Data

Public reviews only, collected with `google-play-scraper` (no Kaggle, no synthetic data).
Usernames, user IDs and profile images are dropped at collection time; e-mails and long digit strings are masked.

### Files in the repository

| Path | Tracked in git | Contents |
| --- | --- | --- |
| `data/processed/uae_bank_reviews.csv` | yes (~18 MB, UTF-8 with BOM) | the analysis dataset: one row per review, 12 columns, anonymized and de-duplicated |
| `data/raw/play_<bank>.csv` | no (git-ignored) | per-bank scraper output in the store's native schema, before cleaning |
| `config/banks.csv` | yes | the 11 banks with their verified Google Play package IDs |
| `config/banks_candidates.csv` | yes | raw output of `scrape_reviews.py resolve`: search hits (app id, title, developer) used to pick the IDs above |
| `config/uae_banking.yaml` | yes | domain profile (aspect taxonomy and terminology) used downstream |
| `data/eval/` | yes (currently empty) | analyst questions and relevance judgments for RAG evaluation |

### Schema of `data/processed/uae_bank_reviews.csv`

| Column | Type | Description |
| --- | --- | --- |
| `review_id` | string | store-assigned review UUID; unique across the file |
| `bank` | string | bank/app label, one of the 11 names in `config/banks.csv` |
| `group` | string | `Conventional`, `Islamic` or `Digital-only` — the comparison factor for the study |
| `platform` | string | source store; `google_play` for every row |
| `review_text` | string | review body after masking (`[EMAIL]`, `[NUMBER]`); never empty |
| `rating` | integer 1-5 | star rating given by the reviewer |
| `review_date` | timestamp | when the review was posted (store local timestamp) |
| `app_version` | string | app version the review was written against; blank when the store did not report it (~12% of rows) |
| `helpful_votes` | integer | thumbs-up count; non-zero for ~13% of rows |
| `bank_reply` | string | the bank's public reply, blank if none |
| `reply_date` | timestamp | when the bank replied, blank if no reply |
| `collected_on` | date | scrape date, for reproducibility |

### Coverage of the current snapshot (collected 2026-10-27)

46,685 Google Play reviews posted between 2024-10-01 and 2026-09-30
(the window hard-coded as `START`/`END` in the scraper).

| Group | Bank | Reviews | Mean rating |
| --- | --- | ---: | ---: |
| Conventional | Emirates NBD (ENBD X) | 10,574 | 4.17 |
| Conventional | Mashreq | 9,011 | 4.20 |
| Conventional | ADCB (ADCB mobile banking) | 7,039 | 3.19 |
| Conventional | First Abu Dhabi Bank (FAB Mobile) | 5,861 | 3.99 |
| Conventional | RAKBANK | 1,280 | 3.64 |
| Conventional | Commercial Bank of Dubai (CBD) | 792 | 3.05 |
| Islamic | Abu Dhabi Islamic Bank (ADIB) | 3,998 | 4.45 |
| Islamic | Emirates Islamic | 3,419 | 4.00 |
| Islamic | Dubai Islamic Bank (DIB) | 2,053 | 3.25 |
| Digital-only | Liv X (Emirates NBD) | 2,200 | 3.60 |
| Digital-only | Wio | 458 | 3.48 |

Ratings are strongly J-shaped: 5★ 30,934 | 4★ 2,685 | 3★ 1,352 | 2★ 1,302 | 1★ 10,412.
About 13% of reviews contain Arabic script (the rest are English or Arabizi, which is Latin-script Arabic and
is not separable by script detection). Reviews are short — median 16 characters, 75th percentile 53 —
so most of the analytical signal sits in the long tail of detailed complaints.

### Known limitations of the snapshot

- **Google Play only.** The study is scoped to Android reviews; iOS users are out of scope and any
  platform-level conclusions should be read with that in mind.
- **Reply bias.** 91% of reviews carry a `bank_reply`, but most replies are templated boilerplate; treat them
  as a responsiveness signal, not as content.
- **Text truncation.** Google Play caps `review_text` at 500 characters.
- **Sampling.** Reviews are pulled newest-first per bank and language (cap `MAX_PER_LANG = 20000`), so counts
  reflect app popularity and review velocity, not a balanced design — weight or stratify before comparing groups.
- **Snapshot, not a feed.** Figures above describe the file as committed; re-scraping on a later date will shift
  counts and ratings.

## Quick start
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `python src/ingest/scrape_reviews.py resolve`  then paste the correct IDs into `config/banks.csv`
4. `python src/ingest/scrape_reviews.py verify`   (every row must show the right bank as developer)
5. `python src/ingest/scrape_reviews.py play`, then `merge`
6. `python tools/make_dataset_screenshot.py`  -> `reports/figures/figure1_dataset_sample.png`

Steps 3-5 overwrite `data/raw/` and `data/processed/uae_bank_reviews.csv`; skip them to work from the
committed snapshot.

## Folder structure
`config/` domain profile and bank list | `data/raw` scraper output (git-ignored) | `data/processed` anonymized dataset |
`data/eval` analyst questions and relevance judgments | `src/` pipeline stages | `tools/` helpers | `reports/figures/` figures

## Reproducibility
Pin versions with `pip freeze > requirements.lock`; record the model name/version, temperature (0) and random seed
for every LLM run. Scraping dates are stored in the `collected_on` column.

## Responsible use
Reviews are public but personal; do not attempt to re-identify reviewers. Check each store's terms before reuse.
