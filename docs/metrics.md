# Pulse metrics

Results per run, newest at the bottom of each phase. Numbers here are what the commands
printed; nothing is typed in by hand.

## Phase 0: synthetic data and ingest

### Run 2026-11-17, seed 42, 5,000 customers

`python -m pulse.cli data`

| Output | Rows | Time |
| --- | --- | --- |
| customers.csv | 5,000 | |
| accounts.csv | 6,491 | |
| transactions.csv | 1,861,932 (186 MB) | |
| products.csv | 11,406 | |
| _truth.csv (hidden, never read by the pipeline) | 5,000 | |
| total | | 64 s |

Planted truth:

| Item | Count |
| --- | --- |
| churners | 504 (10.1%) |
| of which closed the account | 101 |
| of which faded silently | 403 |
| of which show no warning sign | 80 |
| non-churners showing warning signs then recovering (false alarms) | 338 |

`python -m pulse.cli ingest`: 27 s, data-quality report `RESULT OK`, no nulls outside the
nullable columns (`accounts.close_date` 98% null, `products.end_date` 94.5% null, both
expected), no duplicate keys, no orphans, no negative balances, transaction dates
2024-11-01 to 2026-10-31. Report saved to `data/processed/dq_report.txt`.

Transaction mix: USD 1,287,975 rows, ZiG 573,957 rows. Channels by volume: POS, ZIPIT,
mobile app, RTGS, USSD, ATM, branch.

`python -m pytest`: 16 passed (determinism, running-balance consistency, dates inside
account life, planted churn is noisy, silent churners go quiet, life events raise uptake,
ingest ignores `_truth.csv`, column rename handled by `schema.yaml` alone, broken schema
and duplicate keys are reported).

Caught by the data-quality report during the build: two savings accounts opened after
the customer's closure date. Fixed in the generator.

## Phase 1: features and churn target

### Run 2026-11-17, seed 42, 5,000 customers, churn_rate 0.15

`python -m pulse.cli data`: 1,824,564 transactions, 71 s. Planted: 760 churners (15.2%),
150 closed, 610 silent, 114 with no warning, 317 false alarms.

`python -m pulse.cli pipeline` (ingest then features):

| step | time |
| --- | --- |
| ingest | 26.7 s |
| features | 44.0 s |
| total | 70.8 s |

`features.parquet`: 79,860 rows x 69 features, 5,000 customers, 66,616 labelled rows.

Churn rate by snapshot month (labelled rows):

| snapshot | rows | churn % |
| --- | --- | --- |
| 2025-04-30 | 3,831 | 1.3 |
| 2025-07-31 | 4,013 | 3.0 |
| 2025-10-31 | 4,133 | 3.4 |
| 2026-01-31 | 4,229 | 3.7 |
| 2026-04-30 | 4,336 | 3.9 |
| 2026-07-31 | 4,413 | 4.2 |

Overall 3.3% (hard 0.6%, silent 2.8%). Early months are low because planted churn cannot
start before a customer has six months of history.

Labels against `_truth.csv` (the pipeline never reads it):

| check | result |
| --- | --- |
| planted closures labelled 1 at the snapshot before closure | 129 / 129 |
| planted silent churners labelled 1 at the snapshot before churn | 91% of 530 |
| customers never planted to churn but labelled 1 | 0 |
| days_to_churn, 10th / 50th / 90th percentile | 28 / 31 / 62 days |

Feature medians at the snapshot before churn, planted sign vs non-churners:

| feature | churners with the sign | churners without | non-churners |
| --- | --- | --- | --- |
| months_since_salary | 2.00 | 0.00 | 0.00 |
| salary_change_1m | 0.20 | 1.00 | 1.01 |
| leak_share_trend | 0.15 | 0.00 | 0.00 |
| txn_count_change | 0.60 | 1.00 | 0.98 |
| bal_change_vs_6m | 0.21 | 1.10 | 1.16 |

`python -m pytest`: 21 passed, including the leakage test (features unchanged when all
transactions after the snapshot are rewritten, labels change), the empty-window test,
no NaN or inf in any feature, and labels versus planted truth.
