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

## Phase 2: churn model and honest evaluation

### Run 2026-11-18, seed 42, 5,000 customers, label corrected (see notes)

`python -m pulse.cli features` after the label fix: 78,546 rows x 69 features, 61,246
labelled rows, churn rate 2.2% (Apr 2025) rising to 3.2% (Jun 2026), 2.9% overall
(hard 0.6%, silent 2.3%). Last labelled snapshot 2026-06-30.

`python -m pulse.cli train`: 20 s. `python -m pulse.cli pipeline`: ingest 16.3 s,
features 34.6 s, train 26.8 s, total 77.6 s.

First run, before the label fix (kept for the record): ROC-AUC 0.952, PR-AUC 0.838,
capture@10% 89.7%, rules baseline alone 88.2%, six balance features above 0.9 ROC-AUC
alone. Diagnosis against `_truth.csv`: half of the rows labelled 1 were dated after the
planted churn date (median lead -11 days). The label was rewritten so churn is only ever
labelled before it happens; everything below is after that fix.

Everything below is pasted from `models/train_report.md`, which the train step writes.


**Time split (labelled snapshots only; the gap before the test block is dropped)**

| block | from | to | snapshots | rows | churners | churn_% |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2025-04-30 | 2025-10-31 | 7 | 27646 | 703 | 2.54 |
| valid | 2025-11-30 | 2026-01-31 | 3 | 12360 | 364 | 2.94 |
| test | 2026-04-30 | 2026-06-30 | 3 | 12814 | 413 | 3.22 |

**LightGBM candidates, chosen on the validation block (test never looked at)**

| candidate | num_leaves | min_child_samples | feature_fraction | lambda_l2 | learning_rate | rounds | train_pr_auc | valid_pr_auc | chosen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 100 | 0.8 | 1.0 | 0.03 | 277 | 0.809 | 0.548 |  |
| 2 | 7 | 200 | 0.7 | 10.0 | 0.03 | 159 | 0.624 | 0.539 | * |
| 3 | 4 | 300 | 0.6 | 10.0 | 0.03 | 282 | 0.617 | 0.532 |  |
| 4 | 7 | 300 | 0.5 | 30.0 | 0.02 | 533 | 0.686 | 0.54 |  |

**Test block: three models**

| model | PR-AUC | ROC-AUC | capture@5% | capture@10% | capture@20% | lift@5% | lift@10% | lift@20% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | 0.437 | 0.844 | 58.4 | 67.3 | 78.0 | 11.67 | 6.73 | 3.9 |
| logistic | 0.45 | 0.856 | 62.0 | 68.3 | 75.1 | 12.4 | 6.83 | 3.75 |
| lightgbm | 0.512 | 0.875 | 64.6 | 71.7 | 79.2 | 12.93 | 7.17 | 3.96 |

**LightGBM calibration on the test block (decile 10 = riskiest)**

| decile | rows | raw_pred_% | actual_% | calibrated_pred_% |
| --- | --- | --- | --- | --- |
| 1 | 1282 | 4.34 | 0.39 | 0.36 |
| 2 | 1282 | 7.86 | 0.39 | 0.51 |
| 3 | 1282 | 10.45 | 0.86 | 0.68 |
| 4 | 1282 | 12.92 | 0.55 | 0.69 |
| 5 | 1281 | 15.75 | 1.17 | 0.69 |
| 6 | 1281 | 18.9 | 1.01 | 0.85 |
| 7 | 1281 | 22.86 | 1.56 | 1.49 |
| 8 | 1281 | 28.8 | 0.78 | 1.5 |
| 9 | 1281 | 39.85 | 2.42 | 2.26 |
| 10 | 1281 | 76.13 | 23.11 | 24.26 |

**Calibration summary**

| brier_raw | brier_calibrated | ece_raw | ece_calibrated | mean_pred_raw | mean_pred_calibrated | actual_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0857 | 0.0204 | 0.2056 | 0.0027 | 0.2378 | 0.0333 | 0.0322 |

**Top 15 LightGBM features by gain**

| feature | gain_% | reading |
| --- | --- | --- |
| bal_change_vs_6m | 37.9 | current balance divided by the average of the 6 months before (1 = unchanged); churners have lower values |
| bal_slope_6m | 5.8 | monthly change in balance over the last 6 months as a share of the 6-month average (negative = falling); churners have lower values |
| bal_min_3m_ratio | 5.3 | lowest month-end balance of the last 3 months divided by the 6-month average; churners have lower values |
| income_change | 5.2 | this month's income divided by the monthly average of the 6 months before it (1 = unchanged); churners have lower values |
| net_flow_3m_usd | 3.3 | money in minus money out over the last 3 months, USD; churners have lower values |
| wallet_share_1m | 3.1 | share of this month's outflows sent to mobile wallets (EcoCash and similar); churners have higher values |
| txn_3m_vs_prior_3m | 2.8 | transactions in the last 3 months divided by the 3 months before; churners have lower values |
| bal_drop_from_3m_peak | 2.5 | share of the 3-month peak balance that has gone (0 = at peak, 0.8 = 80% gone); churners have higher values |
| remit_amount_3m_usd | 2.2 | remittances received in the last 3 months, USD; churners have similar values |
| txn_count_change | 2.0 | this month's transactions divided by the monthly average of the 6 months before; churners have lower values |
| obank_share_1m | 1.6 | share of this month's outflows sent to other banks; churners have higher values |
| school_amount_12m_usd | 1.5 | school fees paid in the last 12 months, USD; churners have similar values |
| tenure_months | 1.5 | months since the customer opened their first account; churners have lower values |
| atm_share_3m | 1.3 | share of the last 3 months' transactions at an ATM; churners have lower values |
| leak_share_trend | 1.3 | combined leak share, last 3 months minus the 3 months before; churners have higher values |

**Strongest single features on the test block (leakage warning above 0.9)**

| feature | roc_auc_alone |
| --- | --- |
| bal_change_vs_6m | 0.81 |
| net_flow_3m_usd | 0.806 |
| bal_slope_6m | 0.802 |
| bal_drop_from_3m_peak | 0.802 |
| bal_min_3m_ratio | 0.787 |
| net_flow_1m_usd | 0.783 |
| income_change | 0.779 |
| bal_end_usd | 0.731 |
| salary_change_1m | 0.723 |
| txn_count_change | 0.7 |

**LightGBM train vs test (overfitting check)**

| metric | train | test | gap |
| --- | --- | --- | --- |
| PR-AUC | 0.624 | 0.512 | 0.112 |
| ROC-AUC | 0.96 | 0.875 | 0.085 |
| capture@10% | 84.8 | 71.7 | 13.1 |

**Fairness on the test block at the top-10% cut (protected attributes never used as inputs)**

| attribute | group | rows | churn_% | flag_% | capture_% |
| --- | --- | --- | --- | --- | --- |
| gender | F | 6060 | 3.23 | 10.02 | 75.0 |
| gender | M | 6754 | 3.21 | 9.98 | 68.7 |
| age_band | 18-24 | 1250 | 3.12 | 10.0 | 66.7 |
| age_band | 25-34 | 3762 | 2.95 | 10.02 | 72.1 |
| age_band | 35-44 | 3684 | 3.42 | 10.5 | 77.8 |
| age_band | 45-54 | 2351 | 4.47 | 10.8 | 65.7 |
| age_band | 55-64 | 1224 | 1.55 | 7.03 | 68.4 |
| age_band | 65+ | 543 | 2.39 | 9.58 | 76.9 |

`python -m pytest`: 26 passed (21 from Phases 0 and 1, plus split order and gap, split rejects
overlap, protected attributes absent, top-k capture arithmetic, deterministic training and reload).

Diagnostics against `_truth.csv` (test block, never used by the pipeline): churners planted with
no warning sign score ROC-AUC 0.585, those with at least one sign 0.923; median lead between a
row labelled 1 and the planted churn date 32 days; most planted churners are labelled 1 on
three snapshots.
