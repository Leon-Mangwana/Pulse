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

## Phase 3: value at risk and time-to-churn

### Run 2026-11-18, seed 42, 5,000 customers

`python -m pulse.cli train` now also fits the Cox model and writes `models/value_report.md`
(6 s on top of the churn model). `python -m pulse.cli score` writes
`data/processed/worklist.parquet` and `id_map.parquet` in 3.4 s.

Worklist at the latest snapshot (2026-10-31): 4,323 customers; total 12-month value
1,035,344 USD; value at risk 22,374 USD (2.2%). Urgency bands: act this week 52, act this
month 46, watch 4,225. Top 20 by value at risk: all with balance at 0 to 16% of its
6-month average and transaction count down 50 to 90%; 11 SME, 5 salaried, 4 informal.

Everything below is pasted from `models/value_report.md`, which the train step writes.


**Cox coefficients (features standardised; exp(coef) is the hazard multiplier per standard deviation)**

| feature | coef | exp(coef) | p | reading |
| --- | --- | --- | --- | --- |
| wallet_share_1m | 0.126 | 1.134 | 2.16e-08 | one standard deviation higher multiplies the rate of leaving by 1.13 |
| leak_share_trend | 0.094 | 1.099 | 5.02e-05 | one standard deviation higher multiplies the rate of leaving by 1.10 |
| months_since_salary | 0.017 | 1.017 | 0.468 | one standard deviation higher multiplies the rate of leaving by 1.02 |
| net_flow_3m_usd | -0.089 | 0.915 | 0.00951 | one standard deviation higher multiplies the rate of leaving by 0.92 |
| bal_slope_6m | -0.151 | 0.86 | 3.93e-06 | one standard deviation higher multiplies the rate of leaving by 0.86 |
| income_change | -0.208 | 0.812 | 1.37e-12 | one standard deviation higher multiplies the rate of leaving by 0.81 |
| txn_3m_vs_prior_3m | -0.215 | 0.807 | 2.22e-13 | one standard deviation higher multiplies the rate of leaving by 0.81 |
| txn_count_change | -0.238 | 0.788 | 3.86e-15 | one standard deviation higher multiplies the rate of leaving by 0.79 |
| bal_min_3m_ratio | -0.259 | 0.772 | 5.93e-17 | one standard deviation higher multiplies the rate of leaving by 0.77 |
| bal_change_vs_6m | -0.27 | 0.764 | 5.22e-13 | one standard deviation higher multiplies the rate of leaving by 0.76 |

**Concordance on the test block (chance = 0.5): who leaves first?**

| ranking | concordance_test |
| --- | --- |
| Cox partial hazard | 0.848 |
| LightGBM calibrated probability | 0.885 |

**Test churners by when they left, and the Cox probabilities they were given**

| churned | churners | mean_p30_% | mean_p90_% |
| --- | --- | --- | --- |
| within 30 days | 61 | 10.8 | 40.1 |
| 31 to 60 days | 117 | 10.4 | 40.2 |
| 61 to 92 days | 235 | 5.4 | 21.9 |

**Urgency bands on the test block (thresholds in settings.yaml)**

| band | rows | churned_within_90d_% | churned_within_30d_% | share_of_rows_% |
| --- | --- | --- | --- | --- |
| act this week | 121 | 76.0 | 16.5 | 0.9 |
| act this month | 275 | 48.4 | 7.3 | 2.1 |
| watch | 12418 | 1.5 | 0.2 | 96.9 |

**Average 90-day probability versus the actual rate on the test block**

| mean_cox_p90_% | mean_lightgbm_calibrated_% | actual_churn_% |
| --- | --- | --- |
| 2.92 | 3.33 | 3.22 |

**12-month value on the test block, USD**

| stat | fee_income_12m_usd | interest_margin_12m_usd | product_margin_12m_usd | value_12m_usd |
| --- | --- | --- | --- | --- |
| mean | 68.0 | 124.0 | 24.0 | 216.0 |
| 50% | 54.0 | 78.0 | 6.0 | 156.0 |
| 90% | 124.0 | 268.0 | 59.0 | 424.0 |
| max | 365.0 | 1834.0 | 291.0 | 2058.0 |

**Call-list comparison on the test block: 12-month value of customers who actually churned (total 81,894 USD across 413 churners)**

| ordering | top_% | contacted | churned_value_reached_usd | share_of_churned_value_% | churners_reached_% |
| --- | --- | --- | --- | --- | --- |
| random | 5 | 640 | 4590 | 5.6 | 6.3 |
| random | 10 | 1281 | 8971 | 11.0 | 11.9 |
| random | 20 | 2562 | 15870 | 19.4 | 23.2 |
| by churn probability | 5 | 640 | 48749 | 59.5 | 63.9 |
| by churn probability | 10 | 1281 | 55282 | 67.5 | 71.7 |
| by churn probability | 20 | 2562 | 60211 | 73.5 | 78.2 |
| by value at risk | 5 | 640 | 52882 | 64.6 | 64.4 |
| by value at risk | 10 | 1281 | 57958 | 70.8 | 70.7 |
| by value at risk | 20 | 2562 | 68183 | 83.3 | 78.5 |

`python -m pulse.cli pipeline`: ingest 19.5 s, features 29.1 s, train 31.3 s, score 0.9 s, total 80.8 s.

`python -m pytest`: 31 passed (26 from earlier phases, plus value components and floor,
pseudonymisation, value-at-risk ordering beats random, survival probabilities ordered and bands
cover everyone, worklist has one row per customer with no raw id).

## Phase 4: cross-sell engine

### Run 2026-11-18, seed 42, 5,000 customers

`python -m pulse.cli train` now also fits one propensity model per product and writes
`models/crosssell_report.md` (45 s on top of the churn and survival models).
`python -m pulse.cli score` adds recommendation, rec_score, rec_reason and rec_source to the
worklist.

Worklist at 2026-10-31, 4,323 customers, recommendations by source: retention 432,
rule+model 1,455, rule 337, model 483, fallback 1,616, none 0.

First run, before the validation gate (kept for the record): the fixed_deposit and
personal_loan models scored ROC-AUC 0.537 and 0.441 on the test block, chance or worse,
with validation ROC-AUC 0.63 and 0.55. A gate of 0.65 on the validation block now drops
them to rule only. Everything below is after that gate.

Everything below is pasted from `models/crosssell_report.md`, which the train step writes.


**Propensity models per product (test block; rule only where uptakes are too few)**

| product | train_rows | train_positives | status | PR-AUC | ROC-AUC | capture@10% | lift@10% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| education_savings | 27127 | 506 | model (99 rounds, 2s) | 0.047 | 0.768 | 27.3 | 2.73 |
| fixed_deposit | 27443 | 201 | rule only (no signal: validation ROC-AUC 0.63) |  |  |  |  |
| fx_savings | 27230 | 386 | model (147 rounds, 2s) | 0.074 | 0.807 | 60.8 | 6.08 |
| mobile_banking | 11465 | 219 | model (53 rounds, 1s) | 0.186 | 0.917 | 52.1 | 5.21 |
| personal_loan | 26370 | 334 | rule only (no signal: validation ROC-AUC 0.55) |  |  |  |  |
| savings_account | 20068 | 117 | model (520 rounds, 5s) | 0.13 | 0.886 | 40.4 | 4.04 |
| sme_package | 27354 | 293 | model (221 rounds, 4s) | 0.046 | 0.766 | 49.7 | 4.97 |

**Triggers: fires at the latest snapshot (2026-10-31) and 90-day uptake on the test block**

| trigger | product | fires_latest | fires_latest_% | test_triggered | uptake_triggered_% | uptake_untriggered_% | lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diaspora_remittances | fx_savings | 367 | 8.5 | 1236 | 7.36 | 0.53 | 13.9 |
| school_fees | education_savings | 932 | 21.6 | 2989 | 5.45 | 0.6 | 9.1 |
| supplier_payments | sme_package | 600 | 13.9 | 1992 | 5.12 | 0.42 | 12.2 |
| rising_salary_idle_balance | fixed_deposit | 590 | 13.6 | 1627 | 1.6 | 0.46 | 3.5 |

`python -m pulse.cli pipeline`: ingest 21.9 s, features 42.0 s, train 77.1 s, score 13.2 s, total 154 s.

`python -m pytest`: 36 passed (31 from earlier phases, plus trigger fires and skips held products,
a new trigger from YAML alone, unknown operator rejected, every customer gets an action and the
guardrail wins, triggered customers take up more on the test block).

## Phase 5: reason codes and action cards

### Run 2026-11-18, seed 42, 5,000 customers

`python -m pulse.cli score` now also writes `data/processed/cards.json`, `shap_latest.parquet`,
`segments.parquet` / `segments.csv` and `models/shap_summary.md` (about 10 s on top of the
worklist). `python -m pulse.cli explain` rebuilds them from a saved worklist.

SHAP values come from LightGBM's built-in TreeSHAP (`predict(pred_contrib=True)`), since the
`shap` package cannot load on the build machine; a test checks contributions + base value equal
the raw score.

Cards at 2026-10-31, 4,323 customers: reasons per card 3: 1,040 / 2: 1,643 / 1: 1,391 / 0: 249
(the 249 have median risk 0.5%, max 5.6%, none in the act-this-week band).
Action family: leakage 1,827, income_disruption 742, balance_drain 651, activity_drop 490,
default 249, friction 186, new_customer 178.

Planted check (`_truth.csv`): 138 salaried churners with a planted salary stop; 137 show
`months_since_salary >= 1` at their last snapshot before the churn month; 120 of those 137 cards
(88%) name the salary in a reason. Before the salaried template variants: 0 of 137.

**Global SHAP summary, 4,323 customers at 2026-10-31** (base value -1.461 log-odds)

| feature | mean_abs_shap | pushes_up_% | share_% | family |
| --- | --- | --- | --- | --- |
| bal_change_vs_6m | 0.2007 | 22.4 | 6.9 | balance_drain |
| income_change | 0.1924 | 37.2 | 6.6 | income_disruption |
| remit_amount_3m_usd | 0.1847 | 86.8 | 6.4 | life_event |
| school_amount_12m_usd | 0.1393 | 82.0 | 4.8 | (no template) |
| txn_3m_vs_prior_3m | 0.1261 | 69.1 | 4.4 | activity_drop |
| fees_3m_usd | 0.1178 | 30.9 | 4.1 | friction |
| n_credits_1m | 0.1135 | 43.8 | 3.9 | income_disruption |
| bal_min_3m_ratio | 0.1100 | 28.1 | 3.8 | balance_drain |
| bal_slope_6m | 0.1029 | 14.6 | 3.6 | balance_drain |
| pos_share_3m | 0.0986 | 39.0 | 3.4 | (no template) |
| atm_share_3m | 0.0980 | 69.6 | 3.4 | (no template) |
| tenure_months | 0.0844 | 46.0 | 2.9 | new_customer |
| wallet_share_1m | 0.0785 | 27.6 | 2.7 | leakage |
| txn_count_change | 0.0746 | 28.6 | 2.6 | activity_drop |
| salary_growth_6m | 0.0746 | 52.0 | 2.6 | income_disruption |

**Segment view** (`data/processed/segments.csv`; product rows count a customer once per product held)

| dimension | group | customers | churn_rate_% | expected_churners | value_12m_usd | value_at_risk_usd | share_of_var_% | var_per_customer_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| branch | Harare CBD | 938 | 2.72 | 25.5 | 221032 | 4403 | 19.7 | 4.69 |
| branch | Bulawayo | 687 | 2.96 | 20.3 | 155266 | 4308 | 19.3 | 6.27 |
| branch | Gweru | 363 | 4.19 | 15.2 | 91882 | 2541 | 11.4 | 7.00 |
| branch | Masvingo | 346 | 2.99 | 10.3 | 83097 | 2406 | 10.8 | 6.95 |
| branch | Borrowdale | 536 | 2.06 | 11.0 | 131614 | 2027 | 9.1 | 3.78 |
| branch | Avondale | 407 | 3.31 | 13.5 | 95048 | 1977 | 8.8 | 4.86 |
| branch | Kwekwe | 202 | 3.15 | 6.4 | 54690 | 1381 | 6.2 | 6.84 |
| branch | Mutare | 416 | 1.96 | 8.1 | 97344 | 1345 | 6.0 | 3.23 |
| branch | Victoria Falls | 215 | 3.38 | 7.3 | 54010 | 1072 | 4.8 | 4.99 |
| branch | Chinhoyi | 213 | 2.49 | 5.3 | 51360 | 913 | 4.1 | 4.29 |
| segment | salaried | 1969 | 2.81 | 55.4 | 368354 | 8027 | 35.9 | 4.08 |
| segment | sme | 663 | 2.43 | 16.1 | 381961 | 7343 | 32.8 | 11.08 |
| segment | informal | 1251 | 3.50 | 43.8 | 227039 | 6328 | 28.3 | 5.06 |
| segment | diaspora | 440 | 1.73 | 7.6 | 57990 | 675 | 3.0 | 1.53 |
| product | personal_loan | 156 | 3.79 | 5.9 | 51900 | 1362 | 6.1 | 8.73 |
| product | sme_package | 281 | 1.69 | 4.8 | 145626 | 2103 | 9.4 | 7.48 |
| product | savings_account | 1285 | 2.64 | 33.9 | 381995 | 6566 | 29.3 | 5.11 |
| product | transactional_account | 4307 | 2.57 | 110.7 | 1031323 | 19425 | 86.8 | 4.51 |
| product | mobile_banking | 2532 | 2.25 | 57.1 | 619679 | 10505 | 47.0 | 4.15 |
| product | education_savings | 477 | 1.70 | 8.1 | 115339 | 1510 | 6.8 | 3.17 |
| product | fixed_deposit | 171 | 1.43 | 2.5 | 44593 | 511 | 2.3 | 2.99 |
| product | fx_savings | 385 | 1.40 | 5.4 | 88375 | 1005 | 4.5 | 2.61 |

Pitch rows: SMEs are 15% of customers and 33% of value at risk (11 USD per customer vs 4 to 5);
Gweru churns at 4.2% against 2.6% overall; diaspora customers barely churn (1.7%).

Pipeline (`python -m pulse.cli pipeline`): ingest 25.7 s, features 48.0 s, train 79.5 s,
score 26.2 s, total 179.5 s.

Tests (`python -m pytest`): 42 passed (6 new in `tests/test_explain.py`).

## Phase 6: scoring API, monitoring and Docker

### Run 2026-11-19, seed 42, 5,000 customers

`python -m pulse.cli pipeline` now ends with `monitor`: ingest 24.3 s, features 46.3 s,
train 69.4 s, score 29.8 s, monitor 2.8 s, total 172.6 s.

**API** (`api/main.py`, `uvicorn api.main:app`): `/health`, `/worklist`, `/customers/{ref}/card`,
`/segments`, `/metrics`, `POST /simulate`, `POST /reload`; Pydantic response models; reads only
pipeline outputs, pseudonymised references only.

`POST /simulate` at 2026-10-31 with the defaults (top_n 500, 2 USD per contact, 10 USD per save,
save rate 30%):

| outreach | expected churners contacted | expected saves | revenue protected USD | total cost USD | net return USD | return per USD |
| --- | --- | --- | --- | --- | --- | --- |
| targeted (top 500 by value at risk) | 85.1 | 25.5 | 4,617 | 1,255 | +3,361 | 3.68 |
| random (500) | 14.2 | 4.3 | 776 | 1,043 | -266 | 0.75 |

Uplift of targeted over random: +3,628 USD net on 500 calls.

**Monitoring** (`src/pulse/monitor.py`): run log `data/processed/runs.jsonl` (one JSON line per
run); drift report `models/drift_report.md` and `data/processed/drift.parquet`, PSI per feature,
latest snapshot (2026-10-31, 4,323 rows) vs the training block (to 2025-10-31, 27,646 rows), 10
quantile bins fixed on training. Result: 56 stable (< 0.10), 10 watch (0.10 to 0.25), 3
investigate (>= 0.25).

| feature | psi | status | train_mean | latest_mean |
| --- | --- | --- | --- | --- |
| bal_change_vs_6m | 0.479 | investigate | 1.164 | 1.088 |
| bal_slope_6m | 0.476 | investigate | 0.046 | 0.025 |
| salary_growth_6m | 0.389 | investigate | 1.071 | 1.088 |
| bal_avg_6m_usd | 0.218 | watch | 1472.6 | 2390.4 |
| digital_share_change_6m | 0.200 | watch | 0.009 | -0.002 |
| bal_avg_3m_usd | 0.191 | watch | 1589.7 | 2500.1 |
| bal_end_usd | 0.182 | watch | 1654.4 | 2566.4 |
| bal_min_3m_ratio | 0.180 | watch | 0.979 | 0.962 |
| school_terms_12m | 0.178 | watch | 0.586 | 0.825 |
| fees_1m_usd | 0.127 | watch | 5.523 | 5.586 |

Reading: the synthetic generator grows balances through the whole span (average 6-month balance
1,253 USD at the first snapshot, 2,390 USD at the last) and relative growth was faster in 2025,
so the balance-change features really have shifted since training; `salary_growth_6m` is a
short-history artefact (15% of training rows carry the fill value 1.0, 0.2% now).

Tests (`python -m pytest`): 51 passed (9 new in `tests/test_api.py`).

**Docker** (20 Nov 2026, offline build from `wheels/`, 72 wheels, 208 MB):

| check | result |
| --- | --- |
| `docker compose build --build-arg PIP_FLAGS="--no-index"` | ok, 7.5 min, image `pulse:latest` 1.81 GB |
| `docker compose up` with pipeline outputs in `data/` | api healthy in 3 s; `/health` 4,323 customers; `/worklist?limit=3` ranks 1-3 P57DB131092, PED90361E77, P5D159A7254; `/customers/P57DB131092/card` 200 with 3 reasons; unknown ref 404; `/segments`, `/simulate` (uplift 3,627.76 USD), `/metrics` ok; dashboard placeholder 200 on :8501 |
| clean checkout (empty `data/`, `models/`) | first run killed (Docker VM 3.75 GB with 2.2 GB held by unrelated containers), second run crashed writing `docs/features.md` (folder not in the image; fixed in `features/build.py`), third run ok after the VM was raised to 5 GB: data 80.6 s, pipeline 149.9 s (ingest 15.1, features 28.3, train 86.3, score 18.9, monitor 1.2), healthy 4 min after start, `/health` 4,323 customers, `/worklist` rank 1 P57DB131092 (same as the host run), card 200; container peak memory 2,179 MiB |

Online build on the same day failed twice on a 2 to 10 kB/s connection (pip read timeout after
backtracking through matplotlib releases); the wheel cache path is the one to rely on.

## Phase 7: dashboard and ROI simulator

### Run 2026-11-20, same pipeline outputs as Phase 6 (snapshot 2026-10-31, 4,323 customers)

**Pages** (`dashboard/app.py`, Streamlit 1.67, served by the dashboard container on :8501):
Worklist, Customer card, Insights, ROI simulator. Screenshots at 1280x720 in `docs/screens/`.

**API additions** (`api/main.py` 0.7.0): `GET /customers/{ref}/history` (12 month-ends: balance,
money in, salary credit), `GET /drivers` (SHAP share by reason family plus top five readings),
`POST /simulate` now accepts `top_n` from 0 to the whole book.

**Headline numbers on the worklist** (all customers): 98 customers in the two "act" bands;
22,374 USD total value at risk; 67% of it held by the top 433 customers (10% of the ranking).

**Churn drivers by family** (`GET /drivers`, share of mean absolute SHAP at the latest snapshot):

| driver | share of model impact (%) |
| --- | --- |
| Balance drain | 28.3 |
| Income disruption | 26.1 |
| Activity drop | 13.7 |
| Life event | 12.6 |
| Fees and friction | 8.1 |
| New relationship | 5.8 |
| Money leaving to wallets or other banks | 5.4 |

**ROI simulator at the defaults** (2 USD per contact, 10 USD per save, 30% save rate):

| customers contacted | Pulse net return USD | random net return USD | uplift USD |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 435 (peak of the Pulse line) | 3,368 | -232 | 3,600 |
| 500 (default) | 3,361 | -266 | 3,628 |
| 3,059 (Pulse line turns negative) | 0 | -1,629 | 1,629 |
| 4,323 (everyone) | -2,303 | -2,303 | 0 |

**Page render time** (Streamlit AppTest inside the container; budget 2 s):

| page | first render s | cached render s |
| --- | --- | --- |
| Worklist | 4.43 | 0.07 |
| Customer card | 0.86 | 0.10 |
| Insights | 1.25 | 0.14 |
| ROI simulator | 1.16 | 0.11 |

**Tests**: 63 in total. 59 pass in the api container (`tests/test_dashboard_app.py` skips there,
counted as 1) and the 4 page tests pass in the dashboard container against the live api. New in
this phase: 3 API tests (history, drivers, simulate extremes), 5 dashboard logic tests, 4 page
tests. The local venv can no longer import scikit-learn (Smart App Control), so the suite runs
with `docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps api python -m pytest -q`.

## Phase 8: rehearsal datasets, fallback mode, runbook (21 Nov 2026)

**Rehearsal datasets** (`scripts/make_rehearsal.py`, seed 2026): `data/rehearsal/a/` three files,
100 MB; `data/rehearsal/b/` one file, 0.3 MB. Both verified privately end to end (ingest to API)
through a schema file alone, no code change. Numbers withheld on purpose; the rehearsal log in
`docs/runbook.md` is where yours go.

**Fallback mode on the module test table** (`tests/test_summary_mode.py`, 1,200 customers, planted
signal in tenure, activity, balance and complaints): stratified split 720 / 240 / 240 rows with
equal churn rates; logistic and LightGBM only (rules baseline needs transaction features);
test ROC-AUC above 0.6; deterministic across two runs; fairness table on gender and age band;
cards for more than 80% of customers from `config/reasons_summary.yaml`; API serves worklist,
card (history 404), drivers, metrics with `mode: summary` and `drift: null`, simulate.

**Pipeline in transaction mode** unchanged: 150 to 181 s on this laptop.

**Tests**: 66 in total. 61 pass in the api container plus 1 skipped module (the 4 page tests,
which pass in the dashboard container). New: 2 in `tests/test_summary_mode.py` (mode detection,
fallback pipeline end to end).

| run | command | result |
| --- | --- | --- |
| whole suite | `docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps api python -m pytest -q` | 61 passed, 1 skipped, 89 s |
| page tests | `... run --rm dashboard python -m pytest -q tests/test_dashboard_app.py` | 4 passed, 7 s |
