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
