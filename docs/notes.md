# Pulse notes

My explanations of what was built and why, one section per phase. Written so I can
defend each decision to a bank's data team.

## Phase 0: setup and stand-in data (17 Nov 2026)

### Environment

- Python 3.11.9 from python.org (installed with `winget`). The `uv`-managed Python was
  blocked by Windows Application Control, so we use the signed official build.
- Virtual environment in `.venv`. Every package is pinned in `requirements.txt` to the
  version that actually installed on this machine on 17 Nov 2026.
- There is no `make` on Windows. Every target in the Makefile is just
  `python -m pulse.cli <step>`, so run that instead.

### Why synthetic data

We have not seen BancABC's data. Hunting for a public bank dataset would give us
something with different columns and no Zimbabwean signals. Generating our own means:

1. We can plant exactly the behaviours the proposal promises (salary stops, wallet
   leakage, school fees, remittances) and later check the pipeline finds them.
2. We know the true answers (`data/raw/_truth.csv`), so Phase 1 and 2 results can be
   checked against them by hand. The pipeline never reads this file; a test proves it.
3. Swapping to the real data on 4 December means editing `config/schema.yaml` only.

### How the generator makes data realistic but not trivially easy

The generator (`src/pulse/synth.py`) works in three layers.

**Layer 1, persona.** Each customer gets a segment (salaried, SME, informal,
diaspora-supported) which sets how they bank: salaried customers have a monthly RTGS
salary from a named employer and spend by card and EcoCash; SMEs receive many ZIPIT and
wallet payments from customers and pay suppliers; informal traders move many small
amounts through USSD wallets; diaspora-supported customers live on monthly Mukuru or
Western Union remittances. Each customer also draws life events: regular remittances,
school fees in January, May and September, steady supplier payments, a rising salary.
About 30% hold a second savings account, and roughly a third of main accounts are in
ZiG rather than USD, with ZiG amounts drifting upward each month.

**Layer 2, hidden fate.** About 10% of customers are chosen to churn, on a random month
after at least six months of history. Only a fifth of them formally close the account;
the rest just go quiet. Each churner independently draws warning signs, each with its
own probability, so churners look different from each other:

| Sign | Chance among warned churners | What it looks like |
| --- | --- | --- |
| income stops or shrinks | 70% (if they have salary or remittances) | salary disappears, or drops to 30 to 60% |
| balance drains | 70% | 30% to 80% of the balance leaves each month for 2 to 4 months |
| leakage rises | 60% | share of outflows to wallets and other banks climbs by up to 35 points |
| activity drops | 75% | transaction count falls by up to 65% |

The signs start 2 to 4 months before the churn date and strengthen as it nears. To stop
any sign being a perfect rule:

- about 12% of churners show no warning at all and simply vanish;
- about 8% of non-churners (false alarms) show the same signs for 1 to 3 months and then
  recover;
- about 10% of salaried non-churners change employer, sometimes with a one-month salary
  gap, which looks like "salary stopped" for one month;
- 3% of salary months are simply late or missed, for everyone;
- school fees and supplier runs can dwarf a month's other spending, which hides leakage
  in that month (noise that is also realistic).

**Layer 3, statement lines.** The plan says how much money moves per customer per month;
the generator then splits it into individual dated transactions with channel,
counterparty type and a statement-style description (`POS OK ZIMBABWE HARARE`,
`ECOCASH TRANSFER 07...`, `ZIPIT TO CBZ`, `SCHOOL FEES GATEWAY HIGH`, `RTGS FEE`).
Spending comes out of the balance that was there at the start of the month, so the
running balance never goes negative. Interbank transfers get fee lines, 1% of debits get
a reversal credit, closed accounts end with a closure withdrawal to zero. Life events
raise the chance of later taking the matching product (for example 35% versus 4% for
foreign-currency savings), with a delay, never with certainty.

Two small departures from the brief, both deliberate: `counterparty_type` also contains
`cash` (ATM and branch cash) and `self` (transfers between a customer's own accounts),
because real statements have those and there was no honest place for them in the list.

### Schema mapping and ingest

`config/schema.yaml` maps canonical names to source columns, per file, with a date
format and optional value mappings (for example `CR` to `credit`). `src/pulse/ingest.py`
reads each CSV as text, renames and casts through the mapping inside DuckDB, and runs
checks: nulls per column, duplicate keys, date ranges against `settings.yaml`,
orphan foreign keys, transactions outside the account's life, currency mismatches,
negative balances. Hard problems (duplicate or missing keys, unknown columns, orphans)
stop the run; soft ones are printed as warnings. The report is saved to
`data/processed/dq_report.txt`.

### What a data team might ask

- *Is 10% churn realistic?* It is an assumption in `settings.yaml`. Real retail churn
  in Zimbabwe may be higher. The point is the mechanism, not the rate.
- *Why no random split anywhere?* All signals are time-ordered on purpose. Phase 1 will
  build features strictly before a snapshot date and labels strictly after.
- *Could the model just learn the generator?* Partly, which is why the noise layers
  exist and why Phase 8 rehearses on data with different shape. The synthetic data proves
  the pipeline works end to end; the real data on the day gives the real numbers.

## Phase 1: Signal Factory and churn target (17 Nov 2026)

### The snapshot table in one picture

One row per customer per month-end. Features on a row use only transactions dated on or
before that month-end. The label on the row uses only the 90 days after it. Column m of
every internal array holds what happened IN month m; a snapshot at the end of month m
reads columns <= m for features and columns > m for the label. There is no way for the
two to share a month, and `tests/test_features.py` proves it: it rewrites, deletes and
adds transactions after a snapshot date and asserts every feature at or before that date
is byte-for-byte unchanged, while the labels do change.

### How the data flows

1. `features/monthly.py` collapses 1.8 million transactions into (5,000 customers x 24
   months) arrays: counts, active days, money in and out by counterparty type, channel
   counts, the largest credit of the month and its payer, and the month-end balance per
   account carried forward over quiet months and converted to USD.
2. Each feature family (`income`, `balance`, `leakage`, `engagement`, `friction`,
   `life_events`) is a pure function from those arrays to new (customers x months)
   arrays, built with rolling-window helpers that only ever look left.
3. `target.py` computes the label the same way, looking only right.
4. `features/build.py` picks the eligible cells and writes one row each.

### The churn label, exactly

`churn_90d = 1` when, in the 90 days after the snapshot, the customer either

- **hard churn**: an account close date falls inside the window, or
- **silent churn**: their customer-initiated transactions per month (bank fee lines do
  not count) AND their average month-end balance over the next three calendar months
  both fall below 20% of their own average over the six months up to the snapshot.

The 20%, 90 days and 6 months live in `settings.yaml` under `target`. Silent churn uses
three calendar months (89 to 92 days) because activity is aggregated monthly; hard churn
uses exact days. `days_to_churn` is exact for closures and the end of the first quiet
month for silent churn.

Rows are dropped when the customer has under six months of data, has already closed,
has fewer than three customer-initiated transactions in the previous three months
(already inactive: predicting churn for them is trivial and useless), or when fewer
than 90 days of data follow (those rows keep their features for scoring but have an
empty label and `labelled = 0`).

### Salary without labels

The salary detector uses only the pattern: the largest non-own-account credit each
month, the median of those over the last 12 months as the reference, and "a salary
month" means a credit within 30% of that reference. Salary detected = at least 5 such
months in 12 or 4 in 6. `months_since_salary`, `salary_change_1m` and `payer_changed`
(the payer key is the description with digits stripped) all derive from this, so they
will work on a statement that never says "SALARY".

### Three features and why each should predict churn

- `months_since_salary`: income is the root of most silent churn. At the snapshot before
  churn, planted income-stop churners sit at a median of 2 months, everyone else at 0.
- `bal_change_vs_6m`: a balance at 21% of its own six-month average (planted drainers)
  versus 1.1 to 1.2 for everyone else. Money leaving ahead of the customer.
- `leak_share_trend`: the share of spending going to wallets and other banks rose by 15
  points for planted leakage churners and 0 for others. The customer is already banking
  elsewhere.

### What I had to fix in the generator, and why it matters

Building the label exposed three generator bugs that made silent churn invisible:
the opening balance was drawn twice with different random numbers, the overdraft
cushion was measured without the opening balance, and reversal credits and fee lines
drifted the statement away from the plan. All three left a hidden floor under balances
so they never fell to 20%. After the fixes, 91% of planted silent churners and 100% of
planted closures are labelled at the snapshot before their churn date, and no customer
who was not planted to churn is ever labelled 1. The 9% that are missed are mostly
customers who keep shuffling money between their own accounts: by the definition they
are not gone, and that is the right call.

### What a data team might ask

- *Why is the row-level churn rate 3%, not 10%?* 15% of customers churn across 24
  months; each one is labelled 1 on the two or three snapshots whose 90-day window
  contains the churn. That gives about 3% of customer-months, rising through the
  period because planted churn cannot start before month 6.
- *Why no feature for gender or age?* Rule 6. They are not in the table at all; the
  fairness check in Phase 2 joins them back from the customers table.
- *Why USD?* ZiG and USD balances must be comparable. One rate from `settings.yaml`
  for now; the bank's own rate table can replace it.
- *What about pyarrow and SHAP?* Both native libraries are blocked by Windows
  Application Control on the build machine. Parquet goes through DuckDB. Phase 5 will
  use LightGBM's built-in SHAP values instead of the shap package.
