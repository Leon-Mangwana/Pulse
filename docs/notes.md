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

### The churn label, exactly (corrected in Phase 2, see below)

`churn_90d = 1` when, in the 90 days after the snapshot, the customer either

- **hard churn**: an account close date falls inside the window, or
- **silent churn**: the customer goes quiet within the next three calendar months. "Quiet"
  is an event with a month, E: the first month in which customer-initiated transactions
  (bank fee lines do not count) AND the month-end balance both fall below 20% of the
  customer's own average over the six months before E, and stay below it the following
  month too (one quiet month alone could be a holiday).

The 20%, 90 days and 6 months live in `settings.yaml` under `target`. Silent churn uses
three calendar months (89 to 92 days) because activity is aggregated monthly; hard churn
uses exact days. `days_to_churn` is exact for closures and the end of the first quiet
month E for silent churn. Because the quiet month needs one more month to confirm it,
the last labelled snapshot is four months before the end of the data.

Rows are dropped when the customer has under six months of data, has already closed,
has fewer than three customer-initiated transactions in the previous three months
(already inactive: predicting churn for them is trivial and useless), is at or past
their quiet month E (already churned: nothing left to predict), or when the label
window is not yet observable (those rows keep their features for scoring but have an
empty label and `labelled = 0`).

The first version of this label (Phase 1) compared the *average* of the next three
months with the baseline and had no "already churned" exclusion. That let the label
keep firing for two or three snapshots after a customer had gone, so a model could
"predict" churn that had already happened. Phase 2 found it; details there.

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

- *Why is the row-level churn rate 3%, not 15%?* 15% of customers churn across 24
  months; each one is labelled 1 on the three snapshots whose 90-day window contains
  the churn, and never after it. That gives about 3% of customer-months, rising through
  the period because planted churn cannot start before month 6.
- *Why no feature for gender or age?* Rule 6. They are not in the table at all; the
  fairness check in Phase 2 joins them back from the customers table.
- *Why USD?* ZiG and USD balances must be comparable. One rate from `settings.yaml`
  for now; the bank's own rate table can replace it.
- *What about pyarrow and SHAP?* Both native libraries are blocked by Windows
  Application Control on the build machine. Parquet goes through DuckDB. Phase 5 will
  use LightGBM's built-in SHAP values instead of the shap package.

## Phase 2: churn model and honest evaluation (18 Nov 2026)

### The three ideas, in plain words

**PR-AUC.** Only 3 in 100 customer-months are churn. A model that says "nobody churns"
is 97% accurate and useless, so accuracy is never quoted. PR-AUC asks instead: as you
walk down the list from the riskiest customer, how pure is the top of the list (precision)
and how many of all the churners have you reached (recall)? It averages precision over
every level of recall. A random ranking scores the churn rate itself (0.03); 1.0 is perfect.
It rewards getting the rare churners to the top, which is the only thing we care about.

**Capture and lift at the top 10%.** The bank cannot call everyone. If it calls the 10%
the model ranks riskiest, capture is the share of all real churners it reaches; lift is
capture divided by 10%, so lift 7 means seven times better than calling at random.
This is the number for the pitch because it matches what a branch can actually do.

**Calibration.** A score of 0.30 should mean that about 30 in 100 such customers churn.
Phase 3 multiplies the probability by the customer's expected value in dollars, so an
inflated probability is inflated money. The model is trained with churners weighted up
(class weights), which deliberately inflates its raw scores; an isotonic calibrator fitted
on the validation block maps raw scores back to honest rates. The calibration table
checks it: rows sorted by score, cut into ten equal groups, predicted versus actual.

### What the first run taught us: a label bug, not leakage

The first training run gave ROC-AUC 0.952 and the hand-written rules alone captured 88%
of churners in the top 10%. Per the rules in CLAUDE.md that is a reason to investigate,
not celebrate. The leakage test still passed, so features were clean. Joining test
predictions to `_truth.csv` (diagnostics only) showed the real cause: half of all rows
labelled 1 were dated *after* the planted churn date. The Phase 1 label compared the
average of the next three months with the customer's baseline, and once a customer had
gone quiet the label kept firing for two or three more snapshots, because the six-month
baseline still remembered the good months. The model was "predicting" churn that had
already happened, which is why a single balance feature carried half of the signal.

The fix (`src/pulse/target.py`): churn is an event with a month E, the first of two
consecutive quiet months; a snapshot is labelled 1 if E falls inside its 90-day window;
snapshots at or after E are excluded, exactly as already-inactive customers are. After
the fix the median lead between a labelled row and the planted churn date is 32 days,
churners planted with no warning sign score near chance (0.585), and no single feature
exceeds 0.81 alone. Those are the signs of an honest problem.

### The split

| block | snapshots | rows | churn |
| --- | --- | --- | --- |
| train | 2025-04-30 to 2025-10-31 (7) | 27,646 | 2.5% |
| validation | 2025-11-30 to 2026-01-31 (3) | 12,360 | 2.9% |
| gap | 2026-02-28, 2026-03-31 | dropped | |
| test | 2026-04-30 to 2026-06-30 (3) | 12,814 | 3.2% |

The gap exists because a snapshot's label looks three months ahead: a row dated 31 March
knows April to June, which is the test period. Dropping those two months means nothing
the model was fitted on overlaps what it is judged on. The same customer can appear in
train and test at different dates; that is how the bank will use it.

### The three models

- **Rules**: count of four flags (salary gone two months, balance under half its
  average, transactions under half, wallet and other-bank share up 15 points). A branch
  manager could run it in Excel; the model must beat it to earn its keep.
- **Logistic regression**: one weight per feature, features clipped at the 1st and 99th
  percentile and standardised, churners weighted up.
- **LightGBM**: many small trees, each correcting the last. Four fixed candidate settings
  were fitted with early stopping on the validation block; the least overfit one within
  0.01 of the best validation PR-AUC was kept (7 leaves, 200 rows per leaf, 159 rounds).
  The test block was scored once, after that choice.

### Results on the test block

| model | PR-AUC | ROC-AUC | capture@10% | lift@10% |
| --- | --- | --- | --- | --- |
| rules | 0.437 | 0.844 | 67.3% | 6.7 |
| logistic | 0.450 | 0.856 | 68.3% | 6.8 |
| LightGBM | 0.512 | 0.875 | 71.7% | 7.2 |

Calibration: raw LightGBM scores average 24% against a true rate of 3.2%; after the
isotonic calibrator the average is 3.3% and the expected calibration error is 0.003.
The riskiest decile is predicted at 24.3% and churns at 23.1%.

Sanity checks: no single feature above 0.9 alone (highest 0.81). Train versus test
PR-AUC is 0.624 versus 0.512, a gap of 0.11, just over the 0.10 warning line; the more
flexible candidate had a gap of 0.26 for the same test score, which is why the simpler
one was kept. Top features: balance versus its six-month average, balance slope,
lowest recent balance, income change, net flow, wallet share, transaction trend. Balance
first, then income, then wallet leakage: the order the proposal predicted.

Fairness at the top-10% cut: flag rates by gender are 10.0% and 10.0%; capture 75% (F)
versus 69% (M). Across age bands the flag rate follows each band's churn rate (7% for
55 to 64, who churn least, up to 10.8% for 45 to 54) and capture ranges from 66% to
78%. Nothing is flagged disproportionately to its churn rate. Gender and age band were
joined after scoring and are not in the feature list; a test asserts it.

### The sentence for a banker

"Calling the riskiest 10% of customers reaches 72% of those who actually leave in the
next 90 days, seven times better than calling at random." Say it with the caveat that
this is synthetic data built to contain warning signs; the real figure comes from the
bank's own history on the day.

### What a data team might ask

- *Why did the rules baseline do so well?* The synthetic churners carry planted signs
  that the rules were written to catch. On real data the rules will fall further behind;
  LightGBM's advantage comes from combining weak signs, which is what real churn needs.
- *Why is PR-AUC only 0.51 if ROC-AUC is 0.88?* With 3% positives, PR-AUC is the harder
  score; 0.51 against a random baseline of 0.03 is strong. Quote capture, not PR-AUC.
- *Why not more tuning?* Four candidates differed by 0.016 in validation PR-AUC. More
  search would chase noise on 364 validation churners.
- *Why calibrate on validation, not test?* The test block is scored once, after every
  choice has been made, so its numbers are what the bank would see next quarter.

## Phase 3: value at risk and time-to-churn (18 Nov 2026)

### Censoring and Cox, in everyday terms

Watch a bus stop for 90 minutes. Some people board a bus, and you know exactly when.
When you leave, others are still waiting: you only know they waited more than 90 minutes.
Those people are censored. Dropping them says "everyone leaves within 90 minutes";
counting them as leaving at minute 90 says they left when they did not. A survival model
keeps them in, using only the fact that they lasted at least 90 minutes. A Cox model then
asks, for each trait, how much it multiplies the rate of leaving at any moment: "a
customer whose balance has halved leaves at a higher rate than one whose balance is
steady, at every point in time". The shape of the timing curve comes from the data; the
traits scale it up or down. For us the clock starts at the snapshot, the event is churn at
`days_to_churn`, and anyone still here at 90 days is censored at 90.

### Value: three lines of arithmetic, every number in settings.yaml

`value_12m_usd` = fees paid in the last 3 months x 4, plus the 6-month average balance x
a yearly net interest margin (5% on USD, 8% on ZiG, blended by the customer's USD share),
plus a yearly margin per product held at the snapshot date (loan 120, SME package 150,
fixed deposit 40, foreign-currency savings 25, savings 15, mobile banking 6, transactional
account 0 because its value is already in fees and balances). Floored at 1 USD. On the
test block the median customer is worth 156 USD a year, the top 10% over 424 USD.

These are placeholders for finance to replace. The code reads them from `settings.yaml`
under `value`; nothing else needs to change. Products the data mentions but the settings
do not are counted as 0 and listed in a warning.

### Value at risk and the worklist

`value_at_risk_usd` = calibrated 90-day churn probability (Phase 2) x 12-month value.
`python -m pulse.cli score` writes `data/processed/worklist.parquet`: one row per customer
at the latest snapshot (31 Oct 2026, 4,323 customers) with the probability, the value and
its three parts, value at risk, rank, the Cox 30/60/90-day probabilities, the urgency band
and four "why" features. Customer ids are replaced by a salted hash (`pulse/pseudo.py`);
the mapping back is in `id_map.parquet`, which the API and dashboard never read.

Total value at risk at the latest snapshot: 22,374 USD of 1,035,344 USD annual value
(2.2%). The top 20 are SMEs and salaried customers worth 160 to 840 USD a year whose
balance is at 0 to 16% of its 6-month average and whose transaction count is down 50 to
90%: valuable and clearly slipping, which is the test the plan set.

### Time-to-churn: what the Cox model adds, and what it does not

Fitted on the train and validation blocks together (40,006 rows, 1,067 events) on ten of
the strongest Phase 2 features, clipped and standardised, with a small ridge penalty.
Every coefficient points the expected way: balance versus its 6-month average, lowest
recent balance, transaction change and income change all lower the rate of leaving when
higher; wallet share and leakage trend raise it. Months since salary is not significant
once the balance and income features are in.

Concordance on the test block (chance 0.5): Cox 0.848, the LightGBM probability 0.885.
**The survival model ranks customers slightly worse than the churn score, so it adds
little for ranking.** What it adds is a time axis: a probability of churn within 30, 60
and 90 days, and the urgency band built from them. One honesty note: under proportional
hazards every customer shares the same curve shape, so the 30-day and 90-day
probabilities rank customers identically. The band is a risk tier expressed in days, not
an independent prediction of timing. The check that it still means something: test
churners who left within 30 days had a mean 30-day probability of 10.8%, those who left in
days 61 to 92 had 5.4%. Higher risk does mean sooner, roughly.

Bands on the test block, thresholds from `settings.yaml`: "act this week" (30-day
probability at or above 15%) is 0.9% of rows and 76% of them churned within 90 days;
"act this month" (90-day probability at or above 15%) is 2.1% of rows and 48% churned;
"watch" is the rest, 1.5% churned. The Cox 90-day probability averages 2.9% against an
actual 3.2%, so it is roughly calibrated on its own, but the LightGBM figure is the one to
quote and the one used for value at risk.

### The slide: three ways to order the call list (test block)

12-month value of the customers who actually churned, reached by contacting the top
share of the list. Total at stake: 81,894 USD across 413 churners.

| ordering | top 5% | top 10% | top 20% |
| --- | --- | --- | --- |
| random | 5.6% | 11.0% | 19.4% |
| by churn probability | 59.5% | 67.5% | 73.5% |
| by value at risk | 64.6% | 70.8% | 83.3% |

Value at risk reaches more dollars at every list size while reaching almost exactly the
same number of churners (64% versus 64% at 5%, 71% versus 72% at 10%). It trades a few
low-value churners for high-value ones. That is the whole argument for multiplying by
dollars.

### What a data team might ask

- *Is a 5% net interest margin right?* It is an assumption with a comment next to it. The
  point is that the worklist re-sorts itself when finance changes the number.
- *Why not use the Cox probability for value at risk?* The LightGBM probability is better
  ranked (0.885 versus 0.848) and was calibrated on held-out data; the Cox model is kept
  for the time axis only. If a judge asks whether the survival model earns its place: for
  ranking, no; for "this week or this month", yes, modestly.
- *Why can a probability be exactly 1.00 on the worklist?* The isotonic calibrator is a
  step function; its top step is the validation bin where every customer churned. On real
  data with more noise the top step will sit below 1.
- *Why is value at risk only 2.2% of annual value?* Because 3% of customers churn in a
  quarter. The figure is a quarter's exposure, not a year's.
