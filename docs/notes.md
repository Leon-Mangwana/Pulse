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

## Phase 4: cross-sell engine (18 Nov 2026)

### Rules first, because they survive a different product list

`config/crosssell.yaml` holds the triggers. Each one is a product, a list of conditions on
features from `docs/features.md` (every condition must hold), a priority, and a reason
template whose placeholders are filled with the customer's own numbers. Adding a trigger
is adding a block; a test loads an extra trigger from YAML alone and checks it fires. A
product the customer already holds at the snapshot date is never offered; "holds" comes
from the products table's start and end dates. The same file carries the model-only
reason per product, the fallback chain (first product not held: mobile banking, savings,
fixed deposit, FX savings) and the retention template. Nothing shown to a user is
generated; every sentence is one of those templates (rule 8).

The four starting triggers and what they did:

| trigger | fires at 31 Oct 2026 | test-block uptake, triggered vs not | lift |
| --- | --- | --- | --- |
| remittances in 4 of 6 months, over 150 USD in 3 months, offer fx_savings | 8.5% | 7.4% vs 0.5% | 13.9x |
| school fees in 2+ months of 12, offer education_savings | 21.6% | 5.5% vs 0.6% | 9.1x |
| suppliers paid in 4 of 6 months, offer sme_package | 13.9% | 5.1% vs 0.4% | 12.2x |
| salary up 10%+, balance over 1,500 USD, positive net flow, offer fixed_deposit | 13.6% | 1.6% vs 0.5% | 3.5x |

"Uptake" means the product was started within 90 days after the snapshot, among customers
who did not already hold it. Triggered customers take the matching product at 3.5 to 14
times the rate of untriggered ones, which is the "done when" test.

### Propensity models: one per product, same recipe as churn, honest about failures

The label for a product at a snapshot is "started it within the next 90 days", rows
already holding it excluded. Each product gets a small LightGBM with the Phase 2 recipe:
same time split and gap, class weights, early stopping on validation, isotonic
calibration, the same metrics on the test block. Two gates decide whether the model is
used at all, both judged before the test block is touched: at least 100 uptake rows in
the training block, and a validation ROC-AUC of at least 0.65. The bar is above "beats
chance" because early stopping on the validation block inflates its score.

| product | train uptakes | result |
| --- | --- | --- |
| fx_savings | 386 | model, ROC-AUC 0.807, lift@10% 6.1 |
| education_savings | 506 | model, ROC-AUC 0.768, lift@10% 2.7 |
| sme_package | 293 | model, ROC-AUC 0.766, lift@10% 5.0 |
| savings_account | 117 | model, ROC-AUC 0.886, lift@10% 4.0 |
| mobile_banking | 219 | model, ROC-AUC 0.917, lift@10% 5.2 |
| fixed_deposit | 201 | rule only: validation ROC-AUC 0.63 |
| personal_loan | 334 | rule only: validation ROC-AUC 0.55 |

Two things to say plainly. First, the fixed-deposit and personal-loan models learned
nothing (their first versions scored 0.54 and 0.44 on the test block, worse than a coin),
so the fixed-deposit trigger stands alone and personal loans are never offered. The 0.65
gate was set after seeing those two fail; it is a sanity bar, not a tuned number. Second,
the mobile-banking model's 0.92 is suspiciously high: in the synthetic data customers use
the app channel before the product record starts, so the model is partly reading the
product off the channel mix. On real data expect that one to fall.

### One recommendation per customer

1. A triggered rule wins. If several fire, the higher model probability wins, then
   priority. Source `rule+model` when a model exists for the product, else `rule`.
2. No trigger: the best model probability wins if it is at least twice that product's
   base rate (source `model`).
3. Otherwise the first fallback product not held (source `fallback`).
4. Guardrail last: the top decile of value at risk gets a retention action instead of any
   offer (source `retention`), because selling to someone about to leave is how to lose
   them faster.

At 31 Oct 2026: 432 retention, 1,455 rule+model, 337 rule, 483 model, 1,616 fallback, 0
without an action. `rec_score` is the calibrated uptake probability when a model exists,
the rule's observed training uptake rate otherwise, and the value at risk for retention.

### What a data team might ask

- *Why do the models mostly agree with the rules?* Because the synthetic life events are
  the signal. The models earn their place where a rule is missing (savings account, mobile
  banking) and as a tie-breaker when several rules fire. On real data they may find
  patterns no one wrote a rule for; the gates will say whether they did.
- *Why are the uptake probabilities so small (2 to 7%)?* They are 90-day probabilities,
  calibrated. Quote the lift, not the level.
- *What happens on 4 Dec with the bank's product names?* Edit `crosssell.yaml`: rename
  the products, adjust the conditions to the features that exist, add triggers for
  products we have not seen. Products with no margin in `settings.yaml` are counted at 0
  and listed in a warning.

## Phase 5: reason codes and action cards (18 Nov 2026)

### What a SHAP value is, on the first card

The churn model adds up. For any customer its raw score (in log-odds, before calibration) is a
base value, the same for everyone, plus one contribution per feature. SHAP (TreeSHAP for tree
models) is the method that splits a tree model's prediction into those per-feature contributions
fairly, so that they always sum exactly to the prediction. A positive contribution pushed this
customer's score towards churn, a negative one pulled it away.

The customer ranked first at 31 Oct 2026 (SME, Bulawayo, calibrated risk 67%, 564 USD at risk):

| | log-odds |
| --- | --- |
| base value (everyone starts here) | -1.46 |
| bal_change_vs_6m: balance is 0% of its 6-month average | +1.80 |
| bal_min_3m_ratio: balance touched 0% of the average in the last 3 months | +0.62 |
| income_change: money in this month is 0% of the 6-month average | +0.54 |
| bal_drop_from_3m_peak: 100% of the 3-month peak balance has gone | +0.44 |
| net_flow_3m_usd: more out than in over 3 months | +0.36 |
| txn_count_change: 8 transactions against a usual 26 | +0.34 |
| 63 other features, mostly small negatives (little leakage to wallets or other banks) | -0.35 |
| raw score | 3.41 |

Three of the six positive features describe the same thing, the balance collapsing, so the card
keeps one per family: balance drain, income disruption, activity drop. The raw score of 3.41 would
be a 97% probability on its own; the isotonic calibrator from Phase 2 maps it to 67%, because the
model was trained with churners up-weighted and its raw outputs run hot. Calibration only
stretches the scale, it never reorders customers, so "which features pushed this customer up" is
unaffected by it.

The `shap` package itself cannot load on the build machine (Windows App Control blocks a compiled
scikit-learn file it imports). LightGBM has the same TreeSHAP algorithm built in, reached with
`Booster.predict(pred_contrib=True)`, and that is what `explain.py` uses. A test checks that the
contributions plus the base value reproduce the model's raw score to six decimals.

### Why the global SHAP summary and the Phase 2 gain table disagree

Gain (Phase 2) measures how much each feature improved the training loss when the trees were
built. Mean |SHAP| measures how much each feature actually moves scores on the customers being
scored today. `bal_change_vs_6m` tops both (38% of gain, 7% of mean |SHAP|), but features such
as `remit_amount_3m_usd` and `school_amount_12m_usd` rank high on mean |SHAP| while being minor
by gain. They push the score up for 80 to 90% of customers, by a small, uniform amount: in this
synthetic data customers with no remittances or school fees churn slightly more, so "having no
remittances" is a weak, diffuse risk factor. That is not something to tell a relationship
manager, so those features have no template except for the honest case (remittances used to
arrive and have stopped).

### Reason templates: fixed sentences, conditions, variants

`config/reasons.yaml` has one block per feature worth a sentence (36 of the 69 features). Each
block names a family, a template filled with the customer's own numbers, and conditions that must
hold for the sentence to be rendered, so a positive SHAP value on `months_since_salary` for a
customer whose salary did arrive this month produces nothing rather than a wrong sentence.
A block can hold ordered variants: the first whose conditions hold is used. `income_change` has
a salaried variant ("No salary-sized credit for 3 month(s), about 250 USD used to arrive
monthly; money coming in this month is 19% of the 6-month average") ahead of the generic one.
Features without a block are skipped, never described with generated text (rule 8). The three
aliases a template may use besides the features (`_pct`, `_drop_pct`, `_abs_pct`, plus the three
"share before the last 3 months" values) are computed in `explain.py` and listed at the top of
the YAML; nothing else is derived.

Per card: the three largest positive contributions that have a template, one per family. Result
at 31 Oct 2026, 4,323 cards: 1,040 with three reasons, 1,643 with two, 1,391 with one, 249 with
none. The 249 are the safest customers on the list (median risk 0.5%, maximum 5.6%, none in the
act-this-week band): nothing on their account is wrong enough to say. They get the default
check-in action.

### Actions: the top reason's family picks the call

`config/actions.yaml` maps each family to an action and a two-sentence script, with a default.
The family of the first reason (largest contribution) decides. Counts at 31 Oct: leakage 1,827,
income disruption 742, balance drain 651, activity drop 490, default 249, friction 186, new
customer 178. Leakage leads because its sentences fire on moderate risk customers whose wallet
share rose; the high-risk cards are almost all balance drain and income disruption.

### The planted-churner check

In `_truth.csv`, 138 churners are salaried customers whose salary was planted to stop. At each
one's last snapshot before the churn month, 137 have `months_since_salary >= 1`, so the stop is
visible in the features. Their cards: 120 of 137 (88%) carry a sentence that names the salary.
Before the salaried variants were added the figure was 0 of 137: the model attributes the stop to
`income_change` and `bal_change_vs_6m` (ranked 4th and 1st on those rows) and barely uses
`months_since_salary` (median rank 30), because once income has collapsed the salary-specific
features add nothing. The SHAP ranking is honest about what the model uses; the variant puts the
salary fact, which is a fact of the row, into the sentence. The test repeats this on the small
synthetic set and requires at least 60%.

### Segment view

`data/processed/segments.parquet` (and `.csv`): one row per branch, per segment and per product
held, with customers, mean churn probability, expected churners, 12-month value, value at risk,
share of the total, and value at risk per customer. Pitch rows: SMEs are 15% of customers but
carry 33% of value at risk, 11 USD per customer against 4 to 5 for the rest; Gweru has the
highest churn rate of any branch at 4.2% against 2.6% overall; diaspora customers barely churn
(1.7%).

### What a data team might ask

- *Why not show the three biggest SHAP values whatever they are?* Because half of them have no
  honest sentence (channel mix, "no school fees"), and a card that says "atm_share_3m = 0.12"
  helps nobody. Skipping is safer than inventing.
- *Are the reasons causal?* No. They are the features that moved this customer's score, under a
  model trained on correlations. The card says what changed on the account; the call finds out why.
- *Will the templates survive the bank's data?* The YAML references features by name; if a
  feature is renamed or missing, its block is skipped and a test flags unfilled placeholders.
  Thresholds in the `when` conditions are in the customer's own units (months, shares) and should
  be sanity-checked against the bank's distributions on 4 Dec.

## Phase 6: scoring API, monitoring and Docker (19 Nov 2026)

### The API serves files, not models

`api/main.py` never loads a model or raw data. It reads what the pipeline wrote
(`worklist.parquet`, `cards.json`, `segments.parquet`, `metrics.json`, `survival_metrics.json`,
`drift.parquet`, `runs.jsonl`) once, on the first request, and holds it in memory; `POST /reload`
re-reads after a new pipeline run. Two consequences worth saying out loud. Scoring is a batch
job, so the API cannot score a customer it has not seen at the latest snapshot, and it cannot
leak a raw id because the worklist it reads is already pseudonymised (rule 7); a test checks
that none of the small set's raw ids appears in any response. Every response has a Pydantic
model, so the dashboard and the judges see the same field names as `/docs`.

| endpoint | what it returns |
| --- | --- |
| `GET /health` | ok, snapshot date, customer and card counts |
| `GET /worklist?limit=&segment=&branch=&urgency=` | customers ranked by value at risk, with the filter's total value at risk |
| `GET /customers/{ref}/card` | the Phase 5 card; 404 for an unknown reference |
| `GET /segments?dimension=` | the Phase 5 segment view |
| `GET /metrics` | Phase 2 model comparison, calibration, warnings, fairness table; Phase 3 concordance and timing; drift summary; last run |
| `POST /simulate` | return on contacting the top N by value at risk versus N at random |
| `POST /reload` | pick up a new pipeline run |

### The simulator's arithmetic

Inputs: `top_n`, `cost_per_contact`, `incentive_cost` (paid per customer actually saved) and
`save_rate` (the share of would-be churners a call retains). For each contacted customer the
expected number of churners is the calibrated probability, expected saves are that times the
save rate, and revenue protected is saves times the 12-month value. Cost is contacts times the
contact cost plus saves times the incentive. Random outreach uses the same formulas with the
worklist's average probability and the average value of a churner, which is the expected value of
drawing N customers at random. Both figures are expectations, not promises; the save rate is the
assumption the bank has to supply, and the page in Phase 7 will let them move it.

At 31 Oct 2026 with the defaults (500 contacts, 2 USD per contact, 10 USD per save, 30% save
rate): targeted outreach reaches 85 expected churners, saves 26, protects 4,617 USD for 1,255 USD
of cost, net +3,361 USD; random outreach reaches 14 churners and loses 266 USD. The uplift is
3,628 USD on a 500-call campaign, and the numbers scale with the bank's real values.

### Monitoring: a run log and a drift report

Every pipeline run appends one line to `data/processed/runs.jsonl`: timestamp, snapshot, rows,
mean and 50/90/99th percentile probability, band counts, value at risk, recommendation sources
and the drift counts. On 4 Dec that file is the record of what was run when.

Drift uses the population stability index. For each feature, cut the training block into ten
quantile bins (about 10% of training rows each), count the latest snapshot's rows in the same
bins, and sum `(latest share - train share) x ln(latest share / train share)`. Zero means the
same distribution; the usual reading is below 0.10 stable, 0.10 to 0.25 watch, above 0.25
investigate. On the synthetic data 56 features are stable, 10 watch and 3 investigate.
The three are `bal_change_vs_6m` (0.48), `bal_slope_6m` (0.48) and `salary_growth_6m` (0.39).
The first two are real drift: the generator grows balances through the whole span (average
6-month balance 1,253 USD at the first snapshot, 2,390 at the last), and growth relative to the
baseline was faster in 2025 (median ratio 1.18) than now (1.11). The model's single most important
feature has therefore shifted since training, which is exactly what this report is for; the
fix on real data would be to retrain monthly. The third is a short-history artefact: early rows
cannot look back eight months, so 15% of training rows carry the fill value 1.0 against 0.2% now.

### Docker

One image (`python:3.11-slim` plus `libgomp1` for LightGBM, about 1.8 GB) and two services
sharing `./data` and `./models`. On a clean checkout the api container finds no worklist,
generates the synthetic data, runs the pipeline and only then serves; the dashboard waits for
the health check. The offline recipe is in `docs/runbook-docker.md`: either `docker save` the
built image to a file and `docker load` it at the venue, or cache the Linux wheels in `wheels/`
and build with `--no-index`. The dashboard service is a placeholder page until Phase 7.

The first offline build failed twice for reasons worth recording: `.dockerignore` excluded
`wheels/*.whl` (fixed), and `uvicorn[standard]` pulls in `uvloop`, which `pip download` on
Windows skips because of its Linux-only platform marker. Plain `uvicorn==0.54.0` is enough for
this service, so the extra was dropped.

The `shap` pin was removed from `requirements.txt` (nothing imports it since Phase 5), and
pyarrow was removed from the local venv after it started breaking pandas; Streamlit needs
pyarrow, so on the build machine the dashboard runs in the container.

### What the clean-checkout test found

`docker compose up` with pipeline outputs already in `data/` served `/worklist` and a card within
seconds. The clean-checkout path (empty `data/` and `models/`) took three attempts and found two
things. First, memory: data generation peaks at 1.4 GB and the pipeline at 2.1 GB (measured on the
host with `GetProcessMemoryInfo`, 2,179 MiB sampled in the container), and Docker Desktop's VM on
the build machine had 3.75 GB with 2.2 GB held by unrelated containers, so the kernel killed the
data step with a bare `Killed`. The runbook now states a 2.5 GB requirement. Second, a real bug:
the features step writes `docs/features.md`, and `docs/` is not copied into the image, so the
pipeline crashed with `FileNotFoundError`; the writer now creates the folder. With the VM at 5 GB
the container generated data in 80.6 s, ran the pipeline in 149.9 s and served the same ranking
as the host run (rank 1 P57DB131092), which is also a check that the pipeline is deterministic
across operating systems.

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

## Phase 4: cross-sell engine (18 Nov 2026)

### Rules first, because they survive a different product list

`config/crosssell.yaml` holds the triggers. Each one is a product, a list of conditions on
features from `docs/features.md` (every condition must hold), a priority, and a reason
template whose placeholders are filled with the customer's own numbers. Adding a trigger
is adding a block; a test loads an extra trigger from YAML alone and checks it fires. A
product the customer already holds at the snapshot date is never offered; "holds" comes
from the products table's start and end dates. The same file carries the model-only
reason per product, the fallback chain (first product not held: mobile banking, savings,
fixed deposit, FX savings) and the retention template. Nothing shown to a user is
generated; every sentence is one of those templates (rule 8).

The four starting triggers and what they did:

| trigger | fires at 31 Oct 2026 | test-block uptake, triggered vs not | lift |
| --- | --- | --- | --- |
| remittances in 4 of 6 months, over 150 USD in 3 months, offer fx_savings | 8.5% | 7.4% vs 0.5% | 13.9x |
| school fees in 2+ months of 12, offer education_savings | 21.6% | 5.5% vs 0.6% | 9.1x |
| suppliers paid in 4 of 6 months, offer sme_package | 13.9% | 5.1% vs 0.4% | 12.2x |
| salary up 10%+, balance over 1,500 USD, positive net flow, offer fixed_deposit | 13.6% | 1.6% vs 0.5% | 3.5x |

"Uptake" means the product was started within 90 days after the snapshot, among customers
who did not already hold it. Triggered customers take the matching product at 3.5 to 14
times the rate of untriggered ones, which is the "done when" test.

### Propensity models: one per product, same recipe as churn, honest about failures

The label for a product at a snapshot is "started it within the next 90 days", rows
already holding it excluded. Each product gets a small LightGBM with the Phase 2 recipe:
same time split and gap, class weights, early stopping on validation, isotonic
calibration, the same metrics on the test block. Two gates decide whether the model is
used at all, both judged before the test block is touched: at least 100 uptake rows in
the training block, and a validation ROC-AUC of at least 0.65. The bar is above "beats
chance" because early stopping on the validation block inflates its score.

| product | train uptakes | result |
| --- | --- | --- |
| fx_savings | 386 | model, ROC-AUC 0.807, lift@10% 6.1 |
| education_savings | 506 | model, ROC-AUC 0.768, lift@10% 2.7 |
| sme_package | 293 | model, ROC-AUC 0.766, lift@10% 5.0 |
| savings_account | 117 | model, ROC-AUC 0.886, lift@10% 4.0 |
| mobile_banking | 219 | model, ROC-AUC 0.917, lift@10% 5.2 |
| fixed_deposit | 201 | rule only: validation ROC-AUC 0.63 |
| personal_loan | 334 | rule only: validation ROC-AUC 0.55 |

Two things to say plainly. First, the fixed-deposit and personal-loan models learned
nothing (their first versions scored 0.54 and 0.44 on the test block, worse than a coin),
so the fixed-deposit trigger stands alone and personal loans are never offered. The 0.65
gate was set after seeing those two fail; it is a sanity bar, not a tuned number. Second,
the mobile-banking model's 0.92 is suspiciously high: in the synthetic data customers use
the app channel before the product record starts, so the model is partly reading the
product off the channel mix. On real data expect that one to fall.

### One recommendation per customer

1. A triggered rule wins. If several fire, the higher model probability wins, then
   priority. Source `rule+model` when a model exists for the product, else `rule`.
2. No trigger: the best model probability wins if it is at least twice that product's
   base rate (source `model`).
3. Otherwise the first fallback product not held (source `fallback`).
4. Guardrail last: the top decile of value at risk gets a retention action instead of any
   offer (source `retention`), because selling to someone about to leave is how to lose
   them faster.

At 31 Oct 2026: 432 retention, 1,455 rule+model, 337 rule, 483 model, 1,616 fallback, 0
without an action. `rec_score` is the calibrated uptake probability when a model exists,
the rule's observed training uptake rate otherwise, and the value at risk for retention.

### What a data team might ask

- *Why do the models mostly agree with the rules?* Because the synthetic life events are
  the signal. The models earn their place where a rule is missing (savings account, mobile
  banking) and as a tie-breaker when several rules fire. On real data they may find
  patterns no one wrote a rule for; the gates will say whether they did.
- *Why are the uptake probabilities so small (2 to 7%)?* They are 90-day probabilities,
  calibrated. Quote the lift, not the level.
- *What happens on 4 Dec with the bank's product names?* Edit `crosssell.yaml`: rename
  the products, adjust the conditions to the features that exist, add triggers for
  products we have not seen. Products with no margin in `settings.yaml` are counted at 0
  and listed in a warning.

## Phase 5: reason codes and action cards (18 Nov 2026)

### What a SHAP value is, on the first card

The churn model adds up. For any customer its raw score (in log-odds, before calibration) is a
base value, the same for everyone, plus one contribution per feature. SHAP (TreeSHAP for tree
models) is the method that splits a tree model's prediction into those per-feature contributions
fairly, so that they always sum exactly to the prediction. A positive contribution pushed this
customer's score towards churn, a negative one pulled it away.

The customer ranked first at 31 Oct 2026 (SME, Bulawayo, calibrated risk 67%, 564 USD at risk):

| | log-odds |
| --- | --- |
| base value (everyone starts here) | -1.46 |
| bal_change_vs_6m: balance is 0% of its 6-month average | +1.80 |
| bal_min_3m_ratio: balance touched 0% of the average in the last 3 months | +0.62 |
| income_change: money in this month is 0% of the 6-month average | +0.54 |
| bal_drop_from_3m_peak: 100% of the 3-month peak balance has gone | +0.44 |
| net_flow_3m_usd: more out than in over 3 months | +0.36 |
| txn_count_change: 8 transactions against a usual 26 | +0.34 |
| 63 other features, mostly small negatives (little leakage to wallets or other banks) | -0.35 |
| raw score | 3.41 |

Three of the six positive features describe the same thing, the balance collapsing, so the card
keeps one per family: balance drain, income disruption, activity drop. The raw score of 3.41 would
be a 97% probability on its own; the isotonic calibrator from Phase 2 maps it to 67%, because the
model was trained with churners up-weighted and its raw outputs run hot. Calibration only
stretches the scale, it never reorders customers, so "which features pushed this customer up" is
unaffected by it.

The `shap` package itself cannot load on the build machine (Windows App Control blocks a compiled
scikit-learn file it imports). LightGBM has the same TreeSHAP algorithm built in, reached with
`Booster.predict(pred_contrib=True)`, and that is what `explain.py` uses. A test checks that the
contributions plus the base value reproduce the model's raw score to six decimals.

### Why the global SHAP summary and the Phase 2 gain table disagree

Gain (Phase 2) measures how much each feature improved the training loss when the trees were
built. Mean |SHAP| measures how much each feature actually moves scores on the customers being
scored today. `bal_change_vs_6m` tops both (38% of gain, 7% of mean |SHAP|), but features such
as `remit_amount_3m_usd` and `school_amount_12m_usd` rank high on mean |SHAP| while being minor
by gain. They push the score up for 80 to 90% of customers, by a small, uniform amount: in this
synthetic data customers with no remittances or school fees churn slightly more, so "having no
remittances" is a weak, diffuse risk factor. That is not something to tell a relationship
manager, so those features have no template except for the honest case (remittances used to
arrive and have stopped).

### Reason templates: fixed sentences, conditions, variants

`config/reasons.yaml` has one block per feature worth a sentence (36 of the 69 features). Each
block names a family, a template filled with the customer's own numbers, and conditions that must
hold for the sentence to be rendered, so a positive SHAP value on `months_since_salary` for a
customer whose salary did arrive this month produces nothing rather than a wrong sentence.
A block can hold ordered variants: the first whose conditions hold is used. `income_change` has
a salaried variant ("No salary-sized credit for 3 month(s), about 250 USD used to arrive
monthly; money coming in this month is 19% of the 6-month average") ahead of the generic one.
Features without a block are skipped, never described with generated text (rule 8). The three
aliases a template may use besides the features (`_pct`, `_drop_pct`, `_abs_pct`, plus the three
"share before the last 3 months" values) are computed in `explain.py` and listed at the top of
the YAML; nothing else is derived.

Per card: the three largest positive contributions that have a template, one per family. Result
at 31 Oct 2026, 4,323 cards: 1,040 with three reasons, 1,643 with two, 1,391 with one, 249 with
none. The 249 are the safest customers on the list (median risk 0.5%, maximum 5.6%, none in the
act-this-week band): nothing on their account is wrong enough to say. They get the default
check-in action.

### Actions: the top reason's family picks the call

`config/actions.yaml` maps each family to an action and a two-sentence script, with a default.
The family of the first reason (largest contribution) decides. Counts at 31 Oct: leakage 1,827,
income disruption 742, balance drain 651, activity drop 490, default 249, friction 186, new
customer 178. Leakage leads because its sentences fire on moderate risk customers whose wallet
share rose; the high-risk cards are almost all balance drain and income disruption.

### The planted-churner check

In `_truth.csv`, 138 churners are salaried customers whose salary was planted to stop. At each
one's last snapshot before the churn month, 137 have `months_since_salary >= 1`, so the stop is
visible in the features. Their cards: 120 of 137 (88%) carry a sentence that names the salary.
Before the salaried variants were added the figure was 0 of 137: the model attributes the stop to
`income_change` and `bal_change_vs_6m` (ranked 4th and 1st on those rows) and barely uses
`months_since_salary` (median rank 30), because once income has collapsed the salary-specific
features add nothing. The SHAP ranking is honest about what the model uses; the variant puts the
salary fact, which is a fact of the row, into the sentence. The test repeats this on the small
synthetic set and requires at least 60%.

### Segment view

`data/processed/segments.parquet` (and `.csv`): one row per branch, per segment and per product
held, with customers, mean churn probability, expected churners, 12-month value, value at risk,
share of the total, and value at risk per customer. Pitch rows: SMEs are 15% of customers but
carry 33% of value at risk, 11 USD per customer against 4 to 5 for the rest; Gweru has the
highest churn rate of any branch at 4.2% against 2.6% overall; diaspora customers barely churn
(1.7%).

### What a data team might ask

- *Why not show the three biggest SHAP values whatever they are?* Because half of them have no
  honest sentence (channel mix, "no school fees"), and a card that says "atm_share_3m = 0.12"
  helps nobody. Skipping is safer than inventing.
- *Are the reasons causal?* No. They are the features that moved this customer's score, under a
  model trained on correlations. The card says what changed on the account; the call finds out why.
- *Will the templates survive the bank's data?* The YAML references features by name; if a
  feature is renamed or missing, its block is skipped and a test flags unfilled placeholders.
  Thresholds in the `when` conditions are in the customer's own units (months, shares) and should
  be sanity-checked against the bank's distributions on 4 Dec.

## Phase 6: scoring API, monitoring and Docker (19 Nov 2026)

### The API serves files, not models

`api/main.py` never loads a model or raw data. It reads what the pipeline wrote
(`worklist.parquet`, `cards.json`, `segments.parquet`, `metrics.json`, `survival_metrics.json`,
`drift.parquet`, `runs.jsonl`) once, on the first request, and holds it in memory; `POST /reload`
re-reads after a new pipeline run. Two consequences worth saying out loud. Scoring is a batch
job, so the API cannot score a customer it has not seen at the latest snapshot, and it cannot
leak a raw id because the worklist it reads is already pseudonymised (rule 7); a test checks
that none of the small set's raw ids appears in any response. Every response has a Pydantic
model, so the dashboard and the judges see the same field names as `/docs`.

| endpoint | what it returns |
| --- | --- |
| `GET /health` | ok, snapshot date, customer and card counts |
| `GET /worklist?limit=&segment=&branch=&urgency=` | customers ranked by value at risk, with the filter's total value at risk |
| `GET /customers/{ref}/card` | the Phase 5 card; 404 for an unknown reference |
| `GET /segments?dimension=` | the Phase 5 segment view |
| `GET /metrics` | Phase 2 model comparison, calibration, warnings, fairness table; Phase 3 concordance and timing; drift summary; last run |
| `POST /simulate` | return on contacting the top N by value at risk versus N at random |
| `POST /reload` | pick up a new pipeline run |

### The simulator's arithmetic

Inputs: `top_n`, `cost_per_contact`, `incentive_cost` (paid per customer actually saved) and
`save_rate` (the share of would-be churners a call retains). For each contacted customer the
expected number of churners is the calibrated probability, expected saves are that times the
save rate, and revenue protected is saves times the 12-month value. Cost is contacts times the
contact cost plus saves times the incentive. Random outreach uses the same formulas with the
worklist's average probability and the average value of a churner, which is the expected value of
drawing N customers at random. Both figures are expectations, not promises; the save rate is the
assumption the bank has to supply, and the page in Phase 7 will let them move it.

At 31 Oct 2026 with the defaults (500 contacts, 2 USD per contact, 10 USD per save, 30% save
rate): targeted outreach reaches 85 expected churners, saves 26, protects 4,617 USD for 1,255 USD
of cost, net +3,361 USD; random outreach reaches 14 churners and loses 266 USD. The uplift is
3,628 USD on a 500-call campaign, and the numbers scale with the bank's real values.

### Monitoring: a run log and a drift report

Every pipeline run appends one line to `data/processed/runs.jsonl`: timestamp, snapshot, rows,
mean and 50/90/99th percentile probability, band counts, value at risk, recommendation sources
and the drift counts. On 4 Dec that file is the record of what was run when.

Drift uses the population stability index. For each feature, cut the training block into ten
quantile bins (about 10% of training rows each), count the latest snapshot's rows in the same
bins, and sum `(latest share - train share) x ln(latest share / train share)`. Zero means the
same distribution; the usual reading is below 0.10 stable, 0.10 to 0.25 watch, above 0.25
investigate. On the synthetic data 56 features are stable, 10 watch and 3 investigate.
The three are `bal_change_vs_6m` (0.48), `bal_slope_6m` (0.48) and `salary_growth_6m` (0.39).
The first two are real drift: the generator grows balances through the whole span (average
6-month balance 1,253 USD at the first snapshot, 2,390 at the last), and growth relative to the
baseline was faster in 2025 (median ratio 1.18) than now (1.11). The model's single most important
feature has therefore shifted since training, which is exactly what this report is for; the
fix on real data would be to retrain monthly. The third is a short-history artefact: early rows
cannot look back eight months, so 15% of training rows carry the fill value 1.0 against 0.2% now.

### Docker

One image (`python:3.11-slim` plus `libgomp1` for LightGBM, about 1.8 GB) and two services
sharing `./data` and `./models`. On a clean checkout the api container finds no worklist,
generates the synthetic data, runs the pipeline and only then serves; the dashboard waits for
the health check. The offline recipe is in `docs/runbook-docker.md`: either `docker save` the
built image to a file and `docker load` it at the venue, or cache the Linux wheels in `wheels/`
and build with `--no-index`. The dashboard service is a placeholder page until Phase 7.

The first offline build failed twice for reasons worth recording: `.dockerignore` excluded
`wheels/*.whl` (fixed), and `uvicorn[standard]` pulls in `uvloop`, which `pip download` on
Windows skips because of its Linux-only platform marker. Plain `uvicorn==0.54.0` is enough for
this service, so the extra was dropped.

The `shap` pin was removed from `requirements.txt` (nothing imports it since Phase 5), and
pyarrow was removed from the local venv after it started breaking pandas; Streamlit needs
pyarrow, so on the build machine the dashboard runs in the container.

### What the clean-checkout test found

`docker compose up` on this build machine served `/worklist` and a card within seconds, because
`data/` already held the pipeline outputs. The clean-checkout path (empty `data/` and `models/`)
started correctly, found no raw data and began generating it, and was then killed by the Linux
kernel: Docker Desktop's VM here has 3.75 GB and twenty unrelated containers hold 2.2 GB of it,
while data generation peaks at 1.4 GB and the pipeline at 2.1 GB (measured on the host with
`GetProcessMemoryInfo`). So the path is verified up to the point where memory ran out, not to
the end; the runbook now states the 2.5 GB requirement and the three ways to meet it. On a
laptop running nothing else the default 4 GB VM is enough.

### What a data team might ask

- *Why batch scoring behind the API rather than scoring on request?* Features need the whole
  transaction history at a month end; computing them per request would duplicate the feature
  code and risk leakage. The worklist is rebuilt by the pipeline in three minutes.
- *Is PSI enough for monitoring?* It catches input drift. It does not catch label drift, which
  needs the next 90 days to arrive; the run log keeps the score distribution per run so a
  shift in outputs is visible as well.
- *What if the bank's data has different segments or branches?* The API filters by whatever
  values are in the worklist; nothing is hard-coded.

## Phase 7: dashboard and ROI simulator (20 Nov 2026)

### Four pages, one accent colour, nothing the API does not already say

The dashboard (`dashboard/app.py`, Streamlit) never computes a score, a reason or a sentence
about a customer. It reads the API and lays the answers out in the order of the demo:

1. **Worklist**: three numbers (customers in the two "act" urgency bands, total value at risk,
   the share of it held by the top 10% of the ranking) above the ranked table; branch and segment
   filters; selecting a row opens the card.
2. **Customer card**: risk, urgency, value at risk, the three templated reasons with their family
   chips, the action and call script, the next best product, and a 12-month chart of month-end
   balance (line) against salary credits and all money in (bars). The chart comes from a new
   `GET /customers/{ref}/history` that reads the last twelve rows of `features.parquet` for that
   customer, so the reasons on the left are visible in the data on the right: for the top
   customer the balance line climbs to 11,648 USD and falls to zero in the last month, and the
   salary bars stop four months before.
3. **Insights**: churn drivers bank-wide, value at risk by branch, segment and product, the
   model-versus-rules table and the fairness table.
4. **ROI simulator**: four sliders; the headline figures come from `POST /simulate`, the curve
   from the same arithmetic in `dashboard/logic.py` applied cumulatively down the ranking (a
   test asserts the two agree at every point).

Column headers and labels come from one dictionary in `dashboard/logic.py`, so a raw feature
name cannot reach the screen; the page tests scan every rendered string for the feature-name
patterns and fail if one appears.

### Churn drivers in plain English: the one modelling choice here

The Phase 5 SHAP summary is per feature, and feature names are not for a projector. The new
`GET /drivers` adds up the mean absolute SHAP value of all features in each reason family
(balance drain, income disruption, activity drop, fees and friction, new relationship, life
event, money leaving to wallets or other banks) and reports each family's share of the total.
That is a faithful aggregation: SHAP values are additive, so a family's summed impact is the
part of the score it explains on average. The top five individual signals keep the plain
readings already written for `metrics.json` in Phase 2. On the synthetic data balance drain is
28% of model impact, income disruption 26%, activity drop 14%, life event 13%.

### The ROI curve and its extremes

Net return at N contacts is: revenue protected (sum over the top N of probability x 12-month
value, times the save rate) minus contact cost (N x cost per contact) minus incentive cost
(expected saves x incentive). Random outreach at N is N times the population average of the
same quantities, a straight line. At zero contacts both are zero. When everyone is contacted
both lines meet, because the ranking no longer matters; the simulator's uplift reads zero
there, which is the right answer and a quick sanity check for a judge. With the default
assumptions the Pulse line peaks at 435 contacts (3,368 USD net) and goes negative
beyond about 3,060, where the remaining customers' expected value no longer covers the call.

### Two seconds

Every response is cached in the Streamlit process for five minutes, so a page is an in-memory
lookup after its first render. Measured inside the container with Streamlit's AppTest: first
render 4.4 s for the worklist (it pulls all 4,323 rows once) and 0.9 to 1.3 s for the other
pages; cached renders 0.07 to 0.14 s. The sidebar's "Refresh data" button clears the cache
after a new pipeline run.

### What changed around the dashboard

- `docker-compose.dev.yml` mounts source, config and tests into the containers for editing and
  testing without a rebuild. It became necessary on 20 Nov when Smart App Control started
  blocking a scikit-learn file in the local venv, so the API tests now run in the api container.
- `POST /simulate` accepts `top_n = 0` and anything up to the whole book (it used to require at
  least one contact).
- `scripts/screenshot_pages.py` takes 1280x720 screenshots through the DevTools protocol; the
  ones in `docs/screens/` are from the final run of the day.

### What a data team might ask

- *Why not show the SHAP values themselves?* The card shows the three largest positive
  contributions as sentences and the family chips; the number itself (log-odds) means nothing
  to a branch manager. The API still returns it.
- *Is the save rate made up?* Yes, by design: it is the one input the bank must supply, so it is
  a slider rather than a constant, and the page says so.
- *What does the dashboard do if the pipeline reruns?* The api serves the old outputs until
  `/reload`; the dashboard serves its cache until "Refresh data" or five minutes. Both are
  deliberate: a demo should not change under the presenter.

## Phase 8: rehearsal and the 4 December runbook (21 Nov 2026)

### Two surprise datasets

`scripts/make_rehearsal.py` writes them to `data/rehearsal/a/` and `data/rehearsal/b/` (git-ignored,
regenerate with the script; the seed is fixed so every machine gets the same files). I verified
privately that both go from raw files to a working API through schema.yaml alone, with no code
change; the schema files I used are not in the repository, because finding the mapping is the
exercise. Do not read the generator before the rehearsals. What I can say: (a) needs most of the
schema options that now exist (`date_format` per file, `values` maps, `distinct`,
`one_row_per_key`), and (b) exercises the fallback mode below. Both produce believable numbers,
and (a) raises at least one of the plan's red flags, which is deliberate.

### The fallback path, and the one modelling decision in it

When `schema.yaml` maps a single `customer_summary` table, `pulse.config.data_mode` reports
`summary` and every step adapts: ingest loads the one table (churn flag parsed from 1/0, yes/no
or a `values` map; every column under `features:` cast to a number; empty transaction tables
created), the feature step writes one row per customer with medians filled and a `_missing` flag
where more than 1% of a column was empty, training uses a stratified random split, scoring takes
urgency from probability bands and value from the `value.summary` assumptions, explanations use
`config/reasons_summary.yaml`, the monitor logs the run but writes no drift report, and the API
returns null for the time-to-churn fields.

The decision is the split. With one snapshot there is no past and future to separate, so the
only honest option is a stratified random split with the fixed seed: train 60%, validation 20%,
test 20%, each with the same churn rate. Honest, but optimistic: customers in the test block come
from the same month as the training block, so the score says how well the model ranks customers
*like the ones who already left*, not how well it will rank next quarter's. The LightGBM leaf
floor is also scaled to the table (2% of training rows, at least 20), because the Phase 2
candidates were tuned for tens of thousands of customer-months and will not split at all on a
few thousand rows.

### What Pulse cannot claim in fallback mode

Printed by the pipeline and written in the runbook, because a judge will ask:

- no leakage guarantee: the flag was built outside Pulse, and a column like balance may describe
  the customer after they left (zero because the account was closed)
- no time-based validation, so the headline numbers are optimistic
- no 90-day horizon: the flag means whatever the bank meant by it
- no urgency timing: there is no time-to-churn model
- no behavioural reasons: cards describe the mapped columns against the typical value of
  customers who stayed, not what changed in the account
- value at risk rests on three numbers in settings.yaml
- no next best product: the top decile gets a retention action, everyone else a review marker

The sentence for the pitch: "with transactions Pulse predicts who leaves next; with this extract
it ranks who looks like the customers who already left."

### Hardening that fell out of the rehearsal datasets

- `one_row_per_key: true` on a table collapses several rows per primary key into one, keeping
  the first non-null value of each column. A customer file with one row per account, where the
  branch is blank on some rows, cannot be deduplicated with `distinct` alone; the error message
  now names both options.
- `date_formats:` per column, `PULSE_SCHEMA` to point at another schema file and `PULSE_ROOT` to
  run against another project root (the Docker image already relied on it).
- `scripts/profile_data.py` ends with a schema skeleton and guesses which file is which table
  from column types and row counts; on the synthetic data and both rehearsal sets the guesses
  were right.

### What a data team might ask

- *Why not auto-detect every numeric column as an input in fallback mode?* Rule 3: source column
  names must not leak into the code or the screens. Mapping each input in schema.yaml takes a
  minute with the skeleton and gives every feature a canonical name the templates can use.
- *Is a stratified split ever acceptable in production?* No. It is the rehearsal answer for an
  extract with no time axis. The first thing to ask the bank for is a second extract from an
  earlier month, which turns the problem back into a time split.
- *What if the flag is "closed account" rather than churn?* Then the model learns closure, and
  the silent leavers are invisible. Say so, and show the transaction-mode definition on a slide.
