# Hackathon day runbook (4 December 2026)

A timed checklist from opening the bank's files to the final demo. Times assume an 8-hour
window; scale them to half of whatever the organisers give you. Everything runs from the
project folder in PowerShell:

```powershell
cd "C:\Users\k2leo\OneDrive\Documents\Leon Codes\Pulse"
$env:PYTHONIOENCODING = "utf-8"
```

On this laptop scikit-learn is blocked outside Docker, so the pipeline runs in the api container.
Define one alias at the start and use it everywhere `pulse` appears below:

```powershell
function pulse { docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps api python -m pulse.cli @args }
```

On a machine where the venv works, `pulse` is `.\.venv\Scripts\python.exe -m pulse.cli` instead.

## Before the day (tick these on 3 December)

- [ ] `docker compose build --build-arg PIP_FLAGS="--no-index"` succeeds offline; `docker image ls pulse` shows the image
- [ ] `wheels\` holds the 72 Linux wheels; `pulse-image.tar` exported with `docker save pulse:latest -o pulse-image.tar`
- [ ] USB stick: repository, `pulse-image.tar`, `wheels\`, both rehearsal datasets, the demo recording
- [ ] `git status` clean on `main`; the synthetic demo (current `data\` and `models\`) committed to memory as the fallback demo
- [ ] Docker Desktop memory at least 5 GB (Settings > Resources); other containers stopped
- [ ] Judge questions page printed (from the "What a data team might ask" sections in docs/notes.md)
- [ ] Rehearsals 1 and 2 done with times written at the bottom of this file

## Block 1 (first third): understand the data, get features out

**0:00 Open the files.** Copy the bank's files into `data\raw\` (move the synthetic ones to
`data\raw_synthetic\`). Nothing else is touched until the profile is read.

```powershell
.\.venv\Scripts\python.exe scripts\profile_data.py data\raw --out data\raw\_profile.md
```

Read the profile top to bottom. Decide now:

| decision | write it down |
| --- | --- |
| Which file is which table? Is there a transactions file at all? | if not, go to **Fallback mode** below |
| Join keys (the "likely join keys" table) | customer id column in every file |
| Date formats (one per file; per column if they differ) | `date_format` / `date_formats` |
| Category codes for direction, channel, counterparty, currency, segment | `values:` maps |
| One row per account in the customer file? | `distinct: true` on the customers mapping |
| Products in the data and what each is worth per year | `value.product_margin_usd` in settings.yaml, and a `values:` map on `products.product` if the names differ from Pulse's |

**0:20 Fill in schema.yaml.** Copy the skeleton at the end of the profile into `config\schema.yaml`
(keep the old one as `config\schema.synthetic.yaml`). Set `data.start_date` and `data.end_date`
in `config\settings.yaml` to the period the transactions cover, and `fx.zig_per_usd` to the rate
the bank quotes.

**0:30 Ingest.**

```powershell
pulse ingest
```

Check: `RESULT OK` at the end of the quality report. Every `ERROR` line names the fix
(a missing mapping, a value outside `credit/debit`, duplicate keys, orphan rows). `WARNING`
lines are allowed but read each one. Typical fixes: add a `values:` entry, add `distinct: true`,
correct a date format. Repeat until OK. Do not edit the data files.

**0:50 Agree the churn definition with the team.** Pulse's label is: no activity and balance below
20% of the customer's own 6-month baseline for 90 days, or a formal closure. The knobs are under
`target:` in settings.yaml (`horizon_days`, `silent_threshold`). If the bank has its own definition
(for example 60 days dormant), set the knobs to match and write the definition on the slide.

**1:00 Features.**

```powershell
pulse features
```

Check: the churn rate by month at the end of the output. Sensible is 1% to 6% a month. If it is
0% or above 15%, the label or the dates are wrong; go back to the churn definition before
training anything. Confirm `docs\features.md` was rewritten (timestamp).

**1:15 Leakage test on the real data.** This is the one test that must pass before any training:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps api python -m pytest -q tests/test_features.py tests/test_schema.py tests/test_ingest.py
```

Stop sign: a failing leakage test means a feature can see the future; do not train until it passes.

## Block 2 (middle third): train, believe the numbers, build the worklist

**1:30 Train.**

```powershell
pulse train
```

Check, in this order, in the printed report (also in `models\train_report.md`):

1. The split table: train, valid, test blocks in date order with a gap before test.
2. Three models on the test block. LightGBM should beat the rules baseline on PR-AUC and
   capture at 10%. If the rules baseline wins, say so in the pitch and use it; it is still a ranking.
3. Red flags to stop and investigate: ROC-AUC above 0.95; one feature far ahead of all others in
   gain; `LEAKAGE WARNING` or `OVERFIT WARNING`; a top feature you cannot explain in business terms.
4. Calibration: the calibrated mean prediction within a point of the actual rate.
5. The fairness table: flag rates similar across gender and age groups.

**2:00 Value assumptions with the finance lead.** Under `value:` in settings.yaml: fee
annualisation, interest margins (USD and ZiG) and the yearly margin per product. These numbers
multiply straight into value at risk and the ROI page, so write down who agreed them.

**2:15 Score, monitor, serve.**

```powershell
pulse score
pulse monitor
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
Invoke-RestMethod -Method Post http://localhost:8000/reload
```

Check: `/health` shows the right customer count and snapshot date; the top 20 printed by `score`
look believable (a mix of segments, large but not absurd values); the drift report is mostly
`stable` (the training block and the latest month come from the same data, so large drift here
means a date problem). Open http://localhost:8501 and press "Refresh data".

## Block 3 (final third): freeze, find the stories, rehearse

**3:00 Freeze the code.** `git add -A; git commit -m "hackathon data: schema, settings, outputs"`.
No new features after this point.

**3:10 Three real stories.** From the worklist page, pick the top customer, one from the segment
the judges care about, and one whose card reasons are visible in the 12-month chart. Write the
three references on a card. Check each story with `scripts\inspect_worklist.py --cards 3`.

**3:30 Rehearse the five-minute story twice**, timed, from the dashboard only: Worklist headline
numbers, one card, Insights drivers, ROI with the save rate moved once. Fallback if the dashboard
dies: `docs\screens\*.png`.

**4:00 Spare.** Judge questions page, backup recording, charge the laptop.

## Fallback mode: one customer-level table, no transactions

Pulse switches to this mode by itself when `config\schema.yaml` maps a single `customer_summary`
table (see the commented block in schema.yaml) instead of the four raw tables.

```powershell
.\.venv\Scripts\python.exe scripts\profile_data.py data\raw      # the skeleton ends with a customer_summary block
# edit config\schema.yaml: customer_id, churn_flag, the optional meta columns, and every numeric or
# yes/no column you want as an input under features:
pulse pipeline
```

Decisions: which column is the churn flag and what its values mean (`values: churn_flag:`);
which columns are inputs (never gender or age, which go under `gender:` and `age_band:` for the
fairness table only); the value assumptions under `value.summary` in settings.yaml (a balance
column and a product-count column if they exist, else one default figure per customer).

Check: the split table shows the same churn rate in train, valid and test (stratified); the test
ROC-AUC is below 0.95; the top features are columns a banker would expect. Cards describe the
mapped columns against the typical value of customers who stayed.

What Pulse cannot claim in this mode, and should say out loud in the pitch:

- no leakage guarantee: the flag was built outside Pulse, and columns such as balance may
  describe the customer after they left
- no time-based validation: with one snapshot the split is stratified random, which is optimistic
- no 90-day horizon: the flag means whatever the bank meant by it
- no urgency timing: bands come from the probability alone, there is no time-to-churn model
- no behavioural reasons: cards describe the mapped columns, not what changed in the account
- value at risk rests on the `value.summary` assumptions
- no next best product: every recommendation is a retention call

The one-line version for the judges: "With transaction data Pulse predicts who leaves next; with
this extract it ranks who looks like the customers who already left."

## If you fall behind

Cut in this order and say the sentence in the right-hand column (from docs/PLAN.md):

| cut | say |
| --- | --- |
| drift report and monitoring | "Monitoring is the first thing we would add in production" |
| per-product propensity models | trigger rules alone, which are easier to explain |
| survival model | urgency from the churn probability and how fast the balance is falling |
| FastAPI layer | Streamlit reads the pipeline's output files directly |
| Insights page | one slide with the top three drivers |

Never cut: the time-based split, the leakage test, the baselines, the rehearsals.

## Commands in one place

```powershell
.\.venv\Scripts\python.exe scripts\profile_data.py data\raw --out data\raw\_profile.md
pulse ingest
pulse features
pulse train
pulse score
pulse monitor
pulse pipeline                               # all five in order, with timings
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
Invoke-RestMethod -Method Post http://localhost:8000/reload
Invoke-RestMethod http://localhost:8000/health
.\.venv\Scripts\python.exe scripts\inspect_worklist.py --cards 3
docker compose down
```

Another schema file without editing the main one: `$env:PULSE_SCHEMA = "config/schema.bank.yaml"`
(inside the containers set it with `-e PULSE_SCHEMA=config/schema.bank.yaml` on the `run` line).

## Rehearsal log

| rehearsal | dataset | start | working dashboard | notes |
| --- | --- | --- | --- | --- |
| 1 | data/rehearsal/a | | | |
| 2 | data/rehearsal/b | | | |
