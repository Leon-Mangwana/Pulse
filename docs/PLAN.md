# Pulse build plan for Claude Code

## How to use this plan

Run one phase per Claude Code session, in order, and do not start the next phase until the "done when" checks pass. Each phase below has a goal, a prompt to paste, and checks you run yourself.

The working rhythm for every phase:

1. Start a fresh Claude Code session in the repo and paste the phase prompt.
2. Let it propose a plan first. Read it, ask about anything you do not follow, then tell it to build.
3. Run the "done when" checks yourself. Do not accept "it works" without seeing the output.
4. Ask: "Explain what you built and why, as if I had to defend it to a bank's data team." Save the answer in `docs/notes.md`.
5. Commit, then close the session.

Step 4 matters most. Judges will question you, not Claude Code, so every phase should end with you able to explain it.

| Phase | Dates | What exists at the end |
| --- | --- | --- |
| 0. Setup and stand-in data | 17–18 Nov | Repo, synthetic bank data, schema mapping file |
| 1. Signal Factory and churn target | 19–21 Nov | One feature table with a churn label, no leakage |
| 2. Churn model | 22–23 Nov | LightGBM model beating two baselines on later months |
| 3. Time-to-churn and value at risk | 24–25 Nov | Worklist ranked in dollars |
| 4. Cross-sell engine | 26 Nov | Next best product per customer with a reason |
| 5. Reason codes and action cards | 27–28 Nov | Plain-English card for every customer |
| 6. Scoring API and Docker | 29 Nov | `docker compose up` serves scores |
| 7. Dashboard and ROI simulator | 30 Nov–1 Dec | Clickable demo |
| 8. Rehearsal and runbook | 2–3 Dec | Two timed dry runs on unseen data |

Two facts shape the whole plan:

- **You have not seen BancABC's data.** It may have different column names, or no raw transactions at all. Every phase reads data through one mapping file, so swapping datasets on 4 Dec means editing that file, not the code.
- **You may not have internet or Claude Code at the venue.** Ask the organisers this week. Phase 8 assumes the worst and prepares you to run everything offline.

## Phase 0: setup and stand-in data (17–18 Nov)

**Goal:** a working repo and a realistic fake bank dataset you fully control. Build your own synthetic data instead of hunting for a public set: you can plant the exact Zimbabwean signals the proposal promises, and you know the true answers, so you can check the pipeline finds them.

```text
Read CLAUDE.md. Set up the repo skeleton it describes, with a Makefile, pinned
requirements.txt, pytest, and a .gitignore that excludes data/.

Then build a synthetic data generator, src/pulse/synth.py, run with `make data`.

Output four CSVs in data/raw/ covering 24 months for about 5,000 customers:
- customers: customer_id, open_date, segment (salaried, SME, informal, diaspora-supported),
  branch, gender, age_band
- accounts: account_id, customer_id, product_type, currency (USD or ZiG), open_date, close_date
- transactions: txn_id, account_id, customer_id, txn_date, amount, direction (credit/debit),
  currency, channel (branch, ATM, POS, mobile app, USSD, ZIPIT, RTGS),
  counterparty_type (employer, mobile_wallet, other_bank, school, supplier, remittance,
  merchant, fee), description, balance_after
- products: customer_id, product, start_date, end_date

Plant these behaviours, with noise so none of them is a perfect rule:
- About 10% of customers churn. Most churn silently (activity and balance fade) and only a
  few close the account.
- Before churning, many customers show: salary credit stops or shrinks, balance drains over
  2 to 4 months, a growing share of outflows goes to mobile wallets and other banks,
  fewer active days.
- Some customers show the same warning signs and do NOT churn. Some churn with no warning.
- Life events: regular diaspora remittance inflows, school-fee payments in January, May and
  September, steady supplier payments. Customers with these are more likely to later take
  the matching product (foreign-currency savings, education savings, SME package).

Save the hidden truth (who churns and when, who has which life event) to
data/raw/_truth.csv. The pipeline must never read this file. It is only for me to check results.

Then create config/schema.yaml mapping canonical column names to the CSV column names, and
src/pulse/ingest.py that loads everything through that mapping into DuckDB and validates
types, date ranges, duplicates and nulls, printing a short data-quality report.

Propose your plan first and explain how you will make the data realistic but not trivially easy.
```

**Done when:**

- `make data` then `make ingest` run clean and print a data-quality report.
- You open the CSVs and the transactions look like a real statement.
- You rename a column in a CSV, change only `schema.yaml`, and everything still runs.

## Phase 1: Signal Factory and churn target (19–21 Nov)

**Goal:** one table with a row per customer per month, holding about 60 features and a churn label. This is the heart of the project, so it gets the most time. A good feature table with a simple model beats a clever model on weak features.

The idea to hold on to: pick a date, the "snapshot date". Features describe what the customer did **before** it. The label says what happened in the 90 days **after** it. Mixing the two is leakage, the most common way first ML projects produce fake results.

```text
Read CLAUDE.md. Build the Signal Factory and the churn target.

1. Target (src/pulse/target.py). For each customer and each month-end snapshot date, label
   churn_90d = 1 if in the following 90 days the customer either closes their account
   (hard churn) or their transaction count and average balance both fall below 20% of their
   own prior six-month baseline (silent churn). Put the 20%, the 90 days and the six months
   in config/settings.yaml. Exclude snapshots where the customer has under six months of
   history, is already inactive, or where fewer than 90 days of data follow the snapshot.
   Also record days_to_churn for later use.

2. Features (src/pulse/features/), one module per family, each computed only from data
   strictly before the snapshot date:
   - income: salary credit detected, months since last salary, salary amount change,
     payer changed
   - balance: average balance over 1, 3 and 6 months, slope of balance, USD vs ZiG share
     and its change
   - leakage: share of outflows to mobile wallets and to other banks, and the 3-month
     trend of each
   - engagement: active days per month, transaction count, channel mix, digital share,
     and the change versus the customer's own baseline
   - friction: fees paid, failed or reversed transactions
   - life events: remittance inflow regularity, school-fee payments, regular supplier
     payments, rising salary
   Detect things like "salary" from transaction patterns (regular monthly credit of similar
   size), not only from labels, because the real data may not label them.

3. Output data/processed/features.parquet with a data dictionary in docs/features.md that
   gives each feature one plain-English line.

4. Tests: a leakage test that changes transactions after a snapshot date and asserts the
   features for that snapshot are unchanged; a test that every feature handles a customer
   with no transactions in the window.

Before coding, explain to me how the snapshot table works with a tiny worked example of
one customer over four months.
```

**Done when:**

- `make features` produces the table and the leakage test passes.
- The churn rate per month is stable and somewhere between 3% and 15%.
- You compare ten labelled churners against `_truth.csv` by hand and they match.
- You can explain three features and why each should predict churn.

## Phase 2: churn model and honest evaluation (22–23 Nov)

**Goal:** a LightGBM model that predicts 90-day churn, shown to beat two simpler baselines on months it never saw. The number to remember for the pitch is capture at top 10%: of all customers who churned, what share were in the 10% the model ranked riskiest.

```text
Read CLAUDE.md. Build the churn model in src/pulse/models/churn.py, run with `make train`.

1. Split by time: earliest snapshots for training, the next block for validation (tuning),
   the latest labelled block for the final test. Print the date ranges and churn rate of each.

2. Train three models on the same split and compare them in one table:
   - a rules baseline (for example: no salary for two months, or balance down more than half)
   - logistic regression
   - LightGBM with class weights, early stopping on validation, and a small sensible
     parameter set. No large hyperparameter search.

3. Report for each, on the test block only: PR-AUC, ROC-AUC, capture rate and lift at the
   top 5%, 10% and 20%, and a calibration table (predicted vs actual churn rate by decile).
   Calibrate the LightGBM probabilities if they are off, because we multiply them by dollars
   in the next phase.

4. Sanity checks, reported explicitly:
   - top 15 features by importance, with a one-line reading of each
   - any single feature that alone gives ROC-AUC above 0.9 (a leakage warning)
   - train vs test gap (an overfitting warning)

5. Fairness check: capture rate and flag rate by gender and age band, using those columns
   only for this report and never as inputs.

6. Save the model, the feature list and the metrics to models/, and append a dated results
   table to docs/metrics.md.

Explain PR-AUC, lift and calibration to me in plain terms before you show the numbers,
and tell me which single result I should quote to a banker.
```

**Done when:**

- LightGBM beats both baselines at top 10% capture on the test block.
- No metric looks impossibly good. If ROC-AUC is above 0.95, make Claude Code hunt for leakage before moving on.
- The top features make business sense: salary, balance trend and wallet leakage should be near the top.
- You can say one sentence like: "Calling the riskiest 10% reaches X% of the customers who actually left, Y times better than calling at random."

## Phase 3: time-to-churn and value at risk (24–25 Nov)

**Goal:** turn the churn score into a worklist sorted by dollars, with an urgency band per customer. This is what separates Pulse from teams that stop at a classifier, so build the value part first and the survival part second.

```text
Read CLAUDE.md. Add customer value, value at risk and time-to-churn.

1. Value (src/pulse/value.py). Estimate each customer's expected 12-month value to the bank
   as fee income plus interest margin on average balances plus a margin per product held.
   Every rate and margin is an assumption in config/settings.yaml with a comment, so my
   finance teammate can change them without touching code. Convert ZiG to USD with a rate
   from settings.

2. Value at risk = calibrated churn probability x expected 12-month value. Produce
   data/processed/worklist.parquet: one row per customer for the latest snapshot with
   probability, value, value at risk and rank.

3. Time-to-churn (src/pulse/models/survival.py). Fit a Cox proportional hazards model with
   lifelines on a small set of the strongest features, using days_to_churn and treating
   customers who have not churned as censored. Output the probability of churning within
   30, 60 and 90 days and an urgency band: act this week, this month, or watch.
   Report the concordance index on the test block.

4. Compare three ways of ordering the call list on the test block: random, by churn
   probability, by value at risk. For each, show dollars of at-risk value reached when
   contacting the top 5%, 10% and 20%.

Explain censoring and what a Cox model does using an everyday example before you code it.
If the survival model adds little over the churn score, say so plainly.
```

**Done when:**

- The worklist exists and the top 20 customers look like people a banker would want to call: valuable and clearly slipping.
- Ranking by value at risk reaches more dollars than ranking by probability alone. That comparison is a slide.
- Your finance teammate has reviewed the assumptions in `settings.yaml` and can defend each one.

## Phase 4: cross-sell engine (26 Nov)

**Goal:** a next best product for each customer, with a reason a relationship manager would believe. Start with transparent trigger rules, which work even if the real data has no product history, then add a model only if the data supports it.

```text
Read CLAUDE.md. Build the cross-sell engine in src/pulse/crosssell.py.

1. Trigger rules in config/crosssell.yaml, each with a condition on existing features,
   the product to offer, and a plain-English reason template. Start with:
   - regular diaspora remittance inflows -> foreign-currency savings or remittance bundle
   - termly school-fee payments -> education savings plan
   - steady supplier payments -> SME banking package
   - rising salary with idle balance -> fixed deposit or investment product
   Skip any product the customer already holds.

2. Propensity models: where the products table has enough history, train one small
   LightGBM model per product predicting uptake in the next 90 days, with the same
   time-based split and metrics as the churn model. If a product has too few uptakes to
   learn from, say so and fall back to the rule.

3. Combine into one recommendation per customer: product, score, reason, and which
   source produced it (rule or model).

4. Guardrail: if a customer's value at risk is in the top decile, the recommendation is a
   retention action, not a sales offer. Add this to the worklist output.

5. Report how many customers each trigger fires for, and uptake rate among triggered vs
   untriggered customers on the test block.
```

**Done when:**

- Every customer on the worklist has either a retention action or a product with a reason.
- Triggered customers take up the matching product at a visibly higher rate than untriggered ones.
- You can add a new trigger by editing the YAML file only. On 4 Dec the bank's real product list will differ from yours.

## Phase 5: reason codes and action cards (27–28 Nov)

**Goal:** every customer on the worklist gets a card with three plain-English reasons, a recommended action and a short call script. SHAP supplies which features pushed this customer's score up; fixed templates turn them into sentences, so nothing on the card is invented.

```text
Read CLAUDE.md. Build explainability and action cards in src/pulse/explain.py.

1. Compute SHAP values for the churn model with TreeExplainer. Save a global summary
   (top drivers across all customers) and per-customer values for the latest snapshot.

2. Reason templates in config/reasons.yaml: for each feature (or feature family), a
   sentence template filled with the customer's actual values, for example
   "No salary credit for {months_since_salary} months (previously monthly)" or
   "Transfers to mobile wallets rose from {before}% to {after}% of spending".
   For each customer, take the three features with the largest positive SHAP value, skip
   duplicates from the same family, and render their templates. Features without a
   template are skipped, never described with generated text.

3. Action rules in config/actions.yaml mapping the top reason family to a recommended
   action and a two-sentence call script (income disruption -> ask about employer change
   and offer salary-switch incentive, and so on).

4. Output data/processed/cards.json: per customer, pseudonymised ID, risk, urgency band,
   value at risk, three reasons, recommended action, call script, next best product.

5. Segment view: churn risk and total value at risk by branch, segment and product, saved
   as a table for the dashboard.

6. Tests: every card has exactly three reasons or fewer with no empty placeholders, and
   no card contains a raw customer ID.

Explain what a SHAP value means for one customer using a concrete card as the example.
```

**Done when:**

- You read 20 cards and each one tells a believable story that matches the customer's transactions.
- For a planted churner in `_truth.csv` whose salary stopped, the card says so.
- The segment table shows where risk concentrates. Pick the most striking row for the pitch.

## Phase 6: scoring API and Docker (29 Nov)

**Goal:** one command brings up the whole system. This is your home ground, so keep it to a day and spend the saved time on Phases 1 and 2.

```text
Read CLAUDE.md. Wrap the pipeline in a FastAPI service and Docker Compose.

1. `make pipeline` runs ingest -> features -> train -> value -> crosssell -> explain end to
   end and prints the time each step took.

2. FastAPI app in api/ that reads the pipeline outputs:
   - GET /health
   - GET /worklist?limit=&segment=&branch=   ranked by value at risk
   - GET /customers/{id}/card
   - GET /segments
   - GET /metrics   model results and the fairness table
   - POST /simulate  body: top_n, cost_per_contact, incentive_cost, save_rate ->
     revenue protected, cost, net return, and the same for random outreach
   Pydantic response models, pseudonymised IDs only.

3. Light monitoring: log every scoring run with a timestamp, row counts and score
   distribution, and add a drift report comparing the feature distributions of the
   latest snapshot against the training data (population stability index per feature).

4. Dockerfile and docker-compose.yml with two services, api and dashboard, sharing a data
   volume. Images must build with no internet access once dependencies are cached, so
   document how to pre-build them.

5. API tests with pytest for each endpoint.
```

**Done when:**

- `docker compose up` on a clean checkout serves `/worklist` and `/customers/{id}/card`.
- `make pipeline` finishes in under 10 minutes. On 4 Dec you will run it several times.
- You turn off your Wi-Fi and it all still builds and runs.

## Phase 7: dashboard and ROI simulator (30 Nov–1 Dec)

**Goal:** the four screens the demo story needs, and nothing else. Use Streamlit, not Next.js. The proposal left this open, but with two days a Streamlit app that works beats a polished front end that is half finished, and it handles tables, charts and sliders natively.

```text
Read CLAUDE.md. Build the Streamlit dashboard in dashboard/, calling the FastAPI service.

Four pages, in the order of the demo:
1. Worklist: headline numbers at the top (customers at risk, total value at risk, share
   reached by the top 10%), then the ranked table with risk, urgency band, value at risk
   and next action. Filters for branch and segment. Clicking a row opens the card.
2. Customer card: risk, urgency, value at risk, three reasons, recommended action, call
   script, next best product, and a small chart of the customer's balance and salary
   credits over the last 12 months so the reasons are visible in the data.
3. Insights: top churn drivers bank-wide in plain English, value at risk by branch,
   segment and product, and the model-vs-baseline comparison with the fairness table.
4. ROI simulator: sliders for customers contacted, cost per contact, incentive value and
   assumed save rate. Shows revenue protected, cost and net return for Pulse ranking
   versus random outreach, with a chart that updates live.

Design: clean, a single accent colour, large readable numbers, all money in USD,
no raw feature names anywhere on screen. It will be shown on a projector, so test at
1280x720. Every page must load in under two seconds from cached API responses.
```

**Done when:**

- You can run the five-minute demo story from the proposal without touching a terminal.
- A teammate who has not seen it can find the top customer and explain why they are at risk.
- The ROI simulator gives sensible answers at the extremes: zero customers contacted, and everyone contacted.

## Phase 8: rehearsal and the 4 Dec runbook (2–3 Dec)

**Goal:** prove you can go from an unfamiliar dataset to a working demo inside the hackathon's time limit, twice, before the day. Most teams never practise the data swap, and it is where the day is won or lost.

```text
Read CLAUDE.md. Help me rehearse the hackathon day.

1. Create two surprise datasets in data/rehearsal/ that I have not seen:
   a) the same kind of bank data but with different column names, date formats, a merged
      customer-and-account table, missing values and a different product list
   b) a worst case: one customer-level table with summary columns (tenure, balance,
      number of products, activity flag, a churn flag) and no raw transactions
   Do not tell me what is in them.

2. Add a fallback path: if there are no raw transactions, the pipeline builds features
   from whatever customer-level columns exist, uses a provided churn flag as the target
   with a stratified split, and the rest (value at risk, reasons, cards, dashboard, ROI)
   still works. Tell me clearly what Pulse cannot claim in this mode.

3. Write docs/runbook.md: a timed checklist for the day, from opening the data to the
   final demo, with the exact commands, the decisions I must make (churn definition,
   value assumptions, product mapping) and what to check after each step.

4. Write scripts/profile_data.py that takes any CSV folder and prints columns, types,
   date ranges, null rates and likely join keys, so I can fill in schema.yaml quickly.
```

Then run the rehearsal yourself, with a timer and without asking Claude Code to fix things for you unless you are truly stuck:

- [ ] Rehearsal 1 on dataset (a): data to working dashboard in under 3 hours
- [ ] Rehearsal 2 on dataset (b): fallback mode to working dashboard in under 2 hours
- [ ] Full five-minute pitch delivered twice with the team, timed
- [ ] Docker images pre-built and all Python packages cached for offline use
- [ ] Repo, datasets and a backup recording of the demo on a USB stick
- [ ] One page of likely judge questions with your answers, from `docs/notes.md`

On the day, follow the three blocks from your proposal:

1. **First third:** profile the data, fill in `schema.yaml`, agree the churn definition with the team, run features. Do not train anything until the leakage test passes on the real data.
2. **Middle third:** train, check the numbers are believable, set value assumptions with your finance lead, build the worklist.
3. **Final third:** freeze the code. Pull three real stories from the data for the pitch, then rehearse. No new features in the last hour.

## If you fall behind

Cut from the top of this list first. Phases 1, 2, the value-at-risk half of Phase 3, Phase 5 and the worklist and card pages are the core, and a demo with only those can still win.

| Cut order | What to drop | What you say instead |
| --- | --- | --- |
| 1 | Drift report and monitoring (Phase 6) | "Monitoring is the first thing we would add in production" |
| 2 | Per-product propensity models (Phase 4) | Trigger rules alone, which are easier to explain anyway |
| 3 | Survival model (Phase 3) | Urgency band from the churn probability and how fast the balance is falling |
| 4 | FastAPI layer (Phase 6) | Streamlit reads the pipeline's output files directly |
| 5 | Insights page (Phase 7) | One slide with the top three drivers |

Never cut the time-based split, the leakage test, the baselines or the rehearsals. Those are what make your numbers believable.

The rehearsal targets of 3 hours and 2 hours are placeholders. Replace them with half the real working time once the organisers confirm the schedule.

## ML ideas to own before the judges ask

You need to explain these seven ideas in your own words. Ask Claude Code to quiz you on them at the end of Phase 2 and again in Phase 8.

| Idea | In one line | Why a judge cares |
| --- | --- | --- |
| Leakage | The model saw information it would not have at prediction time | It makes results look great and fail in real use |
| Time-based split | Train on the past, test on the future | It mirrors how the bank would use the model |
| Class imbalance | Few customers churn, so "nobody churns" scores about 90% accuracy | It is why you never quote accuracy |
| PR-AUC | How well the model ranks the rare churners near the top | A fair score when churners are rare |
| Lift and capture at top 10% | Share of all churners found in the riskiest tenth, versus random | It matches how many calls the bank can make |
| Calibration | When the model says 30%, about 30% really churn | Value at risk multiplies probability by dollars |
| SHAP | How much each feature pushed one customer's score up or down | It turns a score into reasons a banker can act on |

Red flags to stop and investigate:

- A score that looks too good, such as ROC-AUC above 0.95.
- One feature far more important than all the others.
- Test results much worse than training results.
- A top feature you cannot explain in business terms.
- Results that change when you rerun the same command.
