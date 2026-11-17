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
