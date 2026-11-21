# Pulse

AI customer-intelligence copilot for the BancABC Universities Hackathon. From bank
transaction data it predicts 90-day churn, ranks customers by value at risk, recommends a
next best product and explains each prediction in plain English.

## Setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## Commands

| Step | Command | What it does |
| --- | --- | --- |
| data | `python -m pulse.cli data` | generate synthetic bank data into `data/raw/` |
| ingest | `python -m pulse.cli ingest` | load `data/raw/` through `config/schema.yaml` into DuckDB and print a data-quality report |
| features | `python -m pulse.cli features` | build the snapshot table with 69 features and the churn label into `data/processed/features.parquet` |
| train | `python -m pulse.cli train` | churn model (time split, rules / logistic / LightGBM, calibration, sanity and fairness checks), then the Cox time-to-churn model and the value-at-risk report; saves `models/` |
| train | (same command) | also fits the Cox time-to-churn model and one propensity model per product, and writes `models/value_report.md` and `models/crosssell_report.md` |
| score | `python -m pulse.cli score` | worklist at the latest snapshot: probability, 12-month value, value at risk, rank, urgency band, next best product or retention action with its reason (rules in `config/crosssell.yaml`); writes `data/processed/worklist.parquet` (pseudonymised) and `id_map.parquet` |
| score | (same command) | also writes the action cards `data/processed/cards.json` (three templated reasons, action, call script, next best product per customer), `shap_latest.parquet`, the segment view `segments.parquet`/`.csv` and `models/shap_summary.md`; `python -m pulse.cli explain` rebuilds these from a saved worklist |
| monitor | `python -m pulse.cli monitor` | appends a run-log line to `data/processed/runs.jsonl` and writes the feature drift report (PSI per feature, latest snapshot vs the training block) to `models/drift_report.md` and `data/processed/drift.parquet` |
| pipeline | `python -m pulse.cli pipeline` | ingest, features, train, score, monitor in order, with the time each step took |
| api | `.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload` | FastAPI on http://localhost:8000 (docs at `/docs`): `/health`, `/worklist`, `/customers/{ref}/card`, `/segments`, `/metrics`, `POST /simulate`, `POST /reload`; Phase 7 adds `/customers/{ref}/history` and `/drivers` |
| docker | `docker compose up` | api on :8000 and dashboard on :8501 sharing `./data` and `./models`; see `docs/runbook-docker.md` for offline builds |
| dashboard | (part of `docker compose up`) | Streamlit on http://localhost:8501: Worklist, Customer card, Insights, ROI simulator, all read from the api; `dashboard/app.py` |
| dev | `docker compose -f docker-compose.yml -f docker-compose.dev.yml up` | same, with `src/`, `api/`, `dashboard/`, `config/`, `tests/` mounted so edits apply without a rebuild |
| tests | `docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dashboard python -m pytest -q tests/test_dashboard_app.py` | renders the four pages against the live api; the rest of the suite runs with `python -m pytest` (in the api container on machines where scikit-learn is blocked) |
| screens | `python scripts/screenshot_pages.py docs/screens` | 1280x720 screenshots of the four pages with headless Edge |
| pipeline | `python -m pulse.cli pipeline` | run every built step in order with timings |
| tests | `python -m pytest` | run the test suite |

Use `.\.venv\Scripts\python.exe` in place of `python` if the venv is not activated.
The `Makefile` wraps the same commands for machines that have `make`.

## Using a different dataset

Point each `path` in `config/schema.yaml` at the new files and edit the right-hand side
of each column mapping. Set `date_format` per file and add a `values` block where a
category is coded differently (for example `CR`/`DR`). No code changes are needed.

## Layout

See `CLAUDE.md`. Explanations of every decision are in `docs/notes.md`; results per run
are in `docs/metrics.md`.
