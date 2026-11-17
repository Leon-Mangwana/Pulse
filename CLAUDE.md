# Pulse

AI customer-intelligence copilot for the BancABC Universities Hackathon (4 Dec 2026).
From bank transaction data it predicts 90-day churn, ranks customers by value at risk
(P(churn) x expected 12-month value), recommends a next best product, and explains each
prediction in plain English. Delivered as a Streamlit dashboard backed by a FastAPI scoring API.

The project is built in phases. I will paste one phase prompt at a time. Work only on the
phase I give you.

## Who you are working with
This is my first machine-learning project. I am comfortable with Python, FastAPI and Docker.
My machine runs Windows. Use commands that work in PowerShell, and do not assume `make` is
installed: where a prompt says `make <step>`, provide an equivalent that works on Windows
(for example `python -m pulse.cli <step>`) and tell me the exact command.
- Explain every ML choice in plain language before you implement it.
- Prefer the simplest method that works. No deep learning, no AutoML, no extra frameworks.
- When a result looks too good (for example ROC-AUC above 0.95), assume a bug or leakage
  and investigate before celebrating.

## Hard rules
1. No leakage. Features for a prediction date use only data from before that date.
   Labels use only data after it. Write a test that proves this.
2. Time-based splits only. Train on earlier months, validate on later months. Never a random split.
3. All data access goes through `config/schema.yaml`, which maps our canonical column names
   to the source file's column names. No source column name appears anywhere else in the code.
4. Every step runs from the command line (data, ingest, features, train, score, pipeline)
   and the full pipeline must finish in under 10 minutes on a laptop.
5. Fixed random seeds. Same input gives the same output.
6. Protected attributes (gender, age, and similar) are never model inputs.
   Keep them aside only for the fairness check.
7. Customer IDs are pseudonymised before they reach the dashboard or logs.
8. Text shown to users comes from fixed templates. Never generate free text about a customer.

## Stack
Python 3.11, pandas, DuckDB, LightGBM, scikit-learn, lifelines, SHAP, FastAPI, Streamlit,
Docker Compose, pytest. Pin every version in requirements.txt.

## Layout
config/      schema.yaml, settings.yaml (thresholds, windows, assumptions)
data/        raw/, processed/ (both git-ignored)
src/pulse/   ingest.py, features/, target.py, models/, explain.py, crosssell.py, value.py
api/         FastAPI app
dashboard/   Streamlit app
tests/       pytest, including the leakage test
docs/        notes.md (my explanations), metrics.md (results per run)

## How to work
- Propose a short plan and wait for my go-ahead before writing code.
- Never commit. At the start of a phase create a git branch for it (for example
  `phase-1-features`) and leave all committing to me. Run the tests before saying a task is done.
- After each task, append results to docs/metrics.md and tell me what to check by hand.
