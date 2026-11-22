"""Fallback path: features from a customer-level table when there are no transactions.

schema.yaml maps one `customer_summary` table (see the commented block there). This module turns
it into the same shape as data/processed/features.parquet so that training, scoring, cards, the
API and the dashboard run unchanged:

- one row per customer, a single snapshot date (the mapped snapshot_date or settings end_date)
- the model inputs are exactly the columns mapped under `features:`; nulls are filled with the
  column median and a `<name>_missing` flag is added when more than 1% of rows were empty
- the label is the provided churn flag (`churn_90d` = flag, `labelled` = 1)

What Pulse CANNOT claim in this mode (printed by the CLI and written in docs/runbook.md):
- no leakage guarantee: the flag was built by someone else and the features may describe the
  customer after they left (a balance of zero because the account was closed, for example)
- no time-based validation: with one snapshot the split is stratified random, which tends to
  overstate how well the model will do on next quarter's customers
- no horizon: the flag means whatever the bank meant by it, not "leaves within 90 days"
- no urgency timing: bands come from the probability alone, there is no time-to-churn model
- no behavioural reasons: cards describe the mapped columns, not what changed in the account
- value at risk rests on the assumptions under `value.summary` in settings.yaml
- no next best product: every recommendation is a retention call
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pulse.config import PROJECT_ROOT, SUMMARY_TABLE, load_settings, resolve
from pulse.io import write_parquet

META_COLUMNS = ["customer_id", "snapshot_date", "month_idx", "segment", "branch"]
TARGET_COLUMNS = ["churn_90d", "churn_type", "days_to_churn", "labelled"]
MISSING_FLAG_MIN_SHARE = 0.01

CANNOT_CLAIM = [
    "no leakage guarantee: the churn flag was built outside Pulse and the columns may describe the customer after leaving",
    "no time-based validation: one snapshot, so the split is stratified random (optimistic for next quarter)",
    "no horizon: the flag means what the bank meant by it, not 'leaves within 90 days'",
    "no urgency timing: bands come from the probability alone (no time-to-churn model)",
    "no behavioural reasons: cards describe the mapped columns, not what changed in the account",
    "value at risk rests on the assumptions under value.summary in settings.yaml",
    "no next best product: every recommendation is a retention call",
]


def banner() -> str:
    lines = ["FALLBACK MODE: customer-level table, no transactions. Pulse cannot claim:"]
    lines += [f"  - {c}" for c in CANNOT_CLAIM]
    return "\n".join(lines)


def feature_names_from(df: pd.DataFrame) -> list[str]:
    """Model input columns of a summary-mode feature table: everything that is not meta or target."""
    return [c for c in df.columns if c not in META_COLUMNS + TARGET_COLUMNS]


def build_summary_features(table: pd.DataFrame, feature_cols: list[str], settings: dict) -> pd.DataFrame:
    """One feature row per customer from the loaded customer_summary table."""
    df = table.copy()
    snap = pd.to_datetime(df["snapshot_date"])
    if snap.isna().any():
        snap = snap.fillna(pd.Timestamp(settings["data"]["end_date"]))
    out = pd.DataFrame({
        "customer_id": df["customer_id"].astype(str),
        "snapshot_date": snap,
        "month_idx": 0,
        "segment": df["segment"].fillna("all").astype(str),
        "branch": df["branch"].fillna("all").astype(str),
    })
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        miss = col.isna()
        if miss.all():
            raise ValueError(f"feature {c!r} has no numeric values")
        if miss.mean() > MISSING_FLAG_MIN_SHARE:
            out[f"{c}_missing"] = miss.astype(float).to_numpy()
        out[c] = col.fillna(col.median()).to_numpy(float)
    out["churn_90d"] = pd.to_numeric(df["churn_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
    out["churn_type"] = np.where(out["churn_90d"] == 1, "flag", "")
    out["days_to_churn"] = np.nan
    out["labelled"] = 1
    return out.reset_index(drop=True)


def write_dictionary(path: Path, feature_cols: list[str], flags: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Feature dictionary (fallback mode)", "",
             "One row per customer at a single snapshot. The model inputs are the columns mapped under",
             "`features:` in schema.yaml, as the bank supplied them; Pulse did not compute them from",
             "transactions, so it cannot vouch for when they were measured relative to the churn flag.", "",
             "| column | meaning |", "| --- | --- |",
             "| customer_id | source customer id (pseudonymised before it reaches the dashboard) |",
             "| snapshot_date | the date the row describes |",
             "| churn_90d | the bank's churn flag (1 = churned), used as the label |"]
    lines += [f"| {c} | mapped from the source table (schema.yaml features: {c}) |" for c in feature_cols]
    lines += [f"| {f} | 1 when the source value was empty (filled with the median) |" for f in flags]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_table(root: Path, settings: dict) -> tuple[pd.DataFrame, list[str]]:
    """The customer_summary table and its mapped feature columns, from the ingest database."""
    con = duckdb.connect(str(resolve(root, settings["data"]["duckdb_path"])), read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {SUMMARY_TABLE}").df()
    finally:
        con.close()
    fixed = ["customer_id", "churn_flag", "snapshot_date", "segment", "branch", "gender", "age_band"]
    return df, [c for c in df.columns if c not in fixed]


def main(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    settings = load_settings(root)
    t0 = time.perf_counter()
    print(banner())
    table, feature_cols = load_table(root, settings)
    feats = build_summary_features(table, feature_cols, settings)
    out_dir = resolve(root, settings["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(feats, out_dir / "features.parquet")
    names = feature_names_from(feats)
    write_dictionary(root / "docs" / "features.md", feature_cols, [n for n in names if n.endswith("_missing")])
    print(f"features.parquet: {len(feats):,} customers x {len(names)} inputs ({', '.join(names)}); "
          f"churn rate {feats['churn_90d'].mean():.1%}; built in {time.perf_counter() - t0:.1f}s")
    return feats


if __name__ == "__main__":
    main()
