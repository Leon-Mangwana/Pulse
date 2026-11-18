"""Build the snapshot table: one row per customer per month-end, ~60 features and the label.

    python -m pulse.cli features

Reads customers, accounts and transactions from DuckDB (never _truth.csv), writes
data/processed/features.parquet and docs/features.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pulse import target
from pulse.config import PROJECT_ROOT, load_settings, resolve
from pulse.io import write_parquet
from pulse.features import balance, engagement, friction, income, leakage, life_events
from pulse.features.monthly import MonthlyTable, build_monthly

FAMILIES = [
    ("income", income), ("balance", balance), ("leakage", leakage),
    ("engagement", engagement), ("friction", friction), ("life_events", life_events),
]
META_COLUMNS = ["customer_id", "snapshot_date", "month_idx", "segment", "branch"]


def feature_names() -> list[str]:
    return [name for _, mod in FAMILIES for name in mod.FEATURES]


def build_features(tx: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame,
                   settings: dict, keep_inactive: bool = False) -> pd.DataFrame:
    """Pure function from raw frames to the snapshot table (used by the CLI and by tests)."""
    mt = build_monthly(tx, accounts, customers, settings)
    fcfg, tcfg = settings["features"], settings["target"]
    cols: dict[str, np.ndarray] = {}
    for _, mod in FAMILIES:
        for name, arr in mod.compute(mt, fcfg).items():
            cols[name] = np.asarray(arr, dtype=float)
    labels = target.compute_target(mt, tcfg)

    # which (customer, month) cells become rows
    n, M = mt.n, mt.M
    mm = np.arange(M)[None, :]
    history = mm - np.maximum(mt.open_m, 0)[:, None] + 1
    eligible = mt.exists & (history >= tcfg["min_history_months"])
    if not keep_inactive:
        eligible &= ~target.already_inactive(mt, tcfg)   # trivial to predict, useless to act on
        eligible &= ~target.already_churned(mt, tcfg)    # already gone: nothing left to predict
    ci, mi = np.nonzero(eligible)

    cust = customers.reset_index(drop=True)
    out = pd.DataFrame({
        "customer_id": mt.ids[ci],
        "snapshot_date": mt.month_starts[1:].to_numpy()[mi] - np.timedelta64(1, "D"),
        "month_idx": mi,
        "segment": cust["segment"].to_numpy()[ci],
        "branch": cust["branch"].to_numpy()[ci],
    })
    for name, arr in cols.items():
        out[name] = np.nan_to_num(arr[ci, mi], nan=0.0, posinf=0.0, neginf=0.0)
    for name, arr in labels.items():
        out[name] = arr[ci, mi]
    return out


def write_dictionary(path: Path) -> None:
    lines = ["# Feature dictionary", "",
             "One row per customer per month-end snapshot. Every feature uses only transactions dated",
             "before the snapshot date. Money is in USD (ZiG converted at `fx.zig_per_usd` in settings.yaml).",
             "Ratios 'versus baseline' compare the latest month with the customer's own average over the",
             "6 months before it; 1 means no change. Missing values are filled with 0 for amounts and",
             "shares, and with 1 for change ratios.", ""]
    lines += ["## Identifiers and metadata (not model inputs)", "",
              "| column | meaning |", "| --- | --- |",
              "| customer_id | raw customer id (pseudonymised before any dashboard or log) |",
              "| snapshot_date | last day of the month the features describe |",
              "| month_idx | 0-based month number from the data start |",
              "| segment | customer segment from the customers table |",
              "| branch | home branch from the customers table |", ""]
    for fam, mod in FAMILIES:
        lines += [f"## {fam.replace('_', ' ').title()}", "", "| feature | meaning |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in mod.FEATURES.items()]
        lines.append("")
    lines += ["## Target", "", "| column | meaning |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in target.TARGET_COLUMNS.items()]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    settings = load_settings(root)
    t0 = time.perf_counter()
    con = duckdb.connect(str(resolve(root, settings["data"]["duckdb_path"])), read_only=True)
    try:
        tx = con.execute("SELECT * FROM transactions").df()
        accounts = con.execute("SELECT * FROM accounts").df()
        customers = con.execute("SELECT customer_id, open_date, segment, branch FROM customers").df()
    finally:
        con.close()
    for df, cols in [(tx, ["txn_date"]), (accounts, ["open_date", "close_date"]), (customers, ["open_date"])]:
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    print(f"loaded {len(tx):,} transactions in {time.perf_counter() - t0:.1f}s")

    feats = build_features(tx, accounts, customers, settings)
    out_dir = resolve(root, settings["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(feats, out_dir / "features.parquet")
    write_dictionary(root / "docs" / "features.md")

    names = feature_names()
    lab = feats[feats["labelled"] == 1]
    print(f"features.parquet: {len(feats):,} rows x {len(names)} features, "
          f"{feats['customer_id'].nunique():,} customers, {lab.shape[0]:,} labelled rows")
    by_month = lab.groupby("snapshot_date")["churn_90d"].agg(["count", "mean"])
    by_month["mean"] = (by_month["mean"] * 100).round(1)
    by_month.columns = ["rows", "churn_%"]
    print("\nchurn rate by snapshot month\n" + by_month.to_string())
    print(f"\noverall churn rate: {lab['churn_90d'].mean():.1%}  "
          f"(hard {(lab['churn_type'] == 'hard').mean():.1%}, silent {(lab['churn_type'] == 'silent').mean():.1%})")
    print(f"built in {time.perf_counter() - t0:.1f}s")
    return feats


if __name__ == "__main__":
    main()
