"""Build the worklist: one row per customer at the latest snapshot, sorted by value at risk.

    python -m pulse.cli score

value_at_risk_usd = calibrated 90-day churn probability x expected 12-month value.
Customer ids are pseudonymised here (rule 7); the mapping goes to a separate file.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pulse.config import PROJECT_ROOT, load_settings, resolve
from pulse.io import read_parquet, write_parquet
from pulse.models import churn
from pulse.models.survival import load_products, urgency_band
from pulse import crosssell, explain
from pulse.pseudo import id_map, pseudonymise
from pulse.value import compute_value

WHY_FEATURES = ["bal_change_vs_6m", "months_since_salary", "txn_count_change", "leak_share_trend"]
WORKLIST_COLUMNS = {
    "customer_ref": "pseudonymised customer reference (salted hash; mapping kept in id_map.parquet)",
    "snapshot_date": "the month-end the row describes",
    "segment": "customer segment", "branch": "home branch",
    "p_churn_90d": "calibrated probability of churn in the next 90 days (LightGBM)",
    "value_12m_usd": "expected 12-month value to the bank, USD",
    "fee_income_12m_usd": "of which fee income", "interest_margin_12m_usd": "of which interest margin on balances",
    "product_margin_12m_usd": "of which margin on products held",
    "value_at_risk_usd": "p_churn_90d x value_12m_usd",
    "rank": "1 = highest value at risk",
    "p_churn_30d": "Cox probability of churn within 30 days", "p_churn_60d": "within 60 days",
    "p_churn_90d_cox": "within 90 days (Cox; the calibrated LightGBM figure is the one to quote)",
    "urgency": "act this week / act this month / watch",
    "recommendation": "next best product, or 'retention' for the top decile of value at risk",
    "rec_score": "calibrated 90-day uptake probability when a model exists, else the rule's observed uptake rate",
    "rec_reason": "fixed template filled with the customer's own numbers (config/crosssell.yaml)",
    "rec_source": "rule, rule+model, model, fallback or retention",
    **{f: "why: see docs/features.md" for f in WHY_FEATURES},
}


def build_worklist(feats: pd.DataFrame, products: pd.DataFrame, settings: dict, model_dir: Path,
                   snapshot_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = Path(model_dir)
    snapshot_date = pd.Timestamp(snapshot_date) if snapshot_date is not None else feats["snapshot_date"].max()
    rows = feats[feats["snapshot_date"] == snapshot_date].copy()
    if rows.empty:
        raise ValueError(f"no rows at snapshot {snapshot_date.date()}")
    rows = rows.drop_duplicates("customer_id").reset_index(drop=True)

    p = churn.predict(model_dir, rows)
    cox = joblib.load(model_dir / "survival.joblib")
    probs = cox.predict_probs(rows)
    val = compute_value(rows, products, settings["value"])
    pcfg = settings["pseudonymise"]

    out = pd.DataFrame({
        "customer_ref": pseudonymise(rows["customer_id"], pcfg["salt"], int(pcfg["length"])),
        "snapshot_date": rows["snapshot_date"], "segment": rows["segment"], "branch": rows["branch"],
        "p_churn_90d": p,
    })
    for c in val.columns:
        out[c] = val[c].to_numpy()
    out["value_at_risk_usd"] = out["p_churn_90d"] * out["value_12m_usd"]
    out["rank"] = out["value_at_risk_usd"].rank(ascending=False, method="first").astype(int)
    out["p_churn_30d"] = probs["p_churn_30d"].to_numpy()
    out["p_churn_60d"] = probs["p_churn_60d"].to_numpy()
    out["p_churn_90d_cox"] = probs["p_churn_90d"].to_numpy()
    out["urgency"] = urgency_band(probs, settings["survival"])
    rec = crosssell.recommend(rows, products, settings, model_dir, value_at_risk=out["value_at_risk_usd"].to_numpy())
    for c in rec.columns:
        out[c] = rec[c].to_numpy()
    for f in WHY_FEATURES:
        out[f] = rows[f].to_numpy()
    out = out.sort_values("rank").reset_index(drop=True)
    return out, id_map(rows["customer_id"], pcfg["salt"], int(pcfg["length"]))


def main(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    settings = load_settings(root)
    t0 = time.perf_counter()
    processed = resolve(root, settings["data"]["processed_dir"])
    feats = read_parquet(processed / "features.parquet")
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    products = load_products(root, settings)
    worklist, mapping = build_worklist(feats, products, settings, resolve(root, churn.MODEL_DIR))
    write_parquet(worklist, processed / "worklist.parquet")
    write_parquet(mapping, processed / "id_map.parquet")

    snap = worklist["snapshot_date"].iloc[0].date()
    print(f"worklist.parquet: {len(worklist):,} customers at snapshot {snap}; "
          f"total value at risk {worklist['value_at_risk_usd'].sum():,.0f} USD "
          f"of {worklist['value_12m_usd'].sum():,.0f} USD 12-month value")
    print("\nurgency bands\n" + worklist["urgency"].value_counts().reindex(["act this week", "act this month", "watch"]).to_string())
    print("\nrecommendations by source\n" + worklist.groupby(["rec_source", "recommendation"]).size().to_string())
    show = ["rank", "customer_ref", "segment", "urgency", "p_churn_90d", "value_12m_usd", "value_at_risk_usd",
            "recommendation", "rec_source"] + WHY_FEATURES
    top = worklist[show].head(20).copy()
    for c in ["p_churn_90d"] + WHY_FEATURES:
        top[c] = top[c].round(2)
    for c in ["value_12m_usd", "value_at_risk_usd"]:
        top[c] = top[c].round(0)
    print("\ntop 20 by value at risk\n" + top.to_string(index=False))
    print(f"\nscored in {time.perf_counter() - t0:.1f}s")

    print("\n=== reason codes, action cards and segment view ===")
    rows = feats[feats["snapshot_date"] == pd.Timestamp(snap)]
    explain.run(worklist, rows, products, settings, resolve(root, churn.MODEL_DIR), processed, root)
    return worklist


if __name__ == "__main__":
    main()
