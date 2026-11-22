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

from pulse.config import PROJECT_ROOT, current_mode, load_settings, resolve
from pulse.io import read_parquet, write_parquet
from pulse.models import churn
from pulse.models.survival import BANDS, load_products, urgency_band
from pulse import crosssell, explain
from pulse.pseudo import id_map, pseudonymise
from pulse.value import compute_value, summary_value

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


def fallback_urgency(p: np.ndarray, cfg: dict) -> np.ndarray:
    """Fallback mode: bands from the churn probability alone (no time-to-churn model)."""
    b = cfg["fallback_bands"]
    return np.where(p >= float(b["act_this_week_p"]), BANDS[0],
                    np.where(p >= float(b["act_this_month_p"]), BANDS[1], BANDS[2])).astype(object)


def fallback_recommendation(rows: pd.DataFrame, value_at_risk: np.ndarray, settings: dict, root: Path) -> pd.DataFrame:
    """Fallback mode: no product data, so the top share of value at risk gets the retention action
    and everyone else a review marker. Both texts are fixed templates from crosssell.yaml."""
    rules = crosssell.load_rules(root, settings)
    n = len(rows)
    rec = pd.DataFrame({"recommendation": "review", "rec_score": 0.0,
                        "rec_reason": rules.get("summary_review", "No product data in this dataset: review the relationship on the call."),
                        "rec_source": "none", "rec_trigger": ""}, index=range(n))
    k = max(1, int(np.floor(float(settings["crosssell"]["retention_top_share"]) * n)))
    top = np.argsort(-value_at_risk, kind="stable")[:k]
    for i in top:
        rec.loc[i, ["recommendation", "rec_score", "rec_reason", "rec_source"]] = [
            "retention", float(value_at_risk[i]),
            crosssell.fill(rules["retention"]["action"], {"value_at_risk_usd": value_at_risk[i]}), "retention"]
    return rec


def build_worklist(feats: pd.DataFrame, products: pd.DataFrame, settings: dict, model_dir: Path,
                   snapshot_date=None, mode: str = "transactions", root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = Path(model_dir)
    snapshot_date = pd.Timestamp(snapshot_date) if snapshot_date is not None else feats["snapshot_date"].max()
    rows = feats[feats["snapshot_date"] == snapshot_date].copy()
    if rows.empty:
        raise ValueError(f"no rows at snapshot {snapshot_date.date()}")
    rows = rows.drop_duplicates("customer_id").reset_index(drop=True)

    p = churn.predict(model_dir, rows)
    if mode == "summary":
        probs = None
        val = summary_value(rows, settings["value"])
    else:
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
    if probs is None:
        out["p_churn_30d"] = np.nan
        out["p_churn_60d"] = np.nan
        out["p_churn_90d_cox"] = np.nan
        out["urgency"] = fallback_urgency(out["p_churn_90d"].to_numpy(), settings["survival"])
        rec = fallback_recommendation(rows, out["value_at_risk_usd"].to_numpy(), settings, root)
    else:
        out["p_churn_30d"] = probs["p_churn_30d"].to_numpy()
        out["p_churn_60d"] = probs["p_churn_60d"].to_numpy()
        out["p_churn_90d_cox"] = probs["p_churn_90d"].to_numpy()
        out["urgency"] = urgency_band(probs, settings["survival"])
        rec = crosssell.recommend(rows, products, settings, model_dir, value_at_risk=out["value_at_risk_usd"].to_numpy())
    for c in rec.columns:
        out[c] = rec[c].to_numpy()
    for f in WHY_FEATURES:
        out[f] = rows[f].to_numpy() if f in rows.columns else np.nan
    out = out.sort_values("rank").reset_index(drop=True)
    return out, id_map(rows["customer_id"], pcfg["salt"], int(pcfg["length"]))


def main(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    settings = load_settings(root)
    t0 = time.perf_counter()
    processed = resolve(root, settings["data"]["processed_dir"])
    feats = read_parquet(processed / "features.parquet")
    feats["snapshot_date"] = pd.to_datetime(feats["snapshot_date"])
    products = load_products(root, settings)
    mode = current_mode(root)
    worklist, mapping = build_worklist(feats, products, settings, resolve(root, churn.MODEL_DIR), mode=mode, root=root)
    write_parquet(worklist, processed / "worklist.parquet")
    write_parquet(mapping, processed / "id_map.parquet")

    snap = worklist["snapshot_date"].iloc[0].date()
    print(f"worklist.parquet: {len(worklist):,} customers at snapshot {snap}; "
          f"total value at risk {worklist['value_at_risk_usd'].sum():,.0f} USD "
          f"of {worklist['value_12m_usd'].sum():,.0f} USD 12-month value")
    print("\nurgency bands\n" + worklist["urgency"].value_counts().reindex(["act this week", "act this month", "watch"]).to_string())
    print("\nrecommendations by source\n" + worklist.groupby(["rec_source", "recommendation"]).size().to_string())
    show = ["rank", "customer_ref", "segment", "urgency", "p_churn_90d", "value_12m_usd", "value_at_risk_usd",
            "recommendation", "rec_source"] + (WHY_FEATURES if mode != "summary" else [])
    top = worklist[show].head(20).copy()
    for c in [c for c in ["p_churn_90d"] + WHY_FEATURES if c in top.columns]:
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
