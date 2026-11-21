"""Dashboard arithmetic and table shaping, with no Streamlit import so it can be tested anywhere.

Everything the pages compute lives here: headline numbers, the worklist table in plain words,
the ROI curve (same formulas as POST /simulate) and the long-format frames the charts draw.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

AT_RISK_BANDS = ("act this week", "act this month")   # the urgency bands that count as "at risk"
TOP_SHARE = 0.10                                       # headline: share of value at risk in the top 10%

WORKLIST_COLUMNS = {           # API field -> label on screen (no raw feature names)
    "rank": "Rank",
    "customer_ref": "Customer",
    "branch": "Branch",
    "segment": "Segment",
    "risk_pct": "Risk in 90 days",
    "urgency": "Urgency",
    "value_at_risk_usd": "Value at risk (USD)",
    "recommendation": "Next action",
}

COMPARISON_COLUMNS = {
    "model": "Model",
    "PR-AUC": "Precision-recall AUC",
    "ROC-AUC": "ROC AUC",
    "capture@10%": "Churners caught in top 10% (%)",
    "lift@10%": "Lift in top 10%",
    "capture@20%": "Churners caught in top 20% (%)",
}
MODEL_LABELS = {"rules": "Rules of thumb", "logistic": "Logistic regression", "lightgbm": "Pulse (LightGBM)"}

FAIRNESS_COLUMNS = {
    "attribute": "Attribute",
    "group": "Group",
    "rows": "Customer-months",
    "churn_%": "Actual churn (%)",
    "flag_%": "Flagged in top 10% (%)",
    "capture_%": "Churners caught (%)",
}

SEGMENT_COLUMNS = {
    "group": "Group",
    "customers": "Customers",
    "churn_rate_pct": "Expected churn (%)",
    "value_at_risk_usd": "Value at risk (USD)",
    "share_of_var_pct": "Share of value at risk (%)",
    "var_per_customer_usd": "Value at risk per customer (USD)",
}
DIMENSION_LABELS = {"branch": "Branch", "segment": "Segment", "product": "Product held"}


def usd(x: float, decimals: int = 0) -> str:
    return f"{x:,.{decimals}f} USD"


def pct(x: float, decimals: int = 0) -> str:
    return f"{100 * x:.{decimals}f}%"


# ----------------------------------------------------------------------------- worklist page
def headline(rows: pd.DataFrame) -> dict:
    """Customers at risk, total value at risk and the share of it held by the top 10% of the ranking."""
    if rows.empty:
        return {"customers": 0, "at_risk": 0, "value_at_risk_usd": 0.0, "top_share": 0.0, "top_n": 0}
    ordered = rows.sort_values("value_at_risk_usd", ascending=False)
    total = float(ordered["value_at_risk_usd"].sum())
    top_n = max(1, math.ceil(TOP_SHARE * len(ordered)))
    top = float(ordered["value_at_risk_usd"].head(top_n).sum())
    return {"customers": int(len(rows)), "at_risk": int(rows["urgency"].isin(AT_RISK_BANDS).sum()),
            "value_at_risk_usd": total, "top_share": top / total if total > 0 else 0.0, "top_n": top_n}


def worklist_table(rows: pd.DataFrame) -> pd.DataFrame:
    """The ranked table in plain words; risk as a percentage number for the progress column."""
    df = rows.copy()
    df["risk_pct"] = (100 * df["p_churn_90d"]).round(1)
    df = df[list(WORKLIST_COLUMNS)].rename(columns=WORKLIST_COLUMNS)
    return df.sort_values("Rank").reset_index(drop=True)


def filter_rows(rows: pd.DataFrame, branch: str | None, segment: str | None) -> pd.DataFrame:
    df = rows
    if branch:
        df = df[df["branch"] == branch]
    if segment:
        df = df[df["segment"] == segment]
    return df


# ----------------------------------------------------------------------------- customer card
def history_long(points: list[dict]) -> pd.DataFrame:
    """Long format for the card chart: one row per month per series."""
    df = pd.DataFrame(points)
    if df.empty:
        return pd.DataFrame(columns=["month", "series", "usd"])
    df["month"] = pd.to_datetime(df["month"])
    out = pd.concat([
        pd.DataFrame({"month": df["month"], "series": "Month-end balance", "usd": df["balance_usd"]}),
        pd.DataFrame({"month": df["month"], "series": "Salary credit", "usd": df["salary_credit_usd"]}),
        pd.DataFrame({"month": df["month"], "series": "All money in", "usd": df["money_in_usd"]}),
    ], ignore_index=True)
    return out


# ----------------------------------------------------------------------------- insights page
def comparison_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["model"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    cols = [c for c in COMPARISON_COLUMNS if c in df.columns]
    return df[cols].rename(columns=COMPARISON_COLUMNS)


def fairness_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = [c for c in FAIRNESS_COLUMNS if c in df.columns]
    return df[cols].rename(columns=FAIRNESS_COLUMNS)


def segment_table(rows: list[dict], dimension: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[df["dimension"] == dimension].sort_values("value_at_risk_usd", ascending=False)
    cols = [c for c in SEGMENT_COLUMNS if c in df.columns]
    return df[cols].rename(columns=SEGMENT_COLUMNS).reset_index(drop=True)


def drivers_frame(families: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(families)
    return df.rename(columns={"label": "Driver", "share_pct": "Share of model impact (%)",
                              "customers_pushed_up_pct": "Customers it pushes up (%)"})


# ----------------------------------------------------------------------------- ROI simulator
def outcome(p: np.ndarray, v: np.ndarray, contacted: int, cost: float, incentive: float, save_rate: float) -> dict:
    """Same arithmetic as the API's simulate(): saves = p x save_rate, protected = saves x value."""
    churners = float(p.sum())
    saves = churners * save_rate
    protected = float((p * v).sum()) * save_rate
    total_cost = contacted * cost + saves * incentive
    return {"contacted": int(contacted), "expected_saves": saves, "revenue_protected_usd": protected,
            "total_cost_usd": total_cost, "net_return_usd": protected - total_cost}


def roi_curve(p: np.ndarray, v: np.ndarray, cost: float, incentive: float, save_rate: float, points: int = 80) -> pd.DataFrame:
    """Net return against customers contacted, for Pulse ranking (cumulative down the worklist order)
    and for random outreach (a straight line through the population averages)."""
    n_total = len(p)
    if n_total == 0:
        return pd.DataFrame(columns=["contacted", "outreach", "net_return_usd"])
    cum_p = np.concatenate([[0.0], np.cumsum(p)])
    cum_pv = np.concatenate([[0.0], np.cumsum(p * v)])
    grid = np.unique(np.linspace(0, n_total, num=min(points, n_total + 1)).round().astype(int))
    pulse_net = cum_pv[grid] * save_rate - (grid * cost + cum_p[grid] * save_rate * incentive)
    mean_p = p.mean()
    mean_v_churner = float((p * v).sum() / p.sum()) if p.sum() > 0 else 0.0
    per_customer = mean_p * save_rate * mean_v_churner - cost - mean_p * save_rate * incentive
    random_net = grid * per_customer
    return pd.concat([
        pd.DataFrame({"contacted": grid, "outreach": "Pulse ranking", "net_return_usd": pulse_net}),
        pd.DataFrame({"contacted": grid, "outreach": "Random outreach", "net_return_usd": random_net}),
    ], ignore_index=True)


def best_point(curve: pd.DataFrame) -> dict:
    """The number of contacts with the highest net return on the Pulse line."""
    pulse = curve[curve["outreach"] == "Pulse ranking"]
    if pulse.empty:
        return {"contacted": 0, "net_return_usd": 0.0}
    row = pulse.loc[pulse["net_return_usd"].idxmax()]
    return {"contacted": int(row["contacted"]), "net_return_usd": float(row["net_return_usd"])}
