"""Expected 12-month value of a customer to the bank, and the call-list comparison.

value_12m_usd = fee income + interest margin on balances + margin on products held.
Every rate lives in settings.yaml under `value`, with a comment, so finance can change it
without touching code. Balances and fees are already in USD (ZiG converted in the feature
step at `fx.zig_per_usd`). Products are counted only if they started on or before the
snapshot date and had not ended by then.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = {
    "fee_income_12m_usd": "fees paid in the last 3 months x fee_annualisation",
    "interest_margin_12m_usd": "6-month average balance x yearly net interest margin (USD and ZiG margins blended by the USD share)",
    "product_margin_12m_usd": "sum of the yearly margin of every product held at the snapshot date",
    "value_12m_usd": "the three above added, floored at min_value_usd",
}


def unknown_products(products: pd.DataFrame, cfg: dict) -> list[str]:
    """Product names in the data with no margin in settings (they count as 0)."""
    return sorted(set(products["product"].unique()) - set(cfg["product_margin_usd"]))


def product_margin(rows: pd.DataFrame, products: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Yearly product margin per row of (customer_id, snapshot_date), USD."""
    margins = {k: float(v) for k, v in cfg["product_margin_usd"].items()}
    key = rows[["customer_id", "snapshot_date"]].copy()
    key["_i"] = np.arange(len(rows))
    m = key.merge(products[["customer_id", "product", "start_date", "end_date"]], on="customer_id", how="inner")
    held = (m["start_date"] <= m["snapshot_date"]) & (m["end_date"].isna() | (m["end_date"] > m["snapshot_date"]))
    m = m[held]
    out = np.zeros(len(rows))
    if len(m):
        s = m["product"].map(margins).fillna(0.0).groupby(m["_i"]).sum()
        out[s.index.to_numpy()] = s.to_numpy()
    return out


def compute_value(rows: pd.DataFrame, products: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per input row with the three components and the total."""
    fee = rows["fees_3m_usd"].to_numpy(float) * float(cfg["fee_annualisation"])
    share = rows["usd_share"].to_numpy(float).clip(0, 1)
    margin = share * float(cfg["interest_margin_usd"]) + (1 - share) * float(cfg["interest_margin_zig"])
    interest = rows["bal_avg_6m_usd"].to_numpy(float) * margin
    prod = product_margin(rows, products, cfg)
    total = np.maximum(fee + interest + prod, float(cfg["min_value_usd"]))
    return pd.DataFrame({"fee_income_12m_usd": fee, "interest_margin_12m_usd": interest,
                         "product_margin_12m_usd": prod, "value_12m_usd": total}, index=rows.index)


def call_list_comparison(y: np.ndarray, p: np.ndarray, value: np.ndarray, shares: list[float], seed: int) -> pd.DataFrame:
    """Dollars of 12-month value belonging to customers who actually churned, reached by contacting
    the top share of the list under three orderings: random, by churn probability, by value at risk."""
    y = np.asarray(y, int)
    n = len(y)
    churned_value = np.where(y == 1, value, 0.0)
    total_value, total_churners = churned_value.sum(), y.sum()
    rng = np.random.default_rng(seed)
    orders = {
        "random": rng.permutation(n),
        "by churn probability": np.argsort(-p, kind="stable"),
        "by value at risk": np.argsort(-(p * value), kind="stable"),
    }
    rows = []
    for name, order in orders.items():
        for s in shares:
            k = max(1, int(np.floor(s * n)))
            top = order[:k]
            rows.append({"ordering": name, "top_%": int(round(100 * s)), "contacted": k,
                         "churned_value_reached_usd": round(float(churned_value[top].sum())),
                         "share_of_churned_value_%": round(100 * churned_value[top].sum() / total_value, 1) if total_value else float("nan"),
                         "churners_reached_%": round(100 * y[top].sum() / total_churners, 1) if total_churners else float("nan")})
    return pd.DataFrame(rows)
