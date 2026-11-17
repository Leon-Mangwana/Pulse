"""Income features: is there a salary, has it stopped, shrunk, or changed payer.

Salary is detected from the PATTERN (a large credit of similar size most months),
not from any label, because the bank's data may not label employer credits.
"""
from __future__ import annotations

import numpy as np

from pulse.features.common import lag, months_since, nan_reduce, roll_mean, roll_sum, safe_div, stack_lags

FEATURES = {
    "salary_detected": "1 if a credit of similar size arrived in at least 5 of the last 12 months (or 4 of the last 6)",
    "salary_amount_usd": "typical size of that regular credit over the last 12 months, USD (0 if none)",
    "months_since_salary": "months since the last credit that looked like the salary (0 = this month, capped at 12)",
    "salary_change_1m": "this month's largest credit divided by the typical salary (1 = unchanged, 0 = nothing arrived)",
    "salary_trend_3m": "average largest credit of the last 3 months divided by the 3 months before that",
    "n_salary_months_6m": "how many of the last 6 months had a salary-sized credit",
    "payer_changed": "1 if the most recent salary came from a different payer than the one before",
    "income_1m_usd": "all money in this month, excluding transfers from the customer's own accounts, USD",
    "income_3m_usd": "all money in over the last 3 months, USD",
    "income_change": "this month's income divided by the monthly average of the 6 months before it (1 = unchanged)",
    "top_credit_1m_usd": "largest single credit this month, USD",
}


def compute(mt, cfg: dict) -> dict[str, np.ndarray]:
    A, ex = mt.A, mt.exists
    tol, floor = cfg["salary_tolerance"], cfg["salary_min_usd"]
    cap = cfg["cap_months"]
    top = A["top_credit"]

    L12 = stack_lags(top, 12)
    L12 = np.where(L12 > 0, L12, np.nan)
    ref = nan_reduce(np.nanmedian, L12)
    ref = np.where(np.isnan(ref) | (ref < floor), 0.0, ref)
    close = (np.abs(L12 - ref[..., None]) <= tol * ref[..., None]) & (ref[..., None] > 0)
    hits12 = close.sum(-1)
    hits6 = close[..., :6].sum(-1)
    detected = (hits12 >= 5) | (hits6 >= 4)
    hit_now = (top >= (1 - tol) * ref) & (ref > 0) & (top > 0)
    since = months_since(hit_now, cap)

    # payer: compare the key of the latest salary-like credit with the one before it
    keys = mt.keys
    key_lags = np.stack([lag_obj(keys, k) for k in range(cap)], axis=-1)
    hit_lags = np.nan_to_num(stack_lags(hit_now.astype(float), cap)) > 0
    latest = np.full(keys.shape, "", dtype=object)
    prev = np.full(keys.shape, "", dtype=object)
    found = np.zeros(keys.shape, bool)
    for k in range(cap):
        h = hit_lags[..., k]
        prev = np.where(found & h & (prev == ""), key_lags[..., k], prev)
        latest = np.where(~found & h, key_lags[..., k], latest)
        found = found | h
    payer_changed = (latest != "") & (prev != "") & (latest != prev)

    base_income = lag(roll_mean(A["credit_usd"], cfg["baseline_months"], ex), 1)
    return {
        "salary_detected": detected.astype(float),
        "salary_amount_usd": np.where(detected, ref, 0.0),
        "months_since_salary": np.where(detected | (ref > 0), since, cap),
        "salary_change_1m": np.clip(safe_div(top, ref, 1.0), 0, 3),
        "salary_trend_3m": np.clip(safe_div(roll_sum(top, 3), lag(roll_sum(top, 3), 3), 1.0), 0, 3),
        "n_salary_months_6m": hits6.astype(float),
        "payer_changed": payer_changed.astype(float),
        "income_1m_usd": A["credit_usd"],
        "income_3m_usd": roll_sum(A["credit_usd"], 3),
        "income_change": np.clip(safe_div(A["credit_usd"], base_income, 1.0), 0, 5),
        "top_credit_1m_usd": top,
    }


def lag_obj(K: np.ndarray, k: int) -> np.ndarray:
    out = np.full(K.shape, "", dtype=object)
    if k == 0:
        return K.copy()
    out[:, k:] = K[:, :-k]
    return out
