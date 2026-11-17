"""Engagement features: how often and through which channels the customer uses the bank."""
from __future__ import annotations

import numpy as np

from pulse.features.common import lag, months_since, roll_mean, roll_sum, safe_div

FEATURES = {
    "active_days_1m": "days with at least one transaction this month",
    "active_days_3m_avg": "average active days per month over the last 3 months",
    "active_days_change": "this month's active days divided by the monthly average of the 6 months before (1 = unchanged)",
    "txn_count_1m": "transactions this month",
    "txn_count_3m": "transactions over the last 3 months",
    "txn_count_6m_avg": "average transactions per month over the last 6 months",
    "txn_count_change": "this month's transactions divided by the monthly average of the 6 months before",
    "txn_3m_vs_prior_3m": "transactions in the last 3 months divided by the 3 months before",
    "n_credits_1m": "credits this month",
    "n_debits_1m": "debits this month",
    "digital_share_3m": "share of the last 3 months' transactions made on mobile app, USSD or ZIPIT",
    "digital_share_change_6m": "digital share now minus 6 months ago",
    "branch_share_3m": "share of the last 3 months' transactions made in a branch",
    "atm_share_3m": "share of the last 3 months' transactions at an ATM",
    "pos_share_3m": "share of the last 3 months' transactions at a card terminal",
    "months_since_last_txn": "months since the last month with any transaction (0 = this month, capped at 12)",
    "tenure_months": "months since the customer opened their first account",
}


def compute(mt, cfg: dict) -> dict[str, np.ndarray]:
    A, ex = mt.A, mt.exists
    w = cfg["baseline_months"]
    n_txn = A["n_txn"]
    base_days = lag(roll_mean(A["active_days"], w, ex), 1)
    base_txn = lag(roll_mean(n_txn, w, ex), 1)
    txn3 = roll_sum(n_txn, 3)
    dig3 = safe_div(roll_sum(A["n_digital"], 3), txn3)
    months = np.arange(mt.M)[None, :]
    return {
        "active_days_1m": A["active_days"],
        "active_days_3m_avg": roll_sum(A["active_days"], 3) / 3,
        "active_days_change": np.clip(safe_div(A["active_days"], base_days, 1.0), 0, 5),
        "txn_count_1m": n_txn,
        "txn_count_3m": txn3,
        "txn_count_6m_avg": np.nan_to_num(roll_mean(n_txn, 6, ex)),
        "txn_count_change": np.clip(safe_div(n_txn, base_txn, 1.0), 0, 5),
        "txn_3m_vs_prior_3m": np.clip(safe_div(txn3, lag(txn3, 3), 1.0), 0, 5),
        "n_credits_1m": A["n_credit"],
        "n_debits_1m": A["n_debit"],
        "digital_share_3m": dig3,
        "digital_share_change_6m": np.nan_to_num(dig3 - lag(dig3, 6)),
        "branch_share_3m": safe_div(roll_sum(A["n_branch"], 3), txn3),
        "atm_share_3m": safe_div(roll_sum(A["n_atm"], 3), txn3),
        "pos_share_3m": safe_div(roll_sum(A["n_pos"], 3), txn3),
        "months_since_last_txn": months_since(n_txn > 0, cfg["cap_months"]),
        "tenure_months": (months - mt.open_m[:, None] + 1).astype(float),
    }
