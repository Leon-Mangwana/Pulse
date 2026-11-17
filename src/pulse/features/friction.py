"""Friction features: fees paid and transactions that bounced."""
from __future__ import annotations

import numpy as np

from pulse.features.common import lag, roll_mean, roll_sum, safe_div

FEATURES = {
    "fees_1m_usd": "bank fees charged this month, USD",
    "fees_3m_usd": "bank fees charged over the last 3 months, USD",
    "fee_share_3m": "fees as a share of all money out over the last 3 months",
    "fee_change": "this month's fees divided by the monthly average of the 6 months before (1 = unchanged)",
    "reversals_3m": "reversed transactions in the last 3 months",
    "reversals_6m": "reversed transactions in the last 6 months",
}


def compute(mt, cfg: dict) -> dict[str, np.ndarray]:
    A, ex = mt.A, mt.exists
    fees = A["fee_out"]
    base = lag(roll_mean(fees, cfg["baseline_months"], ex), 1)
    return {
        "fees_1m_usd": fees,
        "fees_3m_usd": roll_sum(fees, 3),
        "fee_share_3m": safe_div(roll_sum(fees, 3), roll_sum(A["debit_usd"] + fees, 3)),
        "fee_change": np.clip(safe_div(fees, base, 1.0), 0, 5),
        "reversals_3m": roll_sum(A["n_reversal"], 3),
        "reversals_6m": roll_sum(A["n_reversal"], 6),
    }
