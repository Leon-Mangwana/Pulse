"""Leakage features: how much of the customer's spending leaves for mobile wallets and other banks."""
from __future__ import annotations

import numpy as np

from pulse.features.common import lag, roll_sum, safe_div

FEATURES = {
    "wallet_share_1m": "share of this month's outflows sent to mobile wallets (EcoCash and similar)",
    "wallet_share_3m": "share of the last 3 months' outflows sent to mobile wallets",
    "wallet_share_trend": "wallet share of the last 3 months minus the 3 months before (positive = rising)",
    "obank_share_1m": "share of this month's outflows sent to other banks",
    "obank_share_3m": "share of the last 3 months' outflows sent to other banks",
    "obank_share_trend": "other-bank share of the last 3 months minus the 3 months before",
    "leak_share_3m": "share of the last 3 months' outflows that left to wallets or other banks combined",
    "leak_share_6m": "the same share over the last 6 months",
    "leak_share_trend": "combined leak share, last 3 months minus the 3 months before",
    "leak_amount_3m_usd": "money sent to wallets and other banks in the last 3 months, USD",
    "cash_share_3m": "share of the last 3 months' outflows taken as cash",
}


def compute(mt, cfg: dict) -> dict[str, np.ndarray]:
    A = mt.A
    out = A["debit_usd"]
    out3, out6 = roll_sum(out, 3), roll_sum(out, 6)
    w1 = safe_div(A["wallet_out"], out)
    w3 = safe_div(roll_sum(A["wallet_out"], 3), out3)
    b1 = safe_div(A["obank_out"], out)
    b3 = safe_div(roll_sum(A["obank_out"], 3), out3)
    leak = A["wallet_out"] + A["obank_out"]
    l3 = safe_div(roll_sum(leak, 3), out3)
    return {
        "wallet_share_1m": w1,
        "wallet_share_3m": w3,
        "wallet_share_trend": np.nan_to_num(w3 - lag(w3, 3)),
        "obank_share_1m": b1,
        "obank_share_3m": b3,
        "obank_share_trend": np.nan_to_num(b3 - lag(b3, 3)),
        "leak_share_3m": l3,
        "leak_share_6m": safe_div(roll_sum(leak, 6), out6),
        "leak_share_trend": np.nan_to_num(l3 - lag(l3, 3)),
        "leak_amount_3m_usd": roll_sum(leak, 3),
        "cash_share_3m": safe_div(roll_sum(A["cash_out"], 3), out3),
    }
