"""Balance features: level, trend and currency mix of month-end balances, in USD."""
from __future__ import annotations

import numpy as np

from pulse.features.common import lag, roll_max, roll_min, roll_nanmean, roll_slope, roll_sum, safe_div

FEATURES = {
    "bal_end_usd": "month-end balance across all accounts, USD",
    "bal_avg_3m_usd": "average month-end balance over the last 3 months, USD",
    "bal_avg_6m_usd": "average month-end balance over the last 6 months, USD",
    "bal_slope_6m": "monthly change in balance over the last 6 months as a share of the 6-month average (negative = falling)",
    "bal_change_vs_6m": "current balance divided by the average of the 6 months before (1 = unchanged)",
    "bal_drop_from_3m_peak": "share of the 3-month peak balance that has gone (0 = at peak, 0.8 = 80% gone)",
    "bal_min_3m_ratio": "lowest month-end balance of the last 3 months divided by the 6-month average",
    "net_flow_1m_usd": "money in minus money out this month, USD",
    "net_flow_3m_usd": "money in minus money out over the last 3 months, USD",
    "usd_share": "share of the balance held in USD rather than ZiG (0 to 1)",
    "usd_share_change_6m": "change in the USD share versus 6 months ago (positive = moving into USD)",
}


def compute(mt, cfg: dict) -> dict[str, np.ndarray]:
    A = mt.A
    bal = A["bal_end"]
    avg3, avg6 = roll_nanmean(bal, 3), roll_nanmean(bal, 6)
    base6 = lag(avg6, 1)
    peak3 = roll_max(bal, 3)
    usd_share = safe_div(np.nan_to_num(A["bal_usd_end"]), np.nan_to_num(A["bal_usd_end"]) + np.nan_to_num(A["bal_zig_end"]), 0.0)
    net = A["credit_usd"] - A["debit_usd"] - A["fee_out"]
    return {
        "bal_end_usd": np.nan_to_num(bal),
        "bal_avg_3m_usd": np.nan_to_num(avg3),
        "bal_avg_6m_usd": np.nan_to_num(avg6),
        "bal_slope_6m": np.clip(safe_div(roll_slope(bal, 6), avg6 + 1.0, 0.0), -2, 2),
        "bal_change_vs_6m": np.clip(safe_div(np.nan_to_num(bal), base6 + 1.0, 1.0), 0, 5),
        "bal_drop_from_3m_peak": np.clip(safe_div(peak3 - np.nan_to_num(bal), peak3 + 1.0, 0.0), 0, 1),
        "bal_min_3m_ratio": np.clip(safe_div(np.nan_to_num(roll_min(bal, 3)), avg6 + 1.0, 1.0), 0, 5),
        "net_flow_1m_usd": net,
        "net_flow_3m_usd": roll_sum(net, 3),
        "usd_share": usd_share,
        "usd_share_change_6m": np.nan_to_num(usd_share - lag(usd_share, 6)),
    }
