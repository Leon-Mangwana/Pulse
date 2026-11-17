"""Life-event features: remittances, school fees, supplier payments, a rising salary.

These matter for cross-sell (Phase 4) as much as for churn.
"""
from __future__ import annotations

import numpy as np

from pulse.features.common import lag, nan_reduce, roll_count, roll_sum, safe_div, stack_lags

FEATURES = {
    "remit_months_6m": "months in the last 6 with a remittance credit",
    "remit_regular": "1 if remittances arrived in at least 4 of the last 6 months",
    "remit_amount_3m_usd": "remittances received in the last 3 months, USD",
    "remit_change": "remittances of the last 3 months divided by the 3 months before (1 = unchanged)",
    "school_terms_12m": "months in the last 12 with a school-fee payment",
    "school_amount_12m_usd": "school fees paid in the last 12 months, USD",
    "school_recent": "1 if school fees were paid in the last 4 months",
    "supplier_months_6m": "months in the last 6 with a supplier payment",
    "supplier_regular": "1 if suppliers were paid in at least 4 of the last 6 months",
    "supplier_amount_3m_usd": "supplier payments in the last 3 months, USD",
    "supplier_share_3m": "supplier payments as a share of the last 3 months' outflows",
    "salary_growth_6m": "typical salary-sized credit of the last 3 months divided by that of 6 to 8 months ago",
    "salary_rising": "1 if the salary-sized credit grew by more than 10% over 6 months",
}


def compute(mt, cfg: dict) -> dict[str, np.ndarray]:
    A = mt.A
    remit_m = roll_count(A["remit_in"] > 0, 6)
    sup_m = roll_count(A["supplier_out"] > 0, 6)
    remit3 = roll_sum(A["remit_in"], 3)
    top = A["top_credit"]
    L = stack_lags(top, 9)
    L = np.where(L > 0, L, np.nan)
    recent = nan_reduce(np.nanmedian, L[..., 0:3])
    earlier = nan_reduce(np.nanmedian, L[..., 6:9])
    growth = np.clip(safe_div(recent, earlier, 1.0), 0, 3)
    growth = np.where(np.isnan(recent) | np.isnan(earlier), 1.0, growth)
    return {
        "remit_months_6m": remit_m,
        "remit_regular": (remit_m >= 4).astype(float),
        "remit_amount_3m_usd": remit3,
        "remit_change": np.clip(safe_div(remit3, lag(remit3, 3), 1.0), 0, 5),
        "school_terms_12m": roll_count(A["school_out"] > 0, 12),
        "school_amount_12m_usd": roll_sum(A["school_out"], 12),
        "school_recent": (roll_count(A["school_out"] > 0, 4) > 0).astype(float),
        "supplier_months_6m": sup_m,
        "supplier_regular": (sup_m >= 4).astype(float),
        "supplier_amount_3m_usd": roll_sum(A["supplier_out"], 3),
        "supplier_share_3m": safe_div(roll_sum(A["supplier_out"], 3), roll_sum(A["debit_usd"], 3)),
        "salary_growth_6m": growth,
        "salary_rising": (growth > 1.1).astype(float),
    }
