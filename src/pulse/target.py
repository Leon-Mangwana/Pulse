"""The churn label.

For a snapshot at the end of month m, churn_90d = 1 if in the following 90 days the
customer either closes their account (hard churn) or goes quiet (silent churn): their
customer-initiated transaction count (fee lines excluded) AND their average balance in the next three months both fall below
`silent_threshold` (20%) of their own baseline over the `baseline_months` (6) before
the snapshot.

Only columns > m are read for the label and only columns <= m for the baseline, so
features (columns <= m) and labels never share a month.

Note on "90 days": hard churn uses exact days from the close date. Silent churn uses the
next three calendar months (89 to 92 days) because activity is aggregated monthly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.features.common import roll_mean, roll_nanmean, roll_sum

TARGET_COLUMNS = {
    "churn_90d": "1 if the customer closed or went silent in the 90 days after the snapshot (empty when the 90 days are not yet over)",
    "churn_type": "hard (account closed), silent (activity and balance collapsed), or none",
    "days_to_churn": "days from the snapshot to the closure or to the end of the first silent month; empty if no churn",
    "labelled": "1 if a full 90 days of data follow the snapshot, so the label is known",
}


def compute_target(mt, cfg: dict) -> dict[str, np.ndarray]:
    A, ex = mt.A, mt.exists
    n, M = mt.n, mt.M
    h_days = int(cfg["horizon_days"])
    h_months = int(round(h_days / 30.4))
    thr = float(cfg["silent_threshold"])
    w = int(cfg["baseline_months"])

    n_txn, bal = A["n_txn"] - A["n_fee"], A["bal_end"]       # customer-initiated activity: bank fee lines do not count
    base_txn = roll_mean(n_txn, w, ex)             # per month, over the last w months incl. m
    base_bal = roll_nanmean(bal, w)

    fwd_txn = np.full((n, M), np.nan)
    fwd_bal = np.full((n, M), np.nan)
    first_quiet = np.full((n, M), np.nan)        # 1..h_months: first forward month that is quiet
    for m in range(M - h_months):
        blk = slice(m + 1, m + 1 + h_months)
        fwd_txn[:, m] = n_txn[:, blk].mean(axis=1)
        fwd_bal[:, m] = np.nanmean(np.nan_to_num(bal[:, blk]), axis=1)
        quiet = (n_txn[:, blk] < thr * base_txn[:, [m]]) & (np.nan_to_num(bal[:, blk]) < thr * np.nan_to_num(base_bal[:, [m]]))
        for k in range(h_months - 1, -1, -1):
            first_quiet[quiet[:, k], m] = k + 1

    snap = mt.month_starts[1:].to_numpy()                        # (M,) snapshot instants
    close = mt.close_date[:, None]
    days_to_close = (close - snap[None, :]) / np.timedelta64(1, "D")
    hard = (days_to_close >= 0) & (days_to_close < h_days)
    silent = (fwd_txn < thr * base_txn) & (fwd_bal < thr * np.nan_to_num(base_bal)) & (base_txn > 0)
    silent = silent & ~hard

    end_of_data = mt.month_starts[-1]
    labelled = (snap + np.timedelta64(h_days, "D") <= end_of_data.to_datetime64())[None, :] & np.ones((n, 1), bool)
    churn = np.where(labelled, (hard | silent).astype(float), np.nan)
    ctype = np.where(hard, "hard", np.where(silent, "silent", "none")).astype(object)
    ctype[~labelled] = ""

    # days to churn: exact for closures; end of the first quiet month for silent churn
    month_len = np.array([(mt.month_starts[i + 1] - mt.month_starts[i]).days for i in range(M)] + [30])
    days_silent = np.full((n, M), np.nan)
    for m in range(M - h_months):
        for k in range(1, h_months + 1):
            cum = sum(month_len[m + 1: m + 1 + k])
            days_silent[(first_quiet[:, m] == k), m] = cum
    days = np.where(hard, days_to_close, np.where(silent, days_silent, np.nan))
    days = np.where(silent & np.isnan(days), float(h_days), days)
    return {"churn_90d": churn, "churn_type": ctype, "days_to_churn": days, "labelled": labelled.astype(float)}


def already_inactive(mt, cfg: dict) -> np.ndarray:
    """(n, M) True where the customer had too few non-fee transactions in the last 3 months."""
    A = mt.A
    return roll_sum(A["n_txn"] - A["n_fee"], 3) < cfg["inactive_min_txns_3m"]
