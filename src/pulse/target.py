"""The churn label.

For a snapshot at the end of month m, churn_90d = 1 if in the following 90 days the
customer either closes their account (hard churn) or goes quiet (silent churn).

Silent churn is an event with a month, E: the first month in which the customer's
customer-initiated transactions (fee lines excluded) AND month-end balance both fall below
`silent_threshold` (20%) of their own baseline over the `baseline_months` (6) before E,
and stay there the following month too (one quiet month alone could be a holiday). A
snapshot at month m is labelled 1 when E is in (m, m + 3], i.e. the customer goes quiet
within the next three calendar months. Snapshots at or after E are excluded by
`already_churned`: once the customer is gone there is nothing to predict. Without that
exclusion the label keeps firing for two or three months after the customer has left and
the model "predicts" churn that has already happened (found in Phase 2).

Only columns > m are read for the label and only columns <= m for the baseline, so
features (columns <= m) and labels never share a month.

Note on "90 days": hard churn uses exact days from the close date. Silent churn uses the
next three calendar months (89 to 92 days) because activity is aggregated monthly, and
needs one further month to confirm the quiet, so the last labelled snapshot is four
months before the end of the data.
"""
from __future__ import annotations

import numpy as np

from pulse.features.common import roll_mean, roll_nanmean, roll_sum

TARGET_COLUMNS = {
    "churn_90d": "1 if the customer closed or went quiet in the 90 days after the snapshot (empty when the window is not yet observable)",
    "churn_type": "hard (account closed), silent (activity and balance collapsed for two months), or none",
    "days_to_churn": "days from the snapshot to the closure or to the end of the first quiet month; empty if no churn",
    "labelled": "1 if the full window (plus the confirming month) is inside the data, so the label is known",
}


def _quiet_event_month(mt, cfg: dict) -> np.ndarray:
    """(n,) month index E of the first two consecutive quiet months, or a large number if none."""
    A, ex = mt.A, mt.exists
    n, M = mt.n, mt.M
    thr, w = float(cfg["silent_threshold"]), int(cfg["baseline_months"])
    n_txn = A["n_txn"] - A["n_fee"]
    bal = np.nan_to_num(A["bal_end"])
    base_txn = roll_mean(n_txn, w, ex)                  # over the w months up to and including m
    base_bal = np.nan_to_num(roll_nanmean(A["bal_end"], w))

    def quiet(c: int, ref: int) -> np.ndarray:          # month c measured against the baseline ending at month ref
        return (base_txn[:, ref] > 0) & (n_txn[:, c] < thr * base_txn[:, ref]) & (bal[:, c] < thr * base_bal[:, ref])

    E = np.full(n, 10 ** 6, dtype=int)
    for c in range(M - 2, 0, -1):                       # walk backwards so the earliest c wins
        hit = quiet(c, c - 1) & quiet(c + 1, c - 1) & ex[:, c - 1]
        E[hit] = c
    return E


def compute_target(mt, cfg: dict) -> dict[str, np.ndarray]:
    n, M = mt.n, mt.M
    h_days = int(cfg["horizon_days"])
    h_months = int(round(h_days / 30.4))

    E = _quiet_event_month(mt, cfg)[:, None]            # (n, 1)
    mm = np.arange(M)[None, :]                          # (1, M)

    snap = mt.month_starts[1:].to_numpy()               # (M,) snapshot instants
    close = mt.close_date[:, None]
    days_to_close = (close - snap[None, :]) / np.timedelta64(1, "D")
    hard = (days_to_close >= 0) & (days_to_close < h_days)
    silent = (E > mm) & (E <= mm + h_months) & ~hard

    # the window needs months m+1..m+3 plus the confirming month m+4 inside the data
    labelled = ((mm + h_months + 1) <= (M - 1)) & np.ones((n, 1), bool)
    churn = np.where(labelled, (hard | silent).astype(float), np.nan)
    ctype = np.where(hard, "hard", np.where(silent, "silent", "none")).astype(object)
    ctype[~labelled] = ""

    # days to churn: exact for closures; end of the first quiet month E for silent churn
    month_len = np.array([(mt.month_starts[i + 1] - mt.month_starts[i]).days for i in range(M)], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(month_len)])          # cum[k] = days from start of month 0 to start of month k
    E_clip = np.clip(E, 0, M - 1)
    days_silent = cum[E_clip + 1] - cum[np.clip(mm + 1, 0, M)]  # start of month m+1 to end of month E
    days = np.where(hard, days_to_close, np.where(silent, days_silent, np.nan))
    return {"churn_90d": churn, "churn_type": ctype, "days_to_churn": days, "labelled": labelled.astype(float)}


def already_inactive(mt, cfg: dict) -> np.ndarray:
    """(n, M) True where the customer had too few non-fee transactions in the last 3 months."""
    A = mt.A
    return roll_sum(A["n_txn"] - A["n_fee"], 3) < cfg["inactive_min_txns_3m"]


def already_churned(mt, cfg: dict) -> np.ndarray:
    """(n, M) True from the quiet event month E onwards: the customer has already gone."""
    E = _quiet_event_month(mt, cfg)[:, None]
    return np.arange(mt.M)[None, :] >= E
