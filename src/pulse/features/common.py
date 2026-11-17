"""Rolling-window helpers on (customers x months) arrays.

Every feature family works on 2-D arrays with one row per customer and one column
per calendar month. Column m holds what happened IN month m. A snapshot taken at the
end of month m may use columns <= m only; these helpers never look right of m.
"""
from __future__ import annotations

import warnings

import numpy as np


def lag(X: np.ndarray, k: int, fill=np.nan) -> np.ndarray:
    """Value k months earlier (column m-k). The first k columns get `fill`."""
    if k == 0:
        return X.copy()
    out = np.full_like(X, fill, dtype=float)
    out[:, k:] = X[:, :-k]
    return out


def stack_lags(X: np.ndarray, w: int) -> np.ndarray:
    """(n, M, w) array: [..., 0] is this month, [..., k] is k months earlier."""
    return np.stack([lag(X, k) for k in range(w)], axis=-1)


def roll_sum(X: np.ndarray, w: int) -> np.ndarray:
    """Sum over the last w months ending at m (inclusive). Missing early months count as 0."""
    n = X.shape[0]
    Xp = np.concatenate([np.zeros((n, w)), np.nan_to_num(X)], axis=1)
    C = np.cumsum(Xp, axis=1)
    return C[:, w:] - C[:, :-w]


def roll_mean(X: np.ndarray, w: int, exists: np.ndarray) -> np.ndarray:
    """Mean over the last w months, counting only months the customer existed. NaN if none."""
    num = roll_sum(np.where(exists, np.nan_to_num(X), 0.0), w)
    den = roll_sum(exists.astype(float), w)
    return safe_div(num, den, np.nan)


def roll_nanmean(X: np.ndarray, w: int) -> np.ndarray:
    """Mean over the last w months ignoring NaN cells. NaN if all are NaN."""
    valid = ~np.isnan(X)
    num = roll_sum(np.where(valid, X, 0.0), w)
    den = roll_sum(valid.astype(float), w)
    return safe_div(num, den, np.nan)


def roll_count(mask: np.ndarray, w: int) -> np.ndarray:
    return roll_sum(mask.astype(float), w)


def nan_reduce(fn, X: np.ndarray, axis=-1) -> np.ndarray:
    """nanmin/nanmax/nanmedian without the all-NaN warning (an all-NaN slice is a valid 'no data')."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(X, axis=axis)


def roll_min(X: np.ndarray, w: int) -> np.ndarray:
    return nan_reduce(np.nanmin, stack_lags(X, w))


def roll_max(X: np.ndarray, w: int) -> np.ndarray:
    return nan_reduce(np.nanmax, stack_lags(X, w))


def roll_slope(X: np.ndarray, w: int) -> np.ndarray:
    """Least-squares slope per month over the last w months (NaN cells ignored)."""
    L = stack_lags(X, w)                       # [..., k] = k months ago
    t = -np.arange(w, dtype=float)             # time axis: 0 now, -1 last month, ...
    valid = ~np.isnan(L)
    x = np.where(valid, L, 0.0)
    cnt = valid.sum(-1)
    st = (valid * t).sum(-1)
    sx = x.sum(-1)
    stt = (valid * t * t).sum(-1)
    stx = (x * t).sum(-1)
    den = cnt * stt - st * st
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (cnt * stx - st * sx) / den
    slope[(cnt < 2) | (den == 0)] = np.nan
    return slope


def safe_div(a, b, fill=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a / b
    bad = ~np.isfinite(out)
    out = np.where(bad, fill, out)
    return out


def months_since(hit: np.ndarray, cap: int) -> np.ndarray:
    """Months since the most recent True in `hit` (0 = this month). `cap` if none within cap months."""
    out = np.full(hit.shape, float(cap))
    L = stack_lags(hit.astype(float), cap)      # [..., k] = hit k months ago (NaN before start)
    L = np.nan_to_num(L) > 0
    for k in range(cap - 1, -1, -1):
        out = np.where(L[..., k], k, out)
    return out
