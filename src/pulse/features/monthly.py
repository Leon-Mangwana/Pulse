"""Collapse raw transactions into one row per customer per calendar month.

This is the only place that touches individual transactions. Everything downstream
(feature families, the churn target) works on (customers x months) arrays, which
makes "use only months before the snapshot" a matter of column indexing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DIGITAL_CHANNELS = {"mobile app", "USSD", "ZIPIT"}


@dataclass
class MonthlyTable:
    ids: np.ndarray                    # (n,) customer ids
    month_starts: pd.DatetimeIndex     # (M+1,) first day of each month, plus the day after the last
    open_m: np.ndarray                 # (n,) month index of the customer's open date (negative = before window)
    close_date: np.ndarray             # (n,) datetime64 of the earliest account closure, NaT if none
    exists: np.ndarray                 # (n, M) True for months the customer was open
    A: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (n, M) float array
    keys: np.ndarray | None = None     # (n, M) payer key of the largest credit that month

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def M(self) -> int:
        return len(self.month_starts) - 1

    def snapshot_date(self, m: int) -> pd.Timestamp:
        """A snapshot 'at the end of month m' is the first instant of month m+1."""
        return self.month_starts[m + 1]


def month_index(dates, start: pd.Timestamp) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return ((d.year - start.year) * 12 + (d.month - start.month)).to_numpy()


def payer_key(desc: pd.Series) -> pd.Series:
    """Strip digits and spaces so 'SALARY ECONET 123' and 'SALARY ECONET 456' match."""
    return desc.fillna("").str.upper().str.replace(r"[\d\s\-/.]+", "", regex=True)


def build_monthly(tx: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame,
                  settings: dict) -> MonthlyTable:
    start = pd.Timestamp(settings["data"]["start_date"])
    end = pd.Timestamp(settings["data"]["end_date"])
    rate = float(settings["fx"]["zig_per_usd"])
    M = (end.year - start.year) * 12 + end.month - start.month + 1
    month_starts = pd.date_range(start, periods=M + 1, freq="MS")

    cust = customers.reset_index(drop=True)
    ids = cust["customer_id"].to_numpy()
    cidx = pd.Series(np.arange(len(ids)), index=ids)
    n = len(ids)

    tx = tx[(tx["txn_date"] >= start) & (tx["txn_date"] <= end)].copy()
    tx["ci"] = cidx.reindex(tx["customer_id"]).to_numpy()
    tx = tx[tx["ci"].notna()]
    tx["ci"] = tx["ci"].astype(int)
    tx["m"] = month_index(tx["txn_date"], start)
    tx["usd"] = np.where(tx["currency"] == "ZiG", tx["amount"] / rate, tx["amount"])
    credit = (tx["direction"] == "credit").to_numpy()
    cp = tx["counterparty_type"].fillna("").to_numpy()
    is_self = cp == "self"
    is_fee = cp == "fee"
    usd = tx["usd"].to_numpy()

    cols = {
        "n_txn": np.ones(len(tx)),
        "n_credit": credit.astype(float),
        "n_debit": (~credit).astype(float),
        "n_fee": is_fee.astype(float),
        "n_reversal": tx["description"].fillna("").str.upper().str.startswith("REVERSAL").to_numpy().astype(float),
        "n_digital": tx["channel"].isin(DIGITAL_CHANNELS).to_numpy().astype(float),
        "n_branch": (tx["channel"] == "branch").to_numpy().astype(float),
        "n_atm": (tx["channel"] == "ATM").to_numpy().astype(float),
        "n_pos": (tx["channel"] == "POS").to_numpy().astype(float),
        "credit_usd": np.where(credit & ~is_self, usd, 0.0),      # money in, excluding own-account transfers
        "debit_usd": np.where(~credit & ~is_self & ~is_fee, usd, 0.0),  # money out, excluding transfers and fees
        "fee_out": np.where(is_fee, usd, 0.0),
        "wallet_out": np.where(~credit & (cp == "mobile_wallet"), usd, 0.0),
        "obank_out": np.where(~credit & (cp == "other_bank"), usd, 0.0),
        "school_out": np.where(~credit & (cp == "school"), usd, 0.0),
        "supplier_out": np.where(~credit & (cp == "supplier"), usd, 0.0),
        "cash_out": np.where(~credit & (cp == "cash"), usd, 0.0),
        "remit_in": np.where(credit & (cp == "remittance"), usd, 0.0),
        "self_in": np.where(credit & is_self, usd, 0.0),
        "self_out": np.where(~credit & is_self, usd, 0.0),
    }
    agg = pd.DataFrame(cols).groupby([tx["ci"].to_numpy(), tx["m"].to_numpy()]).sum()
    A: dict[str, np.ndarray] = {}
    ci_a = agg.index.get_level_values(0).to_numpy()
    m_a = agg.index.get_level_values(1).to_numpy()
    for name in cols:
        arr = np.zeros((n, M))
        arr[ci_a, m_a] = agg[name].to_numpy()
        A[name] = arr
    days = tx.groupby(["ci", "m"])["txn_date"].nunique()
    arr = np.zeros((n, M))
    arr[days.index.get_level_values(0), days.index.get_level_values(1)] = days.to_numpy()
    A["active_days"] = arr

    # largest single credit in the month that is not an own-account transfer: the salary candidate
    cr = tx[credit & ~is_self].sort_values("usd", ascending=False).drop_duplicates(["ci", "m"])
    top = np.zeros((n, M)); top_day = np.zeros((n, M)); keys = np.full((n, M), "", dtype=object)
    top[cr["ci"], cr["m"]] = cr["usd"].to_numpy()
    top_day[cr["ci"], cr["m"]] = cr["txn_date"].dt.day.to_numpy()
    keys[cr["ci"], cr["m"]] = payer_key(cr["description"]).to_numpy()
    A["top_credit"], A["top_credit_day"] = top, top_day

    # month-end balance per account, carried forward over quiet months, converted to USD
    acc = accounts.reset_index(drop=True)
    aidx = pd.Series(np.arange(len(acc)), index=acc["account_id"])
    tx_sorted = tx.sort_values(["txn_date", "txn_id"])
    last = tx_sorted.groupby([tx_sorted["account_id"].map(aidx), tx_sorted["m"]])["balance_after"].last()
    grid = np.full((len(acc), M), np.nan)
    grid[last.index.get_level_values(0).astype(int), last.index.get_level_values(1)] = last.to_numpy()
    grid = pd.DataFrame(grid).ffill(axis=1).to_numpy()
    closed_m = month_index(acc["close_date"].fillna(pd.Timestamp("2100-01-01")), start)
    grid[np.arange(M)[None, :] > closed_m[:, None]] = 0.0        # closed accounts hold nothing
    is_zig = (acc["currency"] == "ZiG").to_numpy()[:, None]
    grid_usd = np.where(is_zig, grid / rate, grid)
    owner = cidx.reindex(acc["customer_id"]).to_numpy().astype(int)
    for name, part in [("bal_end", grid_usd), ("bal_usd_end", np.where(is_zig, np.nan, grid_usd)),
                       ("bal_zig_end", np.where(is_zig, grid_usd, np.nan))]:
        df = pd.DataFrame(part).groupby(owner).sum(min_count=1)
        arr = np.full((n, M), np.nan)
        arr[df.index.to_numpy()] = df.to_numpy()
        A[name] = arr

    open_m = month_index(cust["open_date"], start)
    close = acc.groupby("customer_id")["close_date"].min()
    close_date = pd.to_datetime(close.reindex(ids)).to_numpy()
    close_m = month_index(pd.Series(close_date).fillna(pd.Timestamp("2100-01-01")), start)
    mm = np.arange(M)[None, :]
    exists = (mm >= open_m[:, None]) & (mm <= close_m[:, None])
    return MonthlyTable(ids=ids, month_starts=month_starts, open_m=open_m, close_date=close_date,
                        exists=exists, A=A, keys=keys)
