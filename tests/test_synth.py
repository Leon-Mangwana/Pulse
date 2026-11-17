"""The synthetic data must be deterministic, internally consistent, and carry the planted signals."""
import numpy as np
import pandas as pd

from pulse.config import load_settings
from pulse import synth


def test_same_seed_same_output(small_frames):
    again = synth.generate(load_settings(), n_customers=250, seed=7)
    for name, df in small_frames.items():
        pd.testing.assert_frame_equal(df, again[name])


def test_different_seed_different_output(small_frames):
    other = synth.generate(load_settings(), n_customers=250, seed=8)
    assert not small_frames["transactions"]["amount"].equals(other["transactions"]["amount"])


def test_shapes_and_keys(small_frames):
    cust, acct, tx, prods = (small_frames[k] for k in ["customers", "accounts", "transactions", "products"])
    assert len(cust) == 250
    assert cust["customer_id"].is_unique and acct["account_id"].is_unique and tx["txn_id"].is_unique
    assert acct["customer_id"].isin(cust["customer_id"]).all()
    assert tx["account_id"].isin(acct["account_id"]).all()
    assert prods["customer_id"].isin(cust["customer_id"]).all()
    assert set(tx["direction"]) == {"credit", "debit"}
    assert (tx["amount"] > 0).all()


def test_running_balance_is_consistent(small_frames):
    tx = small_frames["transactions"]
    signed = np.where(tx["direction"] == "credit", tx["amount"], -tx["amount"])
    for _, g in tx.assign(signed=signed).groupby("account_id"):
        diffs = g["balance_after"].diff().dropna().to_numpy()
        assert np.allclose(diffs, g["signed"].to_numpy()[1:], atol=0.02)
    assert (tx["balance_after"] >= -0.01).all()


def test_dates_respect_account_life(small_frames):
    tx, acct = small_frames["transactions"], small_frames["accounts"]
    settings = load_settings()
    assert tx["txn_date"].min() >= pd.Timestamp(settings["data"]["start_date"])
    assert tx["txn_date"].max() <= pd.Timestamp(settings["data"]["end_date"])
    m = tx.merge(acct[["account_id", "open_date", "close_date"]], on="account_id")
    assert (m["txn_date"] >= m["open_date"]).all()
    closed = m[m["close_date"].notna()]
    assert (closed["txn_date"] <= closed["close_date"]).all()


def test_planted_churn_is_noisy_not_a_rule(small_frames):
    t = small_frames["_truth"]
    assert 0.05 <= t["churner"].mean() <= 0.16
    churners = t[t["churner"]]
    assert (churners["churn_type"] == "closed").mean() < 0.5          # most churn silently
    assert churners["no_warning"].any()                               # some churn with no warning
    signs = ["sign_income_stop", "sign_balance_drain", "sign_leakage_rise", "sign_activity_drop"]
    assert t.loc[~t["churner"], signs].any(axis=1).any()              # some non-churners show signs too
    assert t.loc[~t["churner"], signs].any(axis=1).mean() < 0.2


def test_silent_churners_go_quiet(small_frames):
    """After the churn date a silent churner has far fewer transactions than before."""
    t, tx = small_frames["_truth"], small_frames["transactions"]
    s = t[(t["churn_type"] == "silent") & (t["churn_month"] <= 18)]
    m = tx.merge(s[["customer_id", "churn_date"]], on="customer_id")
    before = m[(m["txn_date"] < m["churn_date"]) & (m["txn_date"] >= m["churn_date"] - pd.Timedelta(days=180))]
    after = m[m["txn_date"] > m["churn_date"]]
    end = pd.Timestamp(load_settings()["data"]["end_date"])
    months_after = ((end - s["churn_date"]).dt.days / 30.4).sum()
    per_month_before = len(before) / len(s) / 6
    per_month_after = len(after) / months_after
    assert per_month_after < 0.4 * per_month_before


def test_life_events_raise_uptake(small_frames):
    t = small_frames["_truth"]
    for event, (product, _, _) in synth.UPTAKE.items():
        rate = t.groupby(event)["uptake_product"].apply(lambda s: (s == product).mean())
        if True in rate.index and False in rate.index and (t[event].sum() > 15):
            assert rate[True] > rate[False]


def test_truth_columns_are_not_in_pipeline_tables(small_frames):
    for name in ["customers", "accounts", "transactions", "products"]:
        assert not any(c.startswith("sign_") or c.startswith("churn") for c in small_frames[name].columns)
