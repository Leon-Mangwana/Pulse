"""Phase 1 tests: no leakage, no crashes on empty windows, labels agree with the planted truth."""
import numpy as np
import pandas as pd
import pytest

from pulse.config import load_settings
from pulse.features.build import build_features, feature_names

SNAP = 12   # month index of the snapshot we tamper around (snapshot date = end of that month)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def frames(small_frames):
    return {k: small_frames[k] for k in ["customers", "accounts", "transactions"]}


@pytest.fixture(scope="module")
def base(frames, settings):
    return build_features(frames["transactions"], frames["accounts"], frames["customers"], settings)


def test_features_do_not_change_when_the_future_changes(frames, settings, base):
    """Mutate everything dated after the snapshot; features at and before it must be identical."""
    tx = frames["transactions"].copy()
    snap_date = base.loc[base["month_idx"] == SNAP, "snapshot_date"].iloc[0]
    after = tx["txn_date"] > snap_date
    assert after.sum() > 0
    rng = np.random.default_rng(0)
    tx.loc[after, "amount"] = tx.loc[after, "amount"] * 3 + 1
    tx.loc[after, "balance_after"] = 0.0
    tx.loc[after, "counterparty_type"] = rng.choice(["mobile_wallet", "other_bank", "fee"], after.sum())
    tx.loc[after, "direction"] = np.where(rng.random(after.sum()) < 0.5, "credit", "debit")
    tx = tx[~(after & (rng.random(len(tx)) < 0.5))]          # also delete half of the future rows
    extra = tx[after.reindex(tx.index, fill_value=False)].head(500).copy()
    extra["txn_id"] = "X" + extra["txn_id"]
    extra["txn_date"] = extra["txn_date"] + pd.Timedelta(days=3)
    tx = pd.concat([tx, extra], ignore_index=True)

    changed = build_features(tx, frames["accounts"], frames["customers"], settings)
    names = feature_names()
    for df in (base, changed):
        df.sort_values(["customer_id", "month_idx"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    b = base[base["month_idx"] <= SNAP]
    c = changed[changed["month_idx"] <= SNAP]
    assert len(b) == len(c) and (b["customer_id"].to_numpy() == c["customer_id"].to_numpy()).all()
    pd.testing.assert_frame_equal(b[names].reset_index(drop=True), c[names].reset_index(drop=True))
    # and the labels at that snapshot DID change, proving the mutation reached the label window
    lab_b = b.loc[b["month_idx"] == SNAP, "churn_90d"].to_numpy()
    lab_c = c.loc[c["month_idx"] == SNAP, "churn_90d"].to_numpy()
    assert not np.array_equal(np.nan_to_num(lab_b), np.nan_to_num(lab_c))


def test_customer_with_no_transactions_in_window_gets_finite_features(frames, settings):
    """Drop every transaction of one customer for the 7 months before SNAP; features must still be finite."""
    tx = frames["transactions"].copy()
    cid = tx["customer_id"].value_counts().index[0]
    snap_date = pd.Timestamp(settings["data"]["start_date"]) + pd.DateOffset(months=SNAP + 1) - pd.Timedelta(days=1)
    window = (tx["customer_id"] == cid) & (tx["txn_date"] > snap_date - pd.DateOffset(months=7)) & (tx["txn_date"] <= snap_date)
    assert window.sum() > 0
    tx = tx[~window]
    feats = build_features(tx, frames["accounts"], frames["customers"], settings, keep_inactive=True)
    row = feats[(feats["customer_id"] == cid) & (feats["month_idx"] == SNAP)]
    assert len(row) == 1
    vals = row[feature_names()].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    assert row["txn_count_3m"].iloc[0] == 0 and row["months_since_last_txn"].iloc[0] >= 6


def test_no_nan_or_inf_anywhere(base):
    vals = base[feature_names()].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    assert len(feature_names()) >= 55


def test_labels_match_planted_truth(base, small_frames):
    """A planted churner must be labelled 1 at the last snapshot before the churn date."""
    truth = small_frames["_truth"]
    churners = truth[truth["churner"]]
    lab = base[base["labelled"] == 1]
    hits, ever, total = 0, 0, 0
    for _, r in churners.iterrows():
        rows = lab[(lab["customer_id"] == r["customer_id"]) & (lab["snapshot_date"] < r["churn_date"])]
        if rows.empty:
            continue
        total += 1
        hits += int(rows.sort_values("snapshot_date").iloc[-1]["churn_90d"] == 1)
        ever += int(rows["churn_90d"].max() == 1)
    assert total > 10
    assert hits / total > 0.5, f"only {hits}/{total} planted churners labelled at the snapshot before churn"
    assert ever / total > 0.7, f"only {ever}/{total} planted churners ever labelled"
    # closures are unambiguous: every planted closure must be labelled
    closed = churners[churners["churn_type"] == "closed"]
    for _, r in closed.iterrows():
        rows = lab[(lab["customer_id"] == r["customer_id"]) & (lab["snapshot_date"] < r["churn_date"])]
        if not rows.empty:
            assert rows.sort_values("snapshot_date").iloc[-1]["churn_90d"] == 1
    # nobody who was not planted to churn is labelled as churning
    assert lab[~lab["customer_id"].isin(churners["customer_id"])]["churn_90d"].max() == 0
    assert 0.01 <= lab["churn_90d"].mean() <= 0.15


def test_protected_attributes_are_not_features(base):
    for col in ["gender", "age_band"]:
        assert col not in base.columns
