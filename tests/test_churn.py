"""Phase 2 tests: the split is time-ordered with a gap, protected attributes never enter the
model, training is deterministic, and the saved model reloads and scores."""
import copy

import numpy as np
import pandas as pd
import pytest

from pulse.config import load_settings
from pulse.features.build import build_features, feature_names
from pulse.models import churn


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def feats(small_frames, settings):
    return build_features(small_frames["transactions"], small_frames["accounts"], small_frames["customers"], settings)


@pytest.fixture(scope="module")
def attrs(small_frames):
    return small_frames["customers"][["customer_id"] + churn.PROTECTED]


def test_split_is_time_ordered_with_a_gap(feats, settings):
    horizon = int(settings["target"]["horizon_days"])
    split = churn.time_split(feats, settings["model"], horizon)
    for _, df in split.blocks():
        assert len(df) > 0 and (df["labelled"] == 1).all()
    assert split.train["snapshot_date"].max() < split.valid["snapshot_date"].min()
    # every validation label window (3 calendar months) ends before the first test snapshot
    assert split.valid["snapshot_date"].max() + pd.DateOffset(months=3) <= split.test["snapshot_date"].min()
    keys = [set(zip(df["customer_id"], df["snapshot_date"])) for _, df in split.blocks()]
    assert not (keys[0] & keys[1]) and not (keys[1] & keys[2]) and not (keys[0] & keys[2])


def test_split_rejects_overlap_and_random_order(feats, settings):
    cfg = copy.deepcopy(settings["model"])
    cfg["split"]["test_start"] = "2026-02-28"          # one month after valid_end: label windows overlap
    with pytest.raises(ValueError):
        churn.time_split(feats, cfg, 90)
    cfg = copy.deepcopy(settings["model"])
    cfg["split"]["train_end"] = cfg["split"]["valid_end"]
    with pytest.raises(ValueError):
        churn.time_split(feats, cfg, 90)


def test_protected_attributes_are_never_inputs(feats):
    names = feature_names()
    assert not set(churn.PROTECTED) & set(names)
    assert not set(churn.PROTECTED) & set(feats.columns)


def test_capture_at_top_k():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    score = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.4, 0.5, 0.6, 0.7, 0.05])
    cap, lift, thr = churn.top_k_capture(y, score, 0.2)   # top 2 rows: scores 0.9 and 0.8, both churners
    assert cap == pytest.approx(2 / 3) and lift == pytest.approx(10 / 3) and thr == 0.8


def test_train_is_deterministic_and_reloads(feats, settings, attrs, tmp_path):
    m1 = churn.train(feats, settings, attrs=attrs, out_dir=tmp_path / "a", log=lambda *_: None)
    m2 = churn.train(feats, settings, attrs=attrs, out_dir=tmp_path / "b", log=lambda *_: None)
    assert m1["comparison_test"] == m2["comparison_test"]
    assert m1["lightgbm"]["test"] == m2["lightgbm"]["test"]
    for f in ["churn_lgbm.txt", "calibrator.joblib", "feature_list.json", "metrics.json", "train_report.md"]:
        assert (tmp_path / "a" / f).exists()
    test_rows = churn.time_split(feats, settings["model"], 90).test
    pa = churn.predict(tmp_path / "a", test_rows)
    pb = churn.predict(tmp_path / "b", test_rows)
    assert np.array_equal(pa, pb) and len(pa) == len(test_rows)
    assert (pa >= 0).all() and (pa <= 1).all()
    # the fairness table used the protected columns only after scoring, for the report
    assert {r["attribute"] for r in m1["fairness_test"]} == set(churn.PROTECTED)
    # the model ranks churners well above random even on the small sample
    lgb_row = next(r for r in m1["comparison_test"] if r["model"] == "lightgbm")
    assert lgb_row["lift@10%"] > 1.5
