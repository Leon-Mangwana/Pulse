"""Phase 5 tests: SHAP contributions add up, every card has at most three filled reasons, no raw
customer id reaches the cards, planted salary-stop churners get a salary sentence, and the
segment view adds up to the worklist."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from pulse import crosssell, explain, score
from pulse.config import PROJECT_ROOT, load_settings
from pulse.features.build import build_features
from pulse.models import churn
from pulse.models.survival import train_survival
from pulse.pseudo import pseudonymise


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def feats(small_frames, settings):
    return build_features(small_frames["transactions"], small_frames["accounts"], small_frames["customers"], settings)


@pytest.fixture(scope="module")
def products(small_frames):
    p = small_frames["products"].copy()
    for c in ("start_date", "end_date"):
        p[c] = pd.to_datetime(p[c])
    return p


@pytest.fixture(scope="module")
def models_dir(feats, products, settings, small_frames, tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    attrs = small_frames["customers"][["customer_id"] + churn.PROTECTED]
    churn.train(feats, settings, attrs=attrs, out_dir=d, log=lambda *_: None)
    train_survival(feats, products, settings, d, log=lambda *_: None)
    crosssell.train_propensity(feats, products, settings, d, log=lambda *_: None)
    return d


@pytest.fixture(scope="module")
def latest(feats, products, settings, models_dir):
    worklist, _ = score.build_worklist(feats, products, settings, models_dir)
    rows = feats[feats["snapshot_date"] == feats["snapshot_date"].max()]
    res = explain.explain_snapshot(worklist, rows, settings, models_dir)
    return worklist, rows, res


def test_contributions_add_up_to_the_raw_score(latest, models_dir):
    _, _, res = latest
    total = res["contrib"].sum(axis=1).to_numpy() + res["base"]
    raw = explain.raw_score(models_dir, res["rows"])
    assert np.allclose(total, raw, atol=1e-6)


def test_cards_have_at_most_three_filled_reasons_from_distinct_families(latest, settings):
    worklist, _, res = latest
    cards = res["cards"]
    assert len(cards) == len(worklist)
    for c in cards:
        assert 0 <= len(c["reasons"]) <= settings["explain"]["n_reasons"]
        fams = [r["family"] for r in c["reasons"]]
        assert len(fams) == len(set(fams))
        for text in [r["text"] for r in c["reasons"]] + [c["action"], c["call_script"]]:
            # a missing placeholder prints "?" after a space or bracket; a real question mark follows a letter
            assert text and not re.search(r"(?<![A-Za-z])\?", text) and "{" not in text and "}" not in text
            assert "nan" not in text.lower()
        assert all(r["contribution"] > 0 for r in c["reasons"])
        assert c["action_family"] == (c["reasons"][0]["family"] if c["reasons"] else "default")


def test_no_raw_customer_id_in_cards_or_segments(latest, small_frames, products):
    worklist, rows, res = latest
    text = json.dumps(res["cards"])
    raw_ids = set(small_frames["customers"]["customer_id"])
    assert not any(cid in text for cid in raw_ids)
    assert not re.search(r"\bC\d{6}\b", text)
    seg = explain.segment_table(worklist, res["rows"], products)
    assert not any(cid in seg.to_csv() for cid in raw_ids)


def test_planted_salary_stop_churners_get_a_salary_sentence(feats, products, settings, models_dir, small_frames):
    truth = small_frames["_truth"]
    t = truth[truth["churner"] & truth["sign_income_stop"] & truth["has_salary"] & (truth["stop_variant"] == "stop")]
    f = feats.merge(t[["customer_id", "churn_month"]], on="customer_id")
    f = f[f["month_idx"] < f["churn_month"]].sort_values("month_idx").groupby("customer_id").tail(1)
    f = f[f["months_since_salary"] >= 1]                       # the stop is visible in the features
    assert len(f) >= 3, "the small dataset should plant a few salary-stop churners"
    pcfg = settings["pseudonymise"]
    said, total = 0, 0
    for snap, grp in f.groupby("snapshot_date"):
        worklist, _ = score.build_worklist(feats, products, settings, models_dir, snapshot_date=snap)
        res = explain.explain_snapshot(worklist, feats[feats["snapshot_date"] == snap], settings, models_dir)
        refs = set(pseudonymise(grp["customer_id"], pcfg["salt"], int(pcfg["length"])))
        for c in res["cards"]:
            if c["customer_ref"] in refs:
                total += 1
                said += any("salary" in r["text"].lower() for r in c["reasons"])
    assert total == len(f)
    assert said / total >= 0.6


def test_segment_view_adds_up_and_ranks_risk(latest, products):
    worklist, _, res = latest
    seg = explain.segment_table(worklist, res["rows"], products)
    assert set(seg["dimension"]) == set(explain.SEGMENT_DIMS)
    for dim in ("branch", "segment"):
        part = seg[seg["dimension"] == dim]
        assert part["customers"].sum() == len(worklist)
        assert abs(part["value_at_risk_usd"].sum() - worklist["value_at_risk_usd"].sum()) < 1.0
        assert abs(part["share_of_var_%"].sum() - 100) < 0.5
    assert (seg["churn_rate_%"].between(0, 100)).all()


def test_a_bad_reason_file_is_rejected(settings, tmp_path):
    bad = tmp_path / "reasons.yaml"
    bad.write_text("families: {x: X}\nfeatures:\n  f: {family: y, template: t}\n", encoding="utf-8")
    cfg = {**settings, "explain": {**settings["explain"], "reasons_path": str(bad)}}
    with pytest.raises(ValueError):
        explain.load_reasons(PROJECT_ROOT, cfg)
    bad.write_text("families: {x: X}\nfeatures:\n  f: {family: x, when: [[f, '~', 1]], template: t}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        explain.load_reasons(PROJECT_ROOT, cfg)
