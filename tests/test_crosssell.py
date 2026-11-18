"""Phase 4 tests: rules fire from YAML alone, held products are skipped, the guardrail wins,
every customer gets an action, and triggered customers take up the product more often."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from pulse import crosssell
from pulse.config import PROJECT_ROOT, load_settings
from pulse.features.build import build_features, feature_names
from pulse.models import churn
from pulse.models.survival import train_survival


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def rules(settings):
    return crosssell.load_rules(PROJECT_ROOT, settings)


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


def _toy_rows():
    """Two customers at one snapshot: a remittance receiver and a plain one; every feature present."""
    rows = pd.DataFrame(0.0, index=[0, 1], columns=feature_names())
    rows.insert(0, "customer_id", ["C1", "C2"])
    rows.insert(1, "snapshot_date", pd.Timestamp("2026-01-31"))
    rows.loc[0, ["remit_regular", "remit_months_6m", "remit_amount_3m_usd"]] = [1, 5, 600.0]
    return rows


def test_trigger_fires_fills_the_template_and_skips_held_products(rules):
    rows = _toy_rows()
    no_products = pd.DataFrame(columns=["customer_id", "product", "start_date", "end_date"])
    held = crosssell.products_held(rows, no_products, ["fx_savings"])
    fired = crosssell.fire_triggers(rows, rules, held)
    assert list(fired["row"]) == [0] and fired["product"].iloc[0] == "fx_savings"
    assert "5 of the last 6 months" in fired["reason"].iloc[0] and "600 USD" in fired["reason"].iloc[0]
    # the same customer already holding fx_savings is not offered it
    holding = pd.DataFrame({"customer_id": ["C1"], "product": ["fx_savings"],
                            "start_date": [pd.Timestamp("2025-06-01")], "end_date": [pd.NaT]})
    held = crosssell.products_held(rows, holding, ["fx_savings"])
    assert held.loc[0, "fx_savings"] and crosssell.fire_triggers(rows, rules, held).empty


def test_a_new_trigger_needs_only_yaml(rules):
    extra = copy.deepcopy(rules)
    extra["triggers"].append({"name": "cash_heavy", "product": "mobile_banking", "priority": 9,
                              "conditions": [["cash_share_3m", ">", 0.5]],
                              "reason": "{cash_share_3m:.0%} of outflows are cash withdrawals."})
    rows = _toy_rows()
    rows.loc[1, "cash_share_3m"] = 0.8
    held = pd.DataFrame(False, index=rows.index, columns=["fx_savings", "mobile_banking"])
    fired = crosssell.fire_triggers(rows, extra, held)
    assert set(fired["trigger"]) == {"diaspora_remittances", "cash_heavy"}
    assert fired.loc[fired["trigger"] == "cash_heavy", "reason"].iloc[0] == "80% of outflows are cash withdrawals."


def test_unknown_operator_is_rejected_when_loading(settings, tmp_path):
    bad = tmp_path / "crosssell.yaml"
    bad.write_text("triggers:\n  - {name: x, product: fx_savings, conditions: [[remit_regular, '~', 1]], reason: r}\n", encoding="utf-8")
    cfg = copy.deepcopy(settings)
    cfg["crosssell"]["rules_path"] = str(bad)
    with pytest.raises(ValueError):
        crosssell.load_rules(PROJECT_ROOT, cfg)


def test_every_customer_gets_an_action_and_the_guardrail_wins(feats, products, settings, models_dir, rules):
    latest = feats[feats["snapshot_date"] == feats["snapshot_date"].max()].reset_index(drop=True)
    var = np.linspace(0, 100, len(latest))                    # fake value at risk, highest last
    rec = crosssell.recommend(latest, products, settings, models_dir, value_at_risk=var, rules=rules)
    assert len(rec) == len(latest)
    assert (rec["recommendation"] != "none").all() and (rec["rec_reason"] != "").all()
    assert set(rec["rec_source"]) <= set(crosssell.SOURCES) - {"none"}
    k = int(np.floor(settings["crosssell"]["retention_top_share"] * len(latest)))
    top = np.argsort(-var)[:k]
    assert (rec.loc[top, "rec_source"] == "retention").all() and (rec.loc[top, "recommendation"] == "retention").all()
    assert (rec.drop(index=top)["rec_source"] != "retention").all()
    # a recommended product is never one the customer already holds
    prods = crosssell.product_list(products, settings)
    held = crosssell.products_held(latest, products, prods)
    offers = rec[rec["rec_source"] != "retention"]
    assert not any(held.at[i, p] for i, p in zip(offers.index, offers["recommendation"]) if p in held.columns)
    status = json.loads((models_dir / "crosssell" / "propensity.json").read_text())
    assert set(status) == set(prods) and all(e["status"] in ("model", "rule_only") for e in status.values())


def test_triggered_customers_take_up_more(feats, products, settings, rules):
    split = churn.time_split(feats, settings["model"], 90)
    latest = feats[feats["snapshot_date"] == feats["snapshot_date"].max()].reset_index(drop=True)
    rep = crosssell.trigger_report(latest, split.test, products, rules, settings)
    assert len(rep) == len(rules["triggers"]) and (rep["fires_latest"] > 0).any()
    pooled = rep[rep["test_triggered"] > 0]
    assert (pooled["uptake_triggered_%"].mean() > pooled["uptake_untriggered_%"].mean())
