"""Phase 3 tests: value, pseudonymisation, survival probabilities and the worklist."""
import numpy as np
import pandas as pd
import pytest

from pulse import crosssell, score
from pulse.config import load_settings
from pulse.features.build import build_features
from pulse.models import churn
from pulse.models.survival import BANDS, CoxChurn, train_survival, urgency_band
from pulse.pseudo import id_map, pseudonymise
from pulse.value import call_list_comparison, compute_value


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
    """Churn model + survival model trained on the small sample, saved to a temp dir."""
    d = tmp_path_factory.mktemp("models")
    attrs = small_frames["customers"][["customer_id"] + churn.PROTECTED]
    churn.train(feats, settings, attrs=attrs, out_dir=d, log=lambda *_: None)
    train_survival(feats, products, settings, d, log=lambda *_: None)
    crosssell.train_propensity(feats, products, settings, d, log=lambda *_: None)   # the worklist needs it since Phase 4
    return d


def _rows(balance, fees=10.0, usd_share=1.0, snapshot="2026-01-31"):
    return pd.DataFrame({"customer_id": ["C1"], "snapshot_date": [pd.Timestamp(snapshot)],
                         "fees_3m_usd": [fees], "usd_share": [usd_share], "bal_avg_6m_usd": [balance]})


def test_value_components_and_floor(settings):
    cfg = settings["value"]
    products = pd.DataFrame({"customer_id": ["C1", "C1", "C1"], "product": ["personal_loan", "savings_account", "fx_savings"],
                             "start_date": pd.to_datetime(["2025-01-01", "2025-06-01", "2026-03-01"]),
                             "end_date": pd.to_datetime([None, "2025-12-31", None])})
    v = compute_value(_rows(1000.0, fees=12.0, usd_share=1.0), products, cfg).iloc[0]
    assert v["fee_income_12m_usd"] == pytest.approx(12.0 * cfg["fee_annualisation"])
    assert v["interest_margin_12m_usd"] == pytest.approx(1000.0 * cfg["interest_margin_usd"])
    # loan is held; savings ended before the snapshot; fx_savings starts after it
    assert v["product_margin_12m_usd"] == pytest.approx(cfg["product_margin_usd"]["personal_loan"])
    assert v["value_12m_usd"] == pytest.approx(v["fee_income_12m_usd"] + v["interest_margin_12m_usd"] + v["product_margin_12m_usd"])
    # ZiG balances earn the ZiG margin; value never drops below the floor
    z = compute_value(_rows(1000.0, fees=0.0, usd_share=0.0), products.iloc[:0], cfg).iloc[0]
    assert z["interest_margin_12m_usd"] == pytest.approx(1000.0 * cfg["interest_margin_zig"])
    empty = compute_value(_rows(0.0, fees=0.0), products.iloc[:0], cfg).iloc[0]
    assert empty["value_12m_usd"] == cfg["min_value_usd"]
    low, high = (compute_value(_rows(b), products.iloc[:0], cfg)["value_12m_usd"].iloc[0] for b in (100.0, 5000.0))
    assert high > low


def test_pseudonymise_is_stable_and_hides_the_id(settings):
    cfg = settings["pseudonymise"]
    ids = ["C000001", "C000002", "C000001"]
    refs = pseudonymise(ids, cfg["salt"], cfg["length"])
    assert refs[0] == refs[2] and refs[0] != refs[1]
    assert all("000001" not in r for r in refs) and all(len(r) == cfg["length"] + 1 for r in refs)
    assert (pseudonymise(ids, "another salt", cfg["length"]) != refs).all()
    m = id_map(ids, cfg["salt"], cfg["length"])
    assert len(m) == 2 and set(m.columns) == {"customer_ref", "customer_id"}


def test_value_at_risk_ordering_reaches_more_dollars_than_random():
    rng = np.random.default_rng(1)
    n = 2000
    value = rng.lognormal(5, 1, n)
    p = rng.random(n) * 0.2
    y = (rng.random(n) < p).astype(int)          # churn really does follow p
    t = call_list_comparison(y, p, value, [0.1], seed=42).set_index("ordering")
    assert t.loc["by value at risk", "churned_value_reached_usd"] > t.loc["random", "churned_value_reached_usd"]
    assert t.loc["by value at risk", "churned_value_reached_usd"] >= t.loc["by churn probability", "churned_value_reached_usd"]


def test_survival_probabilities_are_ordered_and_bands_cover_everyone(feats, settings):
    split = churn.time_split(feats, settings["model"], 90)
    cox = CoxChurn(settings["survival"], 90).fit(pd.concat([split.train, split.valid]))
    probs = cox.predict_probs(split.test)
    assert ((probs["p_churn_30d"] <= probs["p_churn_60d"]) & (probs["p_churn_60d"] <= probs["p_churn_90d"])).all()
    assert (probs >= 0).all().all() and (probs <= 1).all().all()
    assert cox.concordance(split.test) > 0.5
    assert set(urgency_band(probs, settings["survival"])) <= set(BANDS)


def test_worklist_one_row_per_customer_without_raw_ids(feats, products, settings, models_dir):
    wl, mapping = score.build_worklist(feats, products, settings, models_dir)
    assert wl["customer_ref"].is_unique and len(wl) == feats.loc[feats["snapshot_date"] == feats["snapshot_date"].max(), "customer_id"].nunique()
    assert "customer_id" not in wl.columns
    assert (wl["rank"].to_numpy() == np.arange(1, len(wl) + 1)).all()
    assert (np.diff(wl["value_at_risk_usd"].to_numpy()) <= 1e-9).all()
    assert np.allclose(wl["value_at_risk_usd"], wl["p_churn_90d"] * wl["value_12m_usd"])
    assert set(wl["urgency"]) <= set(BANDS)
    assert set(mapping["customer_ref"]) == set(wl["customer_ref"])
    assert (models_dir / "survival.joblib").exists() and (models_dir / "value_report.md").exists()
