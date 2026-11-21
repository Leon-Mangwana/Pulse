"""Phase 6 tests: one per endpoint, no raw id in any response, targeted outreach beats random,
and the monitor's PSI behaves."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app, simulate, SimulateRequest
from pulse import crosssell, explain, monitor, score
from pulse.config import PROJECT_ROOT, load_settings
from pulse.features.build import build_features
from pulse.io import write_parquet
from pulse.models import churn
from pulse.models.survival import train_survival


@pytest.fixture(scope="module")
def api_root(small_frames, tmp_path_factory):
    """A project root with config/, models/ and data/processed/ built from the small synthetic set."""
    root = tmp_path_factory.mktemp("api_root")
    shutil.copytree(PROJECT_ROOT / "config", root / "config")
    settings = load_settings(root)
    feats = build_features(small_frames["transactions"], small_frames["accounts"], small_frames["customers"], settings)
    products = small_frames["products"].copy()
    for c in ("start_date", "end_date"):
        products[c] = pd.to_datetime(products[c])
    model_dir, processed = root / "models", root / "data" / "processed"
    model_dir.mkdir(), processed.mkdir(parents=True)
    attrs = small_frames["customers"][["customer_id"] + churn.PROTECTED]
    churn.train(feats, settings, attrs=attrs, out_dir=model_dir, log=lambda *_: None)
    train_survival(feats, products, settings, model_dir, log=lambda *_: None)
    crosssell.train_propensity(feats, products, settings, model_dir, log=lambda *_: None)
    worklist, _ = score.build_worklist(feats, products, settings, model_dir)
    write_parquet(worklist, processed / "worklist.parquet")
    write_parquet(feats, processed / "features.parquet")
    rows = feats[feats["snapshot_date"] == feats["snapshot_date"].max()]
    explain.run(worklist, rows, products, settings, model_dir, processed, root, log=lambda *_: None)
    drift = monitor.drift_report(feats, settings)
    write_parquet(drift, processed / monitor.DRIFT_FILE)
    monitor.append_run(monitor.run_record(worklist, feats, drift), processed / monitor.RUNS_FILE)
    return root


@pytest.fixture(scope="module")
def client(api_root):
    return TestClient(create_app(api_root))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["customers"] > 0 and body["cards"] == body["customers"]


def test_worklist_is_ranked_and_filters(client):
    r = client.get("/worklist", params={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["returned"] <= 10 and body["total"] >= body["returned"]
    ranks = [row["rank"] for row in body["rows"]]
    var = [row["value_at_risk_usd"] for row in body["rows"]]
    assert ranks == sorted(ranks) and var == sorted(var, reverse=True)
    seg = body["rows"][0]["segment"]
    filtered = client.get("/worklist", params={"segment": seg, "limit": 1000}).json()
    assert filtered["total"] < body["total"] and all(row["segment"] == seg for row in filtered["rows"])
    assert client.get("/worklist", params={"limit": 0}).status_code == 422


def test_card_by_reference_and_404(client):
    ref = client.get("/worklist", params={"limit": 1}).json()["rows"][0]["customer_ref"]
    r = client.get(f"/customers/{ref}/card")
    assert r.status_code == 200
    card = r.json()
    assert card["customer_ref"] == ref and len(card["reasons"]) <= 3 and card["action"] and card["call_script"]
    assert card["next_best_product"]["product"]
    assert client.get("/customers/C000001/card").status_code == 404


def test_segments(client):
    r = client.get("/segments")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert {row["dimension"] for row in rows} == {"branch", "segment", "product"}
    seg = client.get("/segments", params={"dimension": "segment"}).json()["rows"]
    assert all(row["dimension"] == "segment" for row in seg)
    assert abs(sum(row["share_of_var_pct"] for row in seg) - 100) < 1
    assert client.get("/segments", params={"dimension": "gender"}).status_code == 422


def test_metrics_has_models_fairness_drift_and_run(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert [x["model"] for x in m["model_comparison_test"]] == ["rules", "logistic", "lightgbm"]
    assert {row["attribute"] for row in m["fairness_test"]} == {"gender", "age_band"}
    assert m["drift"]["features"] > 0 and m["drift"]["stable"] + m["drift"]["watch"] + m["drift"]["investigate"] == m["drift"]["features"]
    assert m["runs_logged"] == 1 and m["last_run"]["customers_scored"] > 0


def test_simulate_targeted_beats_random(client):
    body = {"top_n": 50, "cost_per_contact": 2.0, "incentive_cost": 10.0, "save_rate": 0.3}
    r = client.post("/simulate", json=body)
    assert r.status_code == 200
    s = r.json()
    t, rnd = s["targeted"], s["random"]
    assert t["contacted"] == rnd["contacted"] == 50
    assert t["revenue_protected_usd"] > rnd["revenue_protected_usd"]
    assert abs(s["uplift_net_usd"] - (t["net_return_usd"] - rnd["net_return_usd"])) < 0.02
    assert abs(t["total_cost_usd"] - (t["contact_cost_usd"] + t["incentive_cost_usd"])) < 0.02
    assert client.post("/simulate", json={"top_n": -1}).status_code == 422
    assert client.post("/simulate", json={"save_rate": 2}).status_code == 422


def test_simulate_arithmetic_on_a_toy_worklist():
    wl = pd.DataFrame({"snapshot_date": pd.Timestamp("2026-01-31"), "rank": [1, 2, 3, 4],
                       "p_churn_90d": [0.5, 0.5, 0.0, 0.0], "value_12m_usd": [100.0, 100.0, 100.0, 100.0]})
    s = simulate(wl, SimulateRequest(top_n=2, cost_per_contact=1.0, incentive_cost=10.0, save_rate=0.5))
    # top 2: saves = (0.5 + 0.5) x 0.5 = 0.5; protected = 0.5 x 100 = 50; cost = 2 x 1 + 0.5 x 10 = 7
    assert s.targeted.expected_saves == 0.5 and s.targeted.revenue_protected_usd == 50 and s.targeted.net_return_usd == 43
    # random 2 of 4: mean p 0.25 -> saves 0.25, protected 25, cost 2 + 2.5
    assert s.random.expected_saves == 0.25 and s.random.revenue_protected_usd == 25 and s.random.net_return_usd == 20.5


def test_no_raw_customer_id_in_any_response(client, small_frames):
    raw_ids = set(small_frames["customers"]["customer_id"])
    ref = client.get("/worklist", params={"limit": 1}).json()["rows"][0]["customer_ref"]
    texts = [client.get("/worklist", params={"limit": 5000}).text, client.get(f"/customers/{ref}/card").text,
             client.get("/segments").text, client.get("/metrics").text,
             client.post("/simulate", json={"top_n": 10}).text]
    for text in texts:
        assert not any(cid in text for cid in raw_ids)


def test_psi_zero_for_same_data_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert monitor.psi(a, a) < 1e-9
    assert monitor.psi(a, rng.normal(size=5000)) < 0.05
    assert monitor.psi(a, a + 1.0) > 0.25
    assert monitor.psi(np.ones(100), np.ones(100) * 3) == 0.0            # constant in training: not measurable
    assert monitor.psi_status(0.05, 0.1, 0.25) == "stable" and monitor.psi_status(0.3, 0.1, 0.25) == "investigate"


# ----------------------------------------------------------------------------- Phase 7 additions
def test_history_twelve_months_and_404(client):
    ref = client.get("/worklist", params={"limit": 1}).json()["rows"][0]["customer_ref"]
    r = client.get(f"/customers/{ref}/history")
    assert r.status_code == 200
    h = r.json()
    assert h["customer_ref"] == ref and 1 <= h["months"] <= 12 and len(h["points"]) == h["months"]
    months = [p["month"] for p in h["points"]]
    assert months == sorted(months)
    assert all(p["salary_credit_usd"] >= 0 and p["money_in_usd"] >= 0 for p in h["points"])
    assert client.get("/customers/C000001/history").status_code == 404


def test_drivers_by_family_in_plain_english(client):
    r = client.get("/drivers")
    assert r.status_code == 200
    d = r.json()
    assert d["families"] and abs(sum(f["share_pct"] for f in d["families"]) - 100) < 1.5
    shares = [f["share_pct"] for f in d["families"]]
    assert shares == sorted(shares, reverse=True)
    assert all(" " in f["label"] or f["label"].istitle() for f in d["families"])      # labels, not feature names
    assert all("_" not in f["label"] for f in d["families"])
    assert all("_" not in s["reading"].split(" ")[0] for s in d["top_signals"])


def test_simulate_extremes_zero_and_everyone(client):
    total = client.get("/health").json()["customers"]
    zero = client.post("/simulate", json={"top_n": 0}).json()
    for side in ("targeted", "random"):
        assert zero[side]["contacted"] == 0 and zero[side]["net_return_usd"] == 0 and zero[side]["return_per_usd"] is None
    everyone = client.post("/simulate", json={"top_n": total * 3}).json()
    t, rnd = everyone["targeted"], everyone["random"]
    assert t["contacted"] == rnd["contacted"] == total
    assert abs(t["revenue_protected_usd"] - rnd["revenue_protected_usd"]) < 0.05 * max(1.0, t["revenue_protected_usd"])
    assert abs(everyone["uplift_net_usd"]) < 0.05 * max(1.0, abs(t["net_return_usd"]))
