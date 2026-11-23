"""Phase 8: the fallback path. A customer-level table with a churn flag and no transactions goes
through ingest -> features -> stratified training -> worklist -> cards -> API."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import create_app
from pulse import explain, ingest, monitor, score, summary_mode
from pulse.config import PROJECT_ROOT, data_mode, load_settings, load_schema
from pulse.models import churn

N = 1200


def make_table(seed: int = 3) -> pd.DataFrame:
    """A plausible customer base where churn depends on tenure, activity, balance and complaints."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 120, N)
    balance = np.round(rng.gamma(2.0, 400.0, N), 2)
    products = rng.integers(1, 5, N)
    active = rng.random(N) < 0.8
    complaints = rng.poisson(0.3, N)
    logit = -0.9 - 0.012 * tenure + 1.0 * (~active) - 0.0006 * balance + 0.7 * complaints - 0.3 * (products - 1)
    churn_flag = rng.random(N) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "cust_ref": [f"K{i:05d}" for i in range(N)],
        "months_with_bank": tenure, "avg_bal": balance, "num_products": products,
        "is_active": np.where(active, "Y", "N"), "complaints": complaints.astype(float),
        "seg": rng.choice(["Retail", "Business"], N), "branch_name": rng.choice(["North", "South", "City"], N),
        "sex": rng.choice(["F", "M"], N), "age_grp": rng.choice(["18-34", "35-54", "55+"], N),
        "churned": np.where(churn_flag, "YES", "NO"),
    })
    df.loc[rng.random(N) < 0.05, "avg_bal"] = np.nan          # missing values
    df.loc[rng.random(N) < 0.02, "complaints"] = np.nan
    return df


SCHEMA = {
    "customer_summary": {
        "path": "data/raw/customer_base.csv",
        "columns": {"customer_id": "cust_ref", "churn_flag": "churned", "segment": "seg", "branch": "branch_name",
                    "gender": "sex", "age_band": "age_grp"},
        "features": {"tenure_months": "months_with_bank", "balance_usd": "avg_bal", "n_products": "num_products",
                     "active_flag": "is_active", "complaints_12m": "complaints"},
    }
}


@pytest.fixture(scope="module")
def summary_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("summary_root")
    shutil.copytree(PROJECT_ROOT / "config", root / "config")
    (root / "config" / "schema.yaml").write_text(yaml.safe_dump(SCHEMA), encoding="utf-8")
    (root / "data" / "raw").mkdir(parents=True)
    make_table().to_csv(root / "data" / "raw" / "customer_base.csv", index=False)
    return root


def test_mode_detection(summary_root):
    assert data_mode(load_schema(summary_root)) == "summary"
    assert data_mode(load_schema(PROJECT_ROOT)) == "transactions"
    with pytest.raises(ValueError):
        data_mode({**load_schema(PROJECT_ROOT), **SCHEMA})
    with pytest.raises(ValueError):
        data_mode({"customers": {}})


def test_summary_pipeline_end_to_end(summary_root):
    settings = load_settings(summary_root)
    # ingest: flag words parsed, features cast, empty transaction tables created
    con, report = ingest.load(summary_root)
    assert report.ok, report.errors
    assert ingest.stored_mode(con) == "summary"
    assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    flag = con.execute("SELECT MIN(churn_flag), MAX(churn_flag), AVG(churn_flag) FROM customer_summary").fetchone()
    assert flag[0] == 0 and flag[1] == 1 and 0.05 < flag[2] < 0.5
    active = con.execute("SELECT DISTINCT active_flag FROM customer_summary ORDER BY 1").fetchall()
    assert [a[0] for a in active] == [0.0, 1.0]
    con.close()

    # features: one row per customer, medians filled, missing flag for the 5% column only
    table, cols = summary_mode.load_table(summary_root, settings)
    feats = summary_mode.build_summary_features(table, cols, settings)
    names = summary_mode.feature_names_from(feats)
    assert len(feats) == N and feats["snapshot_date"].nunique() == 1
    assert "balance_usd_missing" in names and "complaints_12m_missing" in names and "tenure_months_missing" not in names
    assert not feats[names].isna().any().any()
    assert not {"gender", "age_band"} & set(names)

    # train: stratified split, no rules baseline, deterministic
    attrs = table[["customer_id", "gender", "age_band"]]
    model_dir = summary_root / "models"
    m1 = churn.train(feats, settings, attrs=attrs, out_dir=model_dir, log=lambda *_: None, names=names, split_mode="stratified")
    m2 = churn.train(feats, settings, attrs=attrs, out_dir=summary_root / "models_b", log=lambda *_: None, names=names, split_mode="stratified")
    assert m1["mode"] == "summary" and m1["split_mode"] == "stratified"
    assert [r["model"] for r in m1["comparison_test"]] == ["logistic", "lightgbm"]
    rates = [float(r["churn_%"]) for r in m1["split"]]
    assert max(rates) - min(rates) < 1.5, rates                      # stratified: same churn rate in every block
    assert m1["lightgbm"]["test"]["roc_auc"] > 0.6                  # the signal planted above is found
    assert m1["lightgbm"]["test"] == m2["lightgbm"]["test"]         # fixed seed
    assert {r["attribute"] for r in m1["fairness_test"]} == {"gender", "age_band"}
    ts = churn.time_split(feats, settings["model"], 90)              # a time split of one snapshot has nothing to train on
    assert ts.train.empty or ts.test.empty
    with pytest.raises(ValueError):
        churn.train(feats, settings, names=names, split_mode="random", log=lambda *_: None)

    # score: worklist, fallback urgency and recommendations, value from assumptions
    processed = summary_root / "data" / "processed"
    products = pd.DataFrame(columns=["customer_id", "product", "start_date", "end_date"])
    worklist, mapping = score.build_worklist(feats, products, settings, model_dir, mode="summary", root=summary_root)
    assert len(worklist) == N and worklist["p_churn_30d"].isna().all()
    assert set(worklist["urgency"]) <= {"act this week", "act this month", "watch"}
    assert set(worklist["rec_source"]) == {"retention", "none"} and (worklist["recommendation"] == "retention").sum() == N // 10
    assert (worklist["value_12m_usd"] > 0).all()
    assert not set(worklist["customer_ref"]) & set(table["customer_id"])
    from pulse.io import write_parquet
    write_parquet(worklist, processed / "worklist.parquet")
    write_parquet(feats, processed / "features.parquet")

    # explain: cards use the summary templates and the generic sentence; no raw feature names in text
    res = explain.run(worklist, feats, products, settings, model_dir, processed, summary_root, log=lambda *_: None, mode="summary")
    cards = res["cards"]
    with_reasons = [c for c in cards if c["reasons"]]
    assert len(with_reasons) > 0.8 * N
    texts = " ".join(r["text"] for c in cards for r in c["reasons"])
    assert "typically" in texts or "Flagged inactive" in texts
    for c in cards:
        for r in c["reasons"]:
            assert "{" not in r["text"] and "nan" not in r["text"].lower(), r["text"]
    assert set(res["segments"]["dimension"]) == {"branch", "segment"}     # no product dimension without products

    # monitor: no drift with one snapshot, run still logged
    drift = monitor.drift_report(feats, settings)
    assert drift.empty
    monitor.append_run(monitor.run_record(worklist, feats, drift), processed / monitor.RUNS_FILE)
    from pulse.io import write_parquet as wp
    wp(drift, processed / monitor.DRIFT_FILE)

    # API: everything the dashboard needs answers
    client = TestClient(create_app(summary_root))
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["customers"] == N
    wl = client.get("/worklist", params={"limit": 5}).json()
    assert wl["rows"][0]["p_churn_30d"] is None and wl["rows"][0]["value_at_risk_usd"] > 0
    ref = wl["rows"][0]["customer_ref"]
    card = client.get(f"/customers/{ref}/card").json()
    assert card["action"] and card["call_script"] and card["next_best_product"]["product"] in ("retention", "review")
    assert client.get(f"/customers/{ref}/history").status_code == 404
    d = client.get("/drivers").json()
    assert d["families"] and all("_" not in f["label"] for f in d["families"])
    m = client.get("/metrics").json()
    assert m["mode"] == "summary" and m["drift"] is None
    sim = client.post("/simulate", json={"top_n": 50}).json()
    assert sim["targeted"]["revenue_protected_usd"] >= sim["random"]["revenue_protected_usd"]
