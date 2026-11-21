"""Phase 7 tests for the dashboard arithmetic (no Streamlit import, so these run anywhere)."""
import numpy as np
import pandas as pd

from dashboard import logic


def toy_worklist() -> pd.DataFrame:
    return pd.DataFrame({
        "rank": [1, 2, 3, 4, 5], "customer_ref": [f"P{i:010d}" for i in range(5)],
        "branch": ["Harare", "Harare", "Gweru", "Gweru", "Harare"], "segment": ["sme", "salaried", "sme", "informal", "salaried"],
        "p_churn_90d": [0.6, 0.5, 0.4, 0.1, 0.0], "value_12m_usd": [100.0, 100.0, 50.0, 100.0, 100.0],
        "value_at_risk_usd": [60.0, 50.0, 20.0, 10.0, 0.0],
        "urgency": ["act this week", "act this month", "watch", "watch", "watch"],
        "recommendation": ["retention"] * 5, "bal_change_vs_6m": [0.1] * 5,     # a raw feature column that must not leak
    })


def test_headline_counts_and_top_share():
    k = logic.headline(toy_worklist())
    assert k["customers"] == 5 and k["at_risk"] == 2
    assert k["value_at_risk_usd"] == 140.0
    assert k["top_n"] == 1 and abs(k["top_share"] - 60 / 140) < 1e-9
    assert logic.headline(toy_worklist().iloc[0:0])["value_at_risk_usd"] == 0.0


def test_worklist_table_uses_plain_labels_only():
    t = logic.worklist_table(toy_worklist())
    assert list(t.columns) == list(logic.WORKLIST_COLUMNS.values())
    assert not any("_" in c for c in t.columns)
    assert t.iloc[0]["Risk in 90 days"] == 60.0 and t.iloc[0]["Customer"] == "P0000000000"


def test_filter_rows():
    rows = toy_worklist()
    assert len(logic.filter_rows(rows, "Gweru", None)) == 2
    assert len(logic.filter_rows(rows, "Harare", "salaried")) == 2
    assert len(logic.filter_rows(rows, None, None)) == 5


def test_roi_curve_matches_point_arithmetic_and_extremes():
    rows = toy_worklist()
    p, v = rows["p_churn_90d"].to_numpy(), rows["value_12m_usd"].to_numpy()
    curve = logic.roi_curve(p, v, cost=1.0, incentive=10.0, save_rate=0.5, points=100)
    pulse = curve[curve["outreach"] == "Pulse ranking"].set_index("contacted")["net_return_usd"]
    rnd = curve[curve["outreach"] == "Random outreach"].set_index("contacted")["net_return_usd"]
    # zero contacts: nothing protected, nothing spent
    assert pulse.loc[0] == 0.0 and rnd.loc[0] == 0.0
    # every point on the Pulse line equals outcome() on the top-n rows
    for n in pulse.index:
        o = logic.outcome(p[:n], v[:n], n, 1.0, 10.0, 0.5)
        assert abs(pulse.loc[n] - o["net_return_usd"]) < 1e-9
    # everyone contacted: Pulse and random protect the same revenue, so the same net return
    n_all = len(p)
    assert abs(pulse.loc[n_all] - rnd.loc[n_all]) < 1e-9
    # top 2 by hand: protected = (0.6x100 + 0.5x100) x 0.5 = 55; cost = 2 + (1.1 x 0.5) x 10 = 7.5
    assert abs(pulse.loc[2] - (55 - 7.5)) < 1e-9
    best = logic.best_point(curve)
    assert best["contacted"] in pulse.index and best["net_return_usd"] == pulse.max()


def test_history_long_and_tables_rename_everything():
    pts = [{"month": "2026-09-30", "balance_usd": 10.0, "money_in_usd": 5.0, "salary_credit_usd": 0.0},
           {"month": "2026-10-31", "balance_usd": 0.0, "money_in_usd": 0.0, "salary_credit_usd": 0.0}]
    long = logic.history_long(pts)
    assert set(long["series"]) == {"Month-end balance", "Salary credit", "All money in"} and len(long) == 6
    comp = logic.comparison_table([{"model": "rules", "PR-AUC": 0.4, "ROC-AUC": 0.8, "capture@10%": 60.0, "lift@10%": 6.0, "capture@20%": 70.0},
                                   {"model": "lightgbm", "PR-AUC": 0.5, "ROC-AUC": 0.9, "capture@10%": 70.0, "lift@10%": 7.0, "capture@20%": 80.0}])
    assert comp.iloc[1]["Model"] == "Pulse (LightGBM)" and "Churners caught in top 10% (%)" in comp.columns
    fair = logic.fairness_table([{"attribute": "gender", "group": "F", "rows": 10, "churn_%": 3.0, "flag_%": 10.0, "capture_%": 70.0}])
    assert list(fair.columns) == list(logic.FAIRNESS_COLUMNS.values())
    seg = logic.segment_table([{"dimension": "segment", "group": "sme", "customers": 5, "churn_rate_pct": 2.0, "expected_churners": 0.1,
                                "value_12m_usd": 1.0, "value_at_risk_usd": 9.0, "share_of_var_pct": 50.0, "var_per_customer_usd": 1.8},
                               {"dimension": "branch", "group": "Gweru", "customers": 5, "churn_rate_pct": 2.0, "expected_churners": 0.1,
                                "value_12m_usd": 1.0, "value_at_risk_usd": 9.0, "share_of_var_pct": 50.0, "var_per_customer_usd": 1.8}], "segment")
    assert len(seg) == 1 and seg.iloc[0]["Group"] == "sme"
    for frame in (comp, fair, seg):
        assert not any("_" in c or "@" in c for c in frame.columns)
    assert logic.usd(1234.6) == "1,235 USD" and logic.pct(0.123, 1) == "12.3%"
