"""Phase 7: the four Streamlit pages rendered with Streamlit's AppTest against a live API.

Needs streamlit importable and the API reachable at PULSE_API_URL, which on the build machine
means running inside the dashboard container:

    docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dashboard \
        python -m pytest -q tests/test_dashboard_app.py

Skipped automatically when streamlit cannot be imported or the API is not up.
"""
import os
import re
import time

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard import api_client as api  # noqa: E402
from dashboard import views  # noqa: E402

health = api.health()
if health is None or health.get("status") != "ok":
    pytest.skip(f"API not reachable at {api.API}", allow_module_level=True)

RAW_NAMES = re.compile(r"\b(bal_[a-z0-9_]+|p_churn_\d+d|income_change|txn_count_[a-z0-9_]+|leak_share_[a-z0-9_]+|"
                       r"months_since_[a-z_]+|salary_[a-z0-9_]+|value_at_risk_usd|value_12m_usd|customer_ref)\b")
LOAD_BUDGET_S = 2.0


def run(page: str, **state) -> tuple[AppTest, float]:
    """Render a page twice: the first run fills the cache, the second is what a viewer sees.

    AppTest.from_function copies only the function's source, losing the module globals, so each
    page is run through a three-line script that imports the views module instead."""
    script = "import streamlit as st" + chr(10) + "from dashboard import views" + chr(10) + "views." + page + "()" + chr(10)
    at = AppTest.from_string(script, default_timeout=60)
    at.session_state["health"] = health
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    t0 = time.perf_counter()
    at.run()
    return at, time.perf_counter() - t0


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.title]
    parts += [h.value for h in at.subheader] + [m.label + " " + str(m.value) for m in at.metric]
    for df in at.dataframe:
        parts += list(map(str, df.value.columns))
    return "\n".join(parts)


def test_worklist_page_renders_fast_with_plain_labels():
    at, secs = run("worklist")
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) == 3 and len(at.dataframe) == 1
    table = at.dataframe[0].value
    assert list(table.columns)[:3] == ["Rank", "Customer", "Branch"] and len(table) > 0
    assert not RAW_NAMES.search(all_text(at)), RAW_NAMES.search(all_text(at)).group(0)
    assert secs < LOAD_BUDGET_S, f"worklist took {secs:.2f}s from cache"


def test_card_page_shows_reasons_action_script_and_history():
    ref = api.worklist()["customer_ref"].iloc[0]
    at, secs = run("card", ref=ref)
    assert not at.exception, [e.value for e in at.exception]
    text = all_text(at)
    card = api.get(f"/customers/{ref}/card")
    assert card["action"] in text and card["call_script"] in text
    for r in card["reasons"]:
        assert r["text"] in text and r["family_label"] in text
    assert "Last 12 months" in text
    assert not RAW_NAMES.search(text), RAW_NAMES.search(text).group(0)
    assert secs < LOAD_BUDGET_S, f"card took {secs:.2f}s from cache"


def test_insights_page_has_drivers_segments_models_and_fairness():
    at, secs = run("insights")
    assert not at.exception, [e.value for e in at.exception]
    text = all_text(at)
    assert "Top churn drivers" in text and "Value at risk by group" in text
    frames = [df.value for df in at.dataframe]
    cols = {c for f in frames for c in f.columns}
    assert "Pulse (LightGBM)" in str([f.to_dict() for f in frames]) and "Churners caught (%)" in cols
    assert not RAW_NAMES.search(text), RAW_NAMES.search(text).group(0)
    assert secs < LOAD_BUDGET_S, f"insights took {secs:.2f}s from cache"


def test_roi_page_extremes_zero_and_everyone():
    at, secs = run("roi")
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.slider) == 4 and len(at.metric) == 4
    assert secs < LOAD_BUDGET_S, f"roi took {secs:.2f}s from cache"
    total = at.slider[0].max
    # zero customers contacted: nothing protected, nothing spent, no uplift
    at.slider[0].set_value(0).run()
    values = {m.label: m.value for m in at.metric}
    assert values["Revenue protected"] == "0 USD" and values["Campaign cost"] == "0 USD" and values["Net return"] == "0 USD"
    # everyone contacted: Pulse and random reach the same customers, so the uplift is zero
    at.slider[0].set_value(total).run()
    net = next(m for m in at.metric if m.label == "Net return")
    assert net.delta.startswith("+0 ") or net.delta.startswith("-0 ") or net.delta.startswith("0 "), net.delta
    protected = next(m for m in at.metric if m.label == "Revenue protected")
    assert protected.delta.split(" ")[0] in ("+0", "-0", "0"), protected.delta
