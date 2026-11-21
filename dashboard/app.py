"""Pulse dashboard: four pages in the order of the demo, all served from the FastAPI service.

    streamlit run dashboard/app.py            (PULSE_API_URL defaults to http://localhost:8000)

In Docker the dashboard service runs this with PULSE_API_URL=http://api:8000. The app refuses to
render until the api reports a loaded worklist, which on a clean checkout means the pipeline has
finished (a few minutes).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `from dashboard import ...` works under streamlit run

from dashboard import api_client as api  # noqa: E402
from dashboard import views  # noqa: E402

st.set_page_config(page_title="Pulse", page_icon=":material/monitor_heart:", layout="wide",
                   initial_sidebar_state="expanded")

health = api.health()
if health is None or health.get("status") != "ok":
    st.title("Pulse")
    msg = "The scoring service is not reachable yet." if health is None else health["status"]
    st.info(f"{msg} On a fresh start the pipeline runs first (about four minutes). This page retries every 15 seconds.")
    st.caption(f"API: {api.API}")
    time.sleep(15)
    st.rerun()
st.session_state["health"] = health

pages = {
    "worklist": st.Page(views.worklist, title="Worklist", icon=":material/list_alt:", default=True),
    "card": st.Page(views.card, title="Customer card", icon=":material/badge:", url_path="card"),
    "insights": st.Page(views.insights, title="Insights", icon=":material/insights:", url_path="insights"),
    "roi": st.Page(views.roi, title="ROI simulator", icon=":material/calculate:", url_path="roi"),
}
views.PAGES = pages

with st.sidebar:
    st.markdown("## Pulse")
    st.caption("Customer-intelligence copilot. Churn risk, value at risk, next best action.")
    nav = st.navigation(list(pages.values()), position="sidebar")
    st.divider()
    st.caption(f"Snapshot {health['snapshot_date']}  \n{health['customers']:,} customers scored")
    if st.button("Refresh data", help="Clear the cache after a new pipeline run and /reload"):
        api.clear()
        st.rerun()

nav.run()
