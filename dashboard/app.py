"""Pulse dashboard placeholder (Phase 7 replaces this page).

Reads the worklist from the API so that `docker compose up` brings up both services now.
    streamlit run dashboard/app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("PULSE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Pulse", layout="wide")
st.title("Pulse: customers ranked by value at risk")
try:
    health = requests.get(f"{API}/health", timeout=5).json()
except Exception as e:  # the api may still be running the pipeline
    st.warning(f"API not reachable at {API}: {e}")
    st.stop()
if health["status"] != "ok":
    st.info(health["status"])
    st.stop()
st.caption(f"Snapshot {health['snapshot_date']}, {health['customers']:,} customers")
wl = requests.get(f"{API}/worklist", params={"limit": 200}, timeout=30).json()
st.dataframe(pd.DataFrame(wl["rows"]), use_container_width=True, hide_index=True)
