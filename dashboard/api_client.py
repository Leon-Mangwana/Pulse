"""Thin, cached client for the Pulse API.

Every page reads through these functions. Responses are cached in the Streamlit process for
CACHE_TTL seconds, so after the first call a page is an in-memory lookup (the two-second rule).
PULSE_API_URL points at the api service (docker-compose sets http://api:8000).
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("PULSE_API_URL", "http://localhost:8000").rstrip("/")
CACHE_TTL = 300
TIMEOUT = 30


def health() -> dict | None:
    """Not cached: this is how the app notices the pipeline is still running."""
    try:
        return requests.get(f"{API}/health", timeout=5).json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get(path: str, **params) -> dict:
    r = requests.get(f"{API}{path}", params=params or None, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def worklist() -> pd.DataFrame:
    """The whole ranked worklist as a frame (max_limit in settings.yaml caps it at 5,000 rows)."""
    body = get("/worklist", limit=5000)
    return pd.DataFrame(body["rows"])


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def simulate(top_n: int, cost_per_contact: float, incentive_cost: float, save_rate: float) -> dict:
    r = requests.post(f"{API}/simulate", json={"top_n": int(top_n), "cost_per_contact": float(cost_per_contact),
                                              "incentive_cost": float(incentive_cost), "save_rate": float(save_rate)},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def clear() -> None:
    st.cache_data.clear()
