"""Pseudonymised customer references.

Rule 7: customer ids are pseudonymised before they reach a dashboard or a log. The
reference is a salted SHA-256 hash of the id, truncated; the salt lives in settings.yaml.
The mapping back to the real id is written to a separate file that the API and the
dashboard never read.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def pseudonymise(ids, salt: str, length: int = 10) -> np.ndarray:
    def one(cid: str) -> str:
        return "P" + hashlib.sha256(f"{salt}:{cid}".encode("utf-8")).hexdigest()[:length].upper()
    return np.array([one(str(c)) for c in ids], dtype=object)


def id_map(ids, salt: str, length: int = 10) -> pd.DataFrame:
    ids = pd.Index(ids).unique()
    return pd.DataFrame({"customer_ref": pseudonymise(ids, salt, length), "customer_id": ids})
