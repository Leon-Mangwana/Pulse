"""Load config/settings.yaml and config/schema.yaml.

Everything else in the package goes through these two functions so that paths,
thresholds and column names are defined in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# PULSE_ROOT points every step at another project root (a rehearsal copy, or /app in Docker);
# PULSE_SCHEMA points at another schema file (one per dataset on rehearsal days).
PROJECT_ROOT = Path(os.environ.get("PULSE_ROOT") or Path(__file__).resolve().parents[2]).resolve()

TRANSACTION_TABLES = ("customers", "accounts", "transactions", "products")
SUMMARY_TABLE = "customer_summary"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_settings(root: Path = PROJECT_ROOT) -> dict:
    return load_yaml(Path(root) / "config" / "settings.yaml")


def schema_path(root: Path = PROJECT_ROOT, path: Path | None = None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("PULSE_SCHEMA")
    return resolve(root, env) if env else Path(root) / "config" / "schema.yaml"


def load_schema(root: Path = PROJECT_ROOT, path: Path | None = None) -> dict:
    return load_yaml(schema_path(root, path))


def data_mode(schema: dict) -> str:
    """'transactions' when the schema maps the four raw tables, 'summary' when it maps only
    one customer-level table with a churn flag (the fallback path, see docs/runbook.md)."""
    has_tx = all(t in schema for t in TRANSACTION_TABLES)
    has_summary = SUMMARY_TABLE in schema
    if has_tx and not has_summary:
        return "transactions"
    if has_summary and not has_tx:
        return "summary"
    if has_tx and has_summary:
        raise ValueError("schema.yaml maps both the transaction tables and customer_summary: keep one")
    missing = [t for t in TRANSACTION_TABLES if t not in schema]
    raise ValueError(f"schema.yaml must map either all of {list(TRANSACTION_TABLES)} (missing {missing}) "
                     f"or a single '{SUMMARY_TABLE}' table")


def current_mode(root: Path = PROJECT_ROOT) -> str:
    return data_mode(load_schema(root))


def resolve(root: Path, rel: str) -> Path:
    """Turn a path from the YAML (relative to the project root) into an absolute path."""
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p
