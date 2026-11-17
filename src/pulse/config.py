"""Load config/settings.yaml and config/schema.yaml.

Everything else in the package goes through these two functions so that paths,
thresholds and column names are defined in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_settings(root: Path = PROJECT_ROOT) -> dict:
    return load_yaml(Path(root) / "config" / "settings.yaml")


def load_schema(root: Path = PROJECT_ROOT, path: Path | None = None) -> dict:
    return load_yaml(Path(path) if path else Path(root) / "config" / "schema.yaml")


def resolve(root: Path, rel: str) -> Path:
    """Turn a path from the YAML (relative to the project root) into an absolute path."""
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p
