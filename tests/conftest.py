"""Shared fixtures: a small synthetic dataset written into a temporary project root."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pulse.config import PROJECT_ROOT, load_settings
from pulse import synth

N_SMALL = 250


@pytest.fixture(scope="session")
def small_frames():
    return synth.generate(load_settings(), n_customers=N_SMALL, seed=7)


@pytest.fixture(scope="session")
def tmp_root(tmp_path_factory, small_frames) -> Path:
    """A throwaway project root with config/ copied and small data in data/raw."""
    root = tmp_path_factory.mktemp("pulse_root")
    shutil.copytree(PROJECT_ROOT / "config", root / "config")
    synth.write(small_frames, root / "data" / "raw")
    return root
