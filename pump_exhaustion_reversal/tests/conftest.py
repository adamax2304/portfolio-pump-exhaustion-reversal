from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture
def default_config_path() -> Path:
    return CONFIG_DIR / "default.yaml"


@pytest.fixture
def backtest_config_path() -> Path:
    return CONFIG_DIR / "backtest.yaml"
