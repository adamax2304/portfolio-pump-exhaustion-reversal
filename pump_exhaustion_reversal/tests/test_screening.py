from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.data import screening


def _daily_df(prices: list[float], start_ts: int = 0) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({"timestamp": [start_ts + i * 86_400_000 for i in range(n)], "close": prices})


def test_compute_screening_score_finds_the_biggest_window():
    # a gradual 3-day ramp to +30% (no single day anywhere near that big),
    # then flat again - only the 3-day window can see the full move
    prices = [100, 100, 108, 116, 130, 100, 100, 100]
    coin = _daily_df(prices)
    btc = _daily_df([100] * len(prices))  # flat BTC -> no adjustment
    result = screening.compute_screening_score(coin, btc)
    assert result["max_move_pct"] == pytest.approx(0.30, abs=1e-6)
    assert result["max_move_window_days"] == 3
    assert result["max_move_ts"] == coin["timestamp"].iloc[4]


def test_compute_screening_score_adjusts_for_btc_move():
    # coin and BTC both rally +20% together -> abnormal (BTC-adjusted) move ~0
    coin = _daily_df([100, 105, 110, 120])
    btc = _daily_df([100, 105, 110, 120])
    result = screening.compute_screening_score(coin, btc)
    assert result["max_move_pct"] == pytest.approx(0.0, abs=1e-6)


def test_compute_screening_score_too_short_series_is_nan():
    coin = _daily_df([100])
    btc = _daily_df([100])
    result = screening.compute_screening_score(coin, btc)
    assert result["max_move_window_days"] is None
    assert np.isnan(result["max_move_pct"])


def test_compute_screening_score_handles_empty_btc_reference():
    coin = _daily_df([100, 100, 150])
    btc = pd.DataFrame(columns=["timestamp", "close"])
    result = screening.compute_screening_score(coin, btc)
    # with no BTC reference, "adjustment" is 0 (falls back to raw move)
    assert result["max_move_pct"] == pytest.approx(0.5, abs=1e-6)


def test_rank_candidates_filters_and_caps():
    scores = [
        {"symbol": "A", "max_move_pct": 0.50, "max_move_window_days": 1, "max_move_ts": 0},
        {"symbol": "B", "max_move_pct": 0.10, "max_move_window_days": 1, "max_move_ts": 0},
        {"symbol": "C", "max_move_pct": 0.30, "max_move_window_days": 3, "max_move_ts": 0},
        {"symbol": "D", "max_move_pct": float("nan"), "max_move_window_days": None, "max_move_ts": None},
    ]
    result = screening.rank_candidates(scores, min_move_pct=0.15)
    assert list(result["symbol"]) == ["A", "C"]

    capped = screening.rank_candidates(scores, min_move_pct=0.0, top_n=1)
    assert list(capped["symbol"]) == ["A"]


def test_rank_candidates_empty_input():
    result = screening.rank_candidates([], min_move_pct=0.1)
    assert result.empty
