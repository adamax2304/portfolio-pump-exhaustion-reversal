from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import first_passage as fp


def _flat_then_move_ohlcv(moves: dict[int, tuple[float, float]], n: int = 20, base: float = 100.0) -> pd.DataFrame:
    """moves: {bar_index: (low, high)} overrides on top of a flat base series."""
    high = [base] * n
    low = [base] * n
    close = [base] * n
    for idx, (lo, hi) in moves.items():
        low[idx], high[idx] = lo, hi
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


def test_first_passage_outcome_profit_hit_first():
    ohlcv = _flat_then_move_ohlcv({3: (94.0, 100.0)})  # bar 3 dips to 94 - within a 5% profit barrier from 100
    result = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=10)
    assert result["outcome"] == "profit"
    assert result["bars"] == 3
    assert result["right_censored"] is False


def test_first_passage_outcome_adverse_hit_first():
    ohlcv = _flat_then_move_ohlcv({2: (100.0, 108.0)})  # bar 2 spikes to 108 - breaches a 5% adverse barrier
    result = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=10)
    assert result["outcome"] == "adverse"
    assert result["bars"] == 2


def test_first_passage_outcome_same_bar_ambiguity_adverse_wins():
    ohlcv = _flat_then_move_ohlcv({2: (93.0, 108.0)})  # one bar touches BOTH barriers
    result = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=10)
    assert result["outcome"] == "adverse"


def test_first_passage_outcome_timeout_when_neither_barrier_touched():
    ohlcv = _flat_then_move_ohlcv({})
    result = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "timeout"
    assert result["bars"] == 5
    assert result["right_censored"] is False  # plenty of bars existed beyond the horizon


def test_first_passage_outcome_right_censored_when_data_runs_out():
    ohlcv = _flat_then_move_ohlcv({}, n=6)  # only 6 bars total, anchor at 0
    result = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=50)
    assert result["outcome"] == "timeout"
    assert result["right_censored"] is True


def test_first_passage_outcome_invalid_anchor_returns_censored_timeout():
    ohlcv = _flat_then_move_ohlcv({})
    result = fp.first_passage_outcome(ohlcv, anchor_idx=999, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "timeout"
    assert result["right_censored"] is True


def test_first_passage_outcome_tracks_mae_mfe_before_adverse_hit():
    # anchor=100; bar1 dips to 97 (favorable move, tracked as MFE), bar2 spikes
    # to 103 which breaches a tight 2% adverse barrier (102) -> adverse wins there.
    ohlcv = _flat_then_move_ohlcv({1: (97.0, 100.0), 2: (100.0, 103.0)})
    result = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.02, horizon_bars=10)
    assert result["outcome"] == "adverse"
    assert result["bars"] == 2
    assert result["mfe_pct"] == pytest.approx(0.03)  # bar1's low=97 -> (100-97)/100
    assert result["mae_pct"] == pytest.approx(0.03)  # bar2's high=103 -> (103-100)/100


def test_first_passage_outcome_timeout_reports_return_at_exit():
    ohlcv = _flat_then_move_ohlcv({3: (98.5, 100.5)})  # a small move, neither barrier touched
    result = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "timeout"
    # exit bar (index 5) is flat at 100 (base) -> return_pct_at_exit == 0
    assert result["return_pct_at_exit"] == pytest.approx(0.0)
    assert result["mae_pct"] == pytest.approx(0.005)
    assert result["mfe_pct"] == pytest.approx(0.015)


def test_first_passage_outcome_empty_result_has_nan_mae_mfe():
    ohlcv = _flat_then_move_ohlcv({})
    result = fp.first_passage_outcome(ohlcv, anchor_idx=999, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    assert np.isnan(result["mae_pct"])
    assert np.isnan(result["mfe_pct"])
    assert np.isnan(result["return_pct_at_exit"])


def test_phase_2b_barrier_families_all_r_based():
    for name, spec in fp.PHASE_2B_BARRIER_FAMILIES.items():
        assert spec["kind"] == "r", name
        assert spec["adverse_r"] == pytest.approx(1.0)


def test_label_event_across_families_accepts_custom_families():
    n = 300
    rng = np.random.default_rng(4)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})
    results = fp.label_event_across_families(ohlcv, anchor_idx=50, bar_interval_ms=300_000, families=fp.PHASE_2B_BARRIER_FAMILIES)
    assert set(results) == set(fp.PHASE_2B_BARRIER_FAMILIES)


def test_resolve_r_based_barriers():
    profit, adverse = fp.resolve_r_based_barriers(stop_distance_pct=0.02, profit_r_multiple=2.0, adverse_r_multiple=1.0)
    assert profit == pytest.approx(0.04)
    assert adverse == pytest.approx(0.02)


def test_compute_atr_stop_distance_pct_reasonable_on_synthetic_series():
    n = 30
    rng = np.random.default_rng(2)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + 1.0
    low = close - 1.0
    ohlcv = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    val = fp.compute_atr_stop_distance_pct(ohlcv, anchor_idx=20, atr_length=14)
    assert 0.0 < val < 0.2  # a sane fraction, not NaN or absurd


def test_compute_atr_stop_distance_pct_nan_at_series_edges():
    ohlcv = _flat_then_move_ohlcv({}, n=10)
    assert np.isnan(fp.compute_atr_stop_distance_pct(ohlcv, anchor_idx=0, atr_length=14))
    assert np.isnan(fp.compute_atr_stop_distance_pct(ohlcv, anchor_idx=999, atr_length=14))


def test_label_event_across_families_covers_all_families():
    n = 300
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})
    results = fp.label_event_across_families(ohlcv, anchor_idx=50, bar_interval_ms=300_000)
    assert set(results) == set(fp.LABEL_FAMILIES)
    for outcome in results.values():
        assert outcome["outcome"] in ("profit", "adverse", "timeout")


def test_summarize_outcomes_basic():
    outcomes = [
        {"outcome": "profit", "bars": 3, "right_censored": False},
        {"outcome": "profit", "bars": 5, "right_censored": False},
        {"outcome": "adverse", "bars": 2, "right_censored": False},
        {"outcome": "timeout", "bars": 10, "right_censored": True},
    ]
    summary = fp.summarize_outcomes(outcomes)
    assert summary["n"] == 4
    assert summary["p_profit"] == pytest.approx(0.5)
    assert summary["p_adverse"] == pytest.approx(0.25)
    assert summary["p_timeout"] == pytest.approx(0.25)
    assert summary["median_bars_to_profit"] == pytest.approx(4.0)


def test_summarize_outcomes_empty():
    summary = fp.summarize_outcomes([])
    assert summary["n"] == 0
    assert np.isnan(summary["p_profit"])


def test_summarize_outcomes_reports_counts_and_timeout_return_and_mae_mfe():
    outcomes = [
        {"outcome": "profit", "bars": 3, "right_censored": False, "mae_pct": 0.01, "mfe_pct": 0.05, "return_pct_at_exit": 0.05},
        {"outcome": "adverse", "bars": 2, "right_censored": False, "mae_pct": 0.03, "mfe_pct": 0.0, "return_pct_at_exit": -0.03},
        {"outcome": "timeout", "bars": 10, "right_censored": True, "mae_pct": 0.01, "mfe_pct": 0.02, "return_pct_at_exit": 0.015},
        {"outcome": "timeout", "bars": 10, "right_censored": True, "mae_pct": 0.02, "mfe_pct": 0.01, "return_pct_at_exit": -0.005},
    ]
    summary = fp.summarize_outcomes(outcomes)
    assert summary["n_profit"] == 1
    assert summary["n_adverse"] == 1
    assert summary["n_timeout"] == 2
    assert summary["avg_return_at_timeout_pct"] == pytest.approx((0.015 - 0.005) / 2)
    assert summary["median_return_at_timeout_pct"] == pytest.approx((0.015 - 0.005) / 2)
    assert summary["avg_mae_pct"] == pytest.approx((0.01 + 0.03 + 0.01 + 0.02) / 4)
    assert summary["avg_mfe_pct"] == pytest.approx((0.05 + 0.0 + 0.02 + 0.01) / 4)
