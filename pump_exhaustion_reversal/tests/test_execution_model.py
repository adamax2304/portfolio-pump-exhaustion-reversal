from __future__ import annotations

import math

import pytest

from pump_exhaustion_reversal.backtest import execution_model as em
from pump_exhaustion_reversal.config import ExecutionModelConfig


def test_apply_slippage_always_against_the_trader():
    assert em.apply_slippage(100.0, 10, side="sell") < 100.0
    assert em.apply_slippage(100.0, 10, side="buy") > 100.0
    with pytest.raises(ValueError):
        em.apply_slippage(100.0, 10, side="hold")


def test_compute_slippage_bps_scales_with_high_volatility():
    cfg = ExecutionModelConfig(default_slippage_bps=10, high_vol_slippage_multiplier=2.5)
    assert em.compute_slippage_bps(cfg, high_volatility=False) == pytest.approx(10.0)
    assert em.compute_slippage_bps(cfg, high_volatility=True) == pytest.approx(25.0)


def test_compute_fee_maker_vs_taker():
    cfg = ExecutionModelConfig(taker_fee_bps=5, maker_fee_bps=2)
    assert em.compute_fee(10_000, cfg, is_maker=False) == pytest.approx(5.0)
    assert em.compute_fee(10_000, cfg, is_maker=True) == pytest.approx(2.0)


def test_compute_funding_cost_short_receives_when_funding_positive():
    cost = em.compute_funding_cost(notional=10_000, funding_rate=0.0001, is_short=True)
    assert cost < 0  # negative cost = a credit/receipt for the short
    assert cost == pytest.approx(-1.0)


def test_compute_funding_cost_short_pays_when_funding_negative():
    cost = em.compute_funding_cost(notional=10_000, funding_rate=-0.0001, is_short=True)
    assert cost > 0
    assert cost == pytest.approx(1.0)


def test_max_fillable_notional_unknown_volume_is_unconstrained():
    cfg = ExecutionModelConfig(max_volume_participation_pct=2)
    assert math.isinf(em.max_fillable_notional(None, cfg))
    assert math.isinf(em.max_fillable_notional(float("nan"), cfg))
    assert em.max_fillable_notional(1_000_000, cfg) == pytest.approx(20_000)


def test_apply_participation_cap_caps_and_reports_ratio():
    cfg = ExecutionModelConfig(max_volume_participation_pct=2)
    result = em.apply_participation_cap(desired_notional=50_000, bar_quote_volume=1_000_000, cfg=cfg)
    assert result["capped"]
    assert result["filled_notional"] == pytest.approx(20_000)
    assert result["fill_ratio"] == pytest.approx(0.4)


def test_simulate_entry_and_exit_fills_directionally_consistent():
    cfg = ExecutionModelConfig(default_slippage_bps=10, stop_slippage_bps=20, high_vol_slippage_multiplier=2.5)
    entry = em.simulate_entry_fill(100.0, cfg, high_volatility=False)
    assert entry["fill_price"] < 100.0  # selling to open a short -> worse (lower) fill

    stop = em.simulate_stop_fill(105.0, cfg)
    assert stop["fill_price"] > 105.0  # buying back at a stop -> worse (higher) fill

    tp = em.simulate_tp_fill(95.0, cfg, high_volatility=True)
    assert tp["fill_price"] > 95.0
    assert tp["slippage_bps"] == pytest.approx(25.0)
