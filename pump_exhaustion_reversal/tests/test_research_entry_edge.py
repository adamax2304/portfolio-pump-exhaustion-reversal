from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest.simulator import EventRecord
from pump_exhaustion_reversal.config import ExecutionModelConfig
from pump_exhaustion_reversal.research import entry_edge as ee


def _ohlcv(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    # rows: (open, high, low, close, volume)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_resolve_entry_price_confirmation_close():
    ohlcv = _ohlcv([(100, 101, 99, 100.5, 10), (100.5, 102, 100, 101.5, 10)])
    assert ee.resolve_entry_price(ohlcv, 0, "confirmation_close") == pytest.approx(100.5)


def test_resolve_entry_price_next_bar_open():
    ohlcv = _ohlcv([(100, 101, 99, 100.5, 10), (100.5, 102, 100, 101.5, 10)])
    assert ee.resolve_entry_price(ohlcv, 0, "next_bar_open") == pytest.approx(100.5)


def test_resolve_entry_price_out_of_range_returns_none():
    ohlcv = _ohlcv([(100, 101, 99, 100.5, 10)])
    assert ee.resolve_entry_price(ohlcv, 0, "next_bar_open") is None
    assert ee.resolve_entry_price(ohlcv, 5, "confirmation_close") is None


def test_resolve_entry_price_unknown_mode_raises():
    ohlcv = _ohlcv([(100, 101, 99, 100.5, 10)])
    with pytest.raises(ValueError):
        ee.resolve_entry_price(ohlcv, 0, "bogus")


def test_compute_atr_stop_price_above_signal_high():
    rows = [(100.0, 101.0, 99.0, 100.0, 10.0)] * 20
    ohlcv = _ohlcv(rows)
    stop = ee.compute_atr_stop_price(ohlcv, 15, atr_multiple=0.25, atr_length=14)
    # true range is flat at 2.0 every bar -> ATR ~2.0; stop = high(101) + 0.25*2 = 101.5
    assert stop == pytest.approx(101.5, abs=0.05)


def test_compute_atr_stop_price_out_of_range_returns_none():
    ohlcv = _ohlcv([(100, 101, 99, 100.5, 10)])
    assert ee.compute_atr_stop_price(ohlcv, 5, atr_multiple=0.25) is None


def test_first_passage_from_entry_matches_first_passage_outcome_for_confirmation_close():
    from pump_exhaustion_reversal.research import first_passage as fp

    ohlcv = _ohlcv([(100, 101, 99, 100.0, 10), (100, 100, 94.0, 96.0, 10), (96, 97, 95, 96.5, 10)])
    reference = fp.first_passage_outcome(ohlcv, anchor_idx=0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    entry_price = float(ohlcv["close"].iloc[0])
    result = ee.first_passage_from_entry(ohlcv, walk_start_idx=1, entry_price=entry_price, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    assert result["outcome"] == reference["outcome"] == "profit"
    assert result["bars"] == reference["bars"]
    assert result["mae_pct"] == pytest.approx(reference["mae_pct"])
    assert result["mfe_pct"] == pytest.approx(reference["mfe_pct"])


def test_first_passage_from_entry_next_bar_open_uses_explicit_entry_price():
    # entry fills at bar1's open (110); bar1 itself spikes to 116 which
    # breaches a 5% adverse barrier off that 110 fill (115.5) - must be
    # caught even though it's the SAME bar as the fill.
    ohlcv = _ohlcv([(100, 101, 99, 100.0, 10), (110, 116, 109, 112, 10)])
    result = ee.first_passage_from_entry(ohlcv, walk_start_idx=1, entry_price=110.0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "adverse"
    assert result["bars"] == 1


def test_first_passage_from_entry_out_of_range_returns_censored_timeout():
    ohlcv = _ohlcv([(100, 101, 99, 100.0, 10)])
    result = ee.first_passage_from_entry(ohlcv, walk_start_idx=5, entry_price=100.0, profit_barrier_pct=0.05, adverse_barrier_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "timeout"
    assert result["right_censored"] is True


def test_compute_r_trades_profit_adverse_timeout():
    execution_cfg = ExecutionModelConfig(taker_fee_bps=5, default_slippage_bps=10, stop_slippage_bps=20)
    outcomes = [
        {"outcome": "profit", "bars": 3, "mae_pct": 0.01, "mfe_pct": 0.04, "return_pct_at_exit": np.nan, "stop_distance_pct": 0.02},
        {"outcome": "adverse", "bars": 2, "mae_pct": 0.02, "mfe_pct": 0.0, "return_pct_at_exit": np.nan, "stop_distance_pct": 0.02},
        {"outcome": "timeout", "bars": 10, "mae_pct": 0.01, "mfe_pct": 0.01, "return_pct_at_exit": 0.01, "stop_distance_pct": 0.02},
    ]
    trades = ee.compute_r_trades(outcomes, profit_r_multiple=2.0, adverse_r_multiple=1.0, execution_cfg=execution_cfg, bar_interval_ms=300_000)
    assert len(trades) == 3
    assert trades[0]["gross_r"] == pytest.approx(2.0)
    assert trades[1]["gross_r"] == pytest.approx(-1.0)
    assert trades[2]["gross_r"] == pytest.approx(0.5)  # 0.01 / 0.02
    # fees_r = 2*5bps/10000/0.02 = 0.05 for every trade
    assert trades[0]["fees_r"] == pytest.approx(0.05)
    # entry leg always uses default_slippage_bps (no high_vol flag supplied -> multiplier not applied)
    assert trades[0]["entry_slippage_r"] == pytest.approx(10 / 10_000 / 0.02)
    assert trades[1]["entry_slippage_r"] == pytest.approx(10 / 10_000 / 0.02)
    # profit exit -> TP-style fill (default_slippage_bps); adverse exit -> STOP-style fill (stop_slippage_bps) - never both on one leg
    assert trades[0]["exit_slippage_r"] == pytest.approx(10 / 10_000 / 0.02)
    assert trades[1]["exit_slippage_r"] == pytest.approx(20 / 10_000 / 0.02)
    # stop_overrun_premium_r (informational, already inside exit_slippage_r) only nonzero when the exit used the stop-style fill
    assert trades[0]["stop_overrun_premium_r"] == pytest.approx(0.0)
    assert trades[1]["stop_overrun_premium_r"] == pytest.approx((20 - 10) / 10_000 / 0.02)
    # timeout with a favorable return_pct_at_exit (0.01 > 0) -> directional TP-style exit fill, same rate as a clean profit exit
    assert trades[2]["exit_slippage_r"] == pytest.approx(10 / 10_000 / 0.02)
    assert trades[2]["stop_overrun_premium_r"] == pytest.approx(0.0)
    assert trades[0]["net_r"] < trades[0]["gross_r"]
    assert trades[0]["net_r"] == pytest.approx(2.0 - 0.05 - 0.05 - 0.05)
    assert trades[1]["net_r"] == pytest.approx(-1.0 - 0.05 - 0.05 - 0.10)


def test_compute_r_trades_skips_missing_stop_distance():
    execution_cfg = ExecutionModelConfig()
    outcomes = [{"outcome": "profit", "bars": 1, "mae_pct": 0.0, "mfe_pct": 0.0, "return_pct_at_exit": np.nan, "stop_distance_pct": np.nan}]
    trades = ee.compute_r_trades(outcomes, profit_r_multiple=2.0, adverse_r_multiple=1.0, execution_cfg=execution_cfg, bar_interval_ms=300_000)
    assert trades == []


def test_summarize_net_expectancy_basic():
    r_trades = [
        {"outcome": "profit", "gross_r": 2.0, "fees_r": 0.05, "entry_slippage_r": 0.05, "exit_slippage_r": 0.05, "funding_r": 0.0, "stop_overrun_premium_r": 0.0, "net_r": 1.85, "mae_r": 0.5, "mfe_r": 2.0},
        {"outcome": "adverse", "gross_r": -1.0, "fees_r": 0.05, "entry_slippage_r": 0.05, "exit_slippage_r": 0.10, "funding_r": 0.0, "stop_overrun_premium_r": 0.05, "net_r": -1.65, "mae_r": 1.0, "mfe_r": 0.2},
    ]
    summary = ee.summarize_net_expectancy(r_trades)
    assert summary["n"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["net_expectancy_r"] == pytest.approx((1.85 - 1.65) / 2)
    assert summary["profit_factor"] == pytest.approx(1.85 / 1.65)
    assert summary["exit_slippage_r"] == pytest.approx((0.05 + 0.10) / 2)


def test_resolve_exit_slippage_bps_never_combines_default_and_stop_rate():
    cfg = ExecutionModelConfig(default_slippage_bps=10, stop_slippage_bps=20, high_vol_slippage_multiplier=2.5)
    # every branch returns EXACTLY one of the two rates, never a sum of both
    assert ee.resolve_exit_slippage_bps("adverse", np.nan, cfg, high_vol=False) == pytest.approx(20)
    assert ee.resolve_exit_slippage_bps("profit", np.nan, cfg, high_vol=False) == pytest.approx(10)
    assert ee.resolve_exit_slippage_bps("timeout", 0.02, cfg, high_vol=False) == pytest.approx(10)  # favorable timeout -> TP-style
    assert ee.resolve_exit_slippage_bps("timeout", -0.01, cfg, high_vol=False) == pytest.approx(20)  # adverse timeout -> STOP-style
    # high_vol only multiplies the default (TP-style) rate - stop_slippage_bps is a fixed model input, not scaled by volatility
    assert ee.resolve_exit_slippage_bps("profit", np.nan, cfg, high_vol=True) == pytest.approx(25)
    assert ee.resolve_exit_slippage_bps("adverse", np.nan, cfg, high_vol=True) == pytest.approx(20)


def test_compute_r_trades_high_vol_multiplies_entry_and_tp_style_slippage():
    cfg = ExecutionModelConfig(taker_fee_bps=5, default_slippage_bps=10, stop_slippage_bps=20, high_vol_slippage_multiplier=2.5)
    outcomes = [{"outcome": "profit", "bars": 1, "mae_pct": 0.0, "mfe_pct": 0.0, "return_pct_at_exit": np.nan, "stop_distance_pct": 0.02, "high_vol": True}]
    trades = ee.compute_r_trades(outcomes, profit_r_multiple=2.0, adverse_r_multiple=1.0, execution_cfg=cfg, bar_interval_ms=300_000)
    assert trades[0]["entry_slippage_r"] == pytest.approx(10 * 2.5 / 10_000 / 0.02)
    assert trades[0]["exit_slippage_r"] == pytest.approx(10 * 2.5 / 10_000 / 0.02)


def test_compute_r_trades_funding_r_passes_through_into_net_r():
    cfg = ExecutionModelConfig(taker_fee_bps=0, default_slippage_bps=0, stop_slippage_bps=0)
    outcomes = [{"outcome": "profit", "bars": 5, "mae_pct": 0.0, "mfe_pct": 0.0, "return_pct_at_exit": np.nan, "stop_distance_pct": 0.02, "funding_r": 0.03}]
    trades = ee.compute_r_trades(outcomes, profit_r_multiple=2.0, adverse_r_multiple=1.0, execution_cfg=cfg, bar_interval_ms=300_000)
    assert trades[0]["funding_r"] == pytest.approx(0.03)
    assert trades[0]["net_r"] == pytest.approx(2.0 + 0.03)  # zero fees/slippage in this cfg -> funding is the only adjustment


def test_summarize_net_expectancy_empty():
    assert ee.summarize_net_expectancy([]) == {"n": 0}


def test_bootstrap_net_expectancy_ci_shape():
    r_trades = [{"outcome": "profit" if i % 3 else "adverse", "net_r": 1.5 if i % 3 else -1.0} for i in range(30)]
    result = ee.bootstrap_net_expectancy_ci(r_trades, n_resamples=200)
    assert result["n"] == 30
    assert "win_rate_ci" in result and "net_expectancy_ci" in result
    assert result["win_rate_ci"]["ci_low"] <= result["win_rate_ci"]["statistic"] <= result["win_rate_ci"]["ci_high"]


def test_stop_distance_audit_percentiles_and_ratio():
    rows = []
    for i, stop_pct in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
        rows.append({"event_id": f"e{i}", "exchange": "binance", "entry_mode": "confirmation_close", "atr_stop_multiple": 0.25, "stop_distance_pct": stop_pct})
        rows.append({"event_id": f"e{i}", "exchange": "binance", "entry_mode": "next_bar_open", "atr_stop_multiple": 0.25, "stop_distance_pct": stop_pct + 0.001})
    candidates_df = pd.DataFrame(rows)
    execution_cfg = ExecutionModelConfig(taker_fee_bps=5, default_slippage_bps=10)
    audit = ee.stop_distance_audit(candidates_df, execution_cfg)
    assert len(audit) == 1  # one (exchange, atr_multiple) group
    row = audit.iloc[0]
    assert row["n"] == 5  # next_bar_open rows excluded, one row per event after dedup
    assert row["stop_distance_bps_p50"] == pytest.approx(300.0)  # median of [100,200,300,400,500] bps
    # round trip cost = 2*10 + 2*5 = 30bps; at median stop 300bps -> 10%
    assert row["round_trip_cost_over_stop_distance_pct_median"] == pytest.approx(10.0)


def test_stop_distance_audit_empty():
    assert ee.stop_distance_audit(pd.DataFrame(), ExecutionModelConfig()).empty


def _outcomes_for_cost_sweep() -> list[dict]:
    return [
        {"outcome": "profit", "bars": 2, "mae_pct": 0.0, "mfe_pct": 0.04, "return_pct_at_exit": np.nan, "stop_distance_pct": 0.02},
        {"outcome": "adverse", "bars": 1, "mae_pct": 0.02, "mfe_pct": 0.0, "return_pct_at_exit": np.nan, "stop_distance_pct": 0.02},
        {"outcome": "profit", "bars": 3, "mae_pct": 0.0, "mfe_pct": 0.04, "return_pct_at_exit": np.nan, "stop_distance_pct": 0.02},
    ]


def test_cost_sensitivity_sweep_zero_profile_has_no_costs():
    base_cfg = ExecutionModelConfig(taker_fee_bps=5, default_slippage_bps=10, stop_slippage_bps=20)
    sweep = ee.cost_sensitivity_sweep(_outcomes_for_cost_sweep(), profit_r_multiple=2.0, adverse_r_multiple=1.0, base_execution_cfg=base_cfg, bar_interval_ms=300_000)
    assert set(sweep["profile"]) == {"ZERO", "LOW", "BASE", "HIGH"}
    zero_row = sweep[sweep["profile"] == "ZERO"].iloc[0]
    assert zero_row["net_expectancy_r"] == pytest.approx(zero_row["gross_expectancy_r"])
    # costs should only ever increase (or leave unchanged) as the profile gets more expensive - net expectancy monotonically non-increasing ZERO->LOW->BASE->HIGH
    ordered = sweep.set_index("profile").loc[["ZERO", "LOW", "BASE", "HIGH"]]["net_expectancy_r"]
    assert list(ordered) == sorted(ordered, reverse=True)


def test_cost_multiplier_curve_zero_multiplier_matches_gross():
    base_cfg = ExecutionModelConfig(taker_fee_bps=5, default_slippage_bps=10, stop_slippage_bps=20)
    curve = ee.cost_multiplier_curve(_outcomes_for_cost_sweep(), 2.0, 1.0, base_cfg, 300_000, multipliers=(0.0, 1.0, 2.0))
    zero_row = curve[curve["cost_multiplier"] == 0.0].iloc[0]
    full_summary = ee.summarize_net_expectancy(ee.compute_r_trades(_outcomes_for_cost_sweep(), 2.0, 1.0, base_cfg, 300_000))
    assert zero_row["net_expectancy_r"] == pytest.approx(full_summary["gross_expectancy_r"])


def test_estimate_breakeven_cost_multiplier_interpolates():
    curve_df = pd.DataFrame({"cost_multiplier": [0.0, 1.0, 2.0], "net_expectancy_r": [0.5, -0.5, -1.5]})
    breakeven = ee.estimate_breakeven_cost_multiplier(curve_df)
    assert breakeven == pytest.approx(0.5)  # crosses zero exactly halfway between 0.0 (0.5) and 1.0 (-0.5)


def test_estimate_breakeven_cost_multiplier_none_when_always_negative():
    curve_df = pd.DataFrame({"cost_multiplier": [0.0, 1.0, 2.0], "net_expectancy_r": [-0.1, -0.5, -1.5]})
    assert ee.estimate_breakeven_cost_multiplier(curve_df) is None


def test_simulate_staged_entry_leg1_only_stopped_out():
    # leg1 signal at bar 0 (close=100), entry at bar1 open=100; stop=110.
    # leg2 (failed_reclaim) never fires; bar1 itself spikes to 111 -> stop hit with only leg1 open.
    ohlcv = _ohlcv([(100, 100, 99, 100.0, 10), (100, 111, 100, 105, 10)])
    result = ee.simulate_staged_entry(ohlcv, leg1_signal_idx=0, entry_mode="confirmation_close", leg2_signal_idx=None, stop_price=110.0, profit_r_multiple=2.0, horizon_bars=5)
    assert result["outcome"] == "adverse"
    assert result["leg2_activated"] is False
    assert result["planned_r"] == pytest.approx(0.5)
    assert result["realized_r"] == pytest.approx(-0.5)  # leg1 alone loses its full 0.5R


def test_simulate_staged_entry_leg2_activates_then_profits():
    # leg1 signal bar0 (close=100) -> entry@bar1 open=100, risk_per_unit=10 (stop=110).
    # leg2 signal bar1 (close=100) -> entry@bar2 open=100, SAME stop=110, SAME risk_per_unit=10 (symmetric case).
    # profit_price = 100 - 2*10 = 80. bar3 dips to 79 with both legs open -> full 1R profit.
    ohlcv = _ohlcv(
        [
            (100, 101, 99, 100.0, 10),  # bar0: leg1 signal bar
            (100, 101, 99, 100.0, 10),  # bar1: leg1 fills here (open=100); also leg2 signal bar (close=100)
            (100, 101, 99, 100.0, 10),  # bar2: leg2 fills here (open=100)
            (100, 101, 79, 90, 10),  # bar3: dips to 79 <= profit_price(80) with both legs active
        ]
    )
    result = ee.simulate_staged_entry(ohlcv, leg1_signal_idx=0, entry_mode="confirmation_close", leg2_signal_idx=1, stop_price=110.0, profit_r_multiple=2.0, horizon_bars=10)
    assert result["outcome"] == "profit"
    assert result["leg2_activated"] is True
    assert result["planned_r"] == pytest.approx(1.0)
    assert result["realized_r"] == pytest.approx(2.0)  # symmetric legs, both hit the full 2R profit price


def test_simulate_staged_entry_invalid_geometry_returns_none():
    # leg1 signal bar0 close=100 -> entry@bar1 open=100; stop=95 is BELOW entry -> invalid for a short.
    ohlcv = _ohlcv([(100, 100, 99, 100.0, 10), (100, 101, 99, 100.5, 10)])
    assert ee.simulate_staged_entry(ohlcv, leg1_signal_idx=0, entry_mode="confirmation_close", leg2_signal_idx=None, stop_price=95.0, profit_r_multiple=2.0, horizon_bars=5) is None


def test_simulate_staged_entry_leg2_never_reached_times_out_leg1_only():
    ohlcv = _ohlcv([(100, 100, 99, 100.0, 10)] + [(100, 101, 99, 100.0, 10)] * 4)
    result = ee.simulate_staged_entry(ohlcv, leg1_signal_idx=0, entry_mode="confirmation_close", leg2_signal_idx=None, stop_price=110.0, profit_r_multiple=2.0, horizon_bars=4)
    assert result["outcome"] == "timeout"
    assert result["leg2_activated"] is False
    assert result["planned_r"] == pytest.approx(0.5)


def _fake_event(event_id: str, attempt_count: int, score_peaks: list[float], reasons: list[str | None]) -> EventRecord:
    structure_breaks: list[int | None] = [10 * (i + 1) for i in range(attempt_count)]
    return EventRecord(
        event_id=event_id, exchange="binance", symbol="TESTUSDT", pump_start_ts=0, pump_detected_ts=0, pump_peak_ts=0,
        pump_score=90.0, pump_class="SPOT_ORGANIC", pump_class_confidence="LOW", data_confidence="HIGH",
        volume_zscore=0.0, oi_zscore=np.nan, funding_percentile=np.nan, short_liq_zscore=np.nan, btc_return_concurrent=0.0,
        news_flag=False, final_state="CLOSED", exhaustion_score_peak=50.0, subsequent_return_15m=0.0, subsequent_return_1h=0.0,
        subsequent_return_4h=0.0, subsequent_return_24h=0.0, max_adverse_after_detection=0.0, max_favorable_after_detection=0.0,
        attempt_count=attempt_count, attempt_climax_local_idxs=list(range(attempt_count)),
        attempt_structure_break_local_idxs=structure_breaks, attempt_exhaustion_score_peaks=score_peaks,
        attempt_invalidation_reason_codes=reasons,
    )


def test_compare_attempts_one_row_per_attempt():
    events = [
        _fake_event("e1", 1, [40.0], [None]),
        _fake_event("e2", 2, [30.0, 55.0], ["new_pump_high", None]),
    ]
    df = ee.compare_attempts(events)
    assert len(df) == 3
    e2_rows = df[df["event_id"] == "e2"]
    assert list(e2_rows["attempt_number"]) == [1, 2]
    assert e2_rows.iloc[0]["invalidation_reason_code"] == "new_pump_high"
    assert bool(e2_rows.iloc[0]["is_final_attempt"]) is False
    assert bool(e2_rows.iloc[1]["is_final_attempt"]) is True


def test_summarize_attempts_groups_by_attempt_number():
    events = [_fake_event("e1", 1, [40.0], [None]), _fake_event("e2", 2, [30.0, 55.0], ["new_pump_high", None])]
    attempts_df = ee.compare_attempts(events)
    summary = ee.summarize_attempts(attempts_df)
    assert set(summary["attempt_number"]) == {1, 2}
    attempt1 = summary[summary["attempt_number"] == 1].iloc[0]
    assert attempt1["n"] == 2
    assert attempt1["mean_exhaustion_score_peak"] == pytest.approx((40.0 + 30.0) / 2)


def test_summarize_attempts_empty():
    result = ee.summarize_attempts(pd.DataFrame())
    assert list(result.columns) == ["attempt_number", "n", "mean_exhaustion_score_peak", "pct_had_structure_break", "pct_retried_further"]
    assert result.empty


def test_continuation_risk_after_failed_reclaim_upside_wins():
    n = 30
    high = [100.0] * n
    low = [100.0] * n
    close = [100.0] * n
    high[5], low[5], close[5] = 111.0, 100.0, 110.0  # +10% up move at bar 5
    ohlcv = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    results = ee.continuation_risk_after_failed_reclaim(ohlcv, anchor_idx=0, stop_distance_pct=0.02, bar_interval_ms=3_600_000, horizon_hours=20)
    key = "up10%_before_down1R"
    assert results[key]["outcome"] == "adverse"  # continuation-up hit first in first_passage_outcome's framing


def test_summarize_continuation_risk_aggregates():
    per_event = [
        {"thresh_a": {"outcome": "adverse", "bars": 1, "right_censored": False}},
        {"thresh_a": {"outcome": "profit", "bars": 2, "right_censored": False}},
    ]
    summary = ee.summarize_continuation_risk(per_event)
    row = summary[summary["threshold"] == "thresh_a"].iloc[0]
    assert row["n"] == 2
    assert row["p_continuation_first"] == pytest.approx(0.5)
    assert row["p_downside_first"] == pytest.approx(0.5)


def test_summarize_continuation_risk_empty():
    result = ee.summarize_continuation_risk([])
    assert result.empty
