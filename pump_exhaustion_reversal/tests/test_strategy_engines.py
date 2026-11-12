from __future__ import annotations

import pytest

from pump_exhaustion_reversal.config import AppConfig, LeverageConfig, StopsConfig, TakeProfitConfig, TimeStopConfig
from pump_exhaustion_reversal.strategy import entry_engine, exit_engine, risk_engine, state_machine
from pump_exhaustion_reversal.strategy.catalyst_filter import CatalystStatus

# ---- risk_engine.py --------------------------------------------------


def test_compute_structural_stop_normal_case():
    cfg = StopsConfig(atr_buffer_multiplier=0.25, min_stop_bps=40, max_stop_bps=1200)
    result = risk_engine.compute_structural_stop(entry_price=100.0, stop_reference_price=99.0, atr_value=1.0, cfg=cfg, is_short=True)
    # raw stop = 99 + 0.25*1 = 99.25, distance = (99.25-100)/100*10000 = 75bps, within [40,1200]
    assert result.stop_price == pytest.approx(99.25)
    assert result.stop_distance_bps == pytest.approx(75.0)
    assert not result.widened_to_min
    assert not result.exceeds_max_stop


def test_compute_structural_stop_widens_to_min():
    cfg = StopsConfig(atr_buffer_multiplier=0.0, min_stop_bps=40, max_stop_bps=1200)
    # raw stop = reference exactly (buffer=0), distance only 5bps -> too tight, must widen to 40bps
    result = risk_engine.compute_structural_stop(entry_price=100.0, stop_reference_price=100.05, atr_value=1.0, cfg=cfg, is_short=True)
    assert result.widened_to_min
    assert result.stop_distance_bps == pytest.approx(40.0)
    assert result.stop_price == pytest.approx(100.40)


def test_compute_structural_stop_flags_but_does_not_clamp_when_too_wide():
    cfg = StopsConfig(atr_buffer_multiplier=1.0, min_stop_bps=40, max_stop_bps=100)
    result = risk_engine.compute_structural_stop(entry_price=100.0, stop_reference_price=95.0, atr_value=1.0, cfg=cfg, is_short=True)
    # raw stop = 95+1=96, distance=400bps > max(100bps)
    assert result.exceeds_max_stop
    assert result.stop_price == pytest.approx(96.0)  # NOT silently clamped to the max


def test_compute_position_size_matches_manual_formula():
    cfg = LeverageConfig(default_max_leverage=10)
    result = risk_engine.compute_position_size(equity=10_000, entry_price=100.0, stop_price=99.0, risk_per_trade_pct=0.20, cfg_leverage=cfg)
    # risk_dollars = 10000*0.002 = 20; stop_distance_pct=0.01; notional=20/0.01=2000
    assert result.valid
    assert result.risk_dollars == pytest.approx(20.0)
    assert result.notional == pytest.approx(2000.0)
    assert result.quantity == pytest.approx(20.0)
    assert not result.capped_by_leverage


def test_compute_position_size_caps_at_max_leverage():
    cfg = LeverageConfig(default_max_leverage=1)  # very low cap
    result = risk_engine.compute_position_size(equity=10_000, entry_price=100.0, stop_price=99.9, risk_per_trade_pct=0.20, cfg_leverage=cfg)
    # uncapped notional would be huge (tiny stop distance) -> must cap at equity*1
    assert result.capped_by_leverage
    assert result.notional == pytest.approx(10_000)
    assert result.leverage_needed == pytest.approx(1.0)


def test_compute_position_size_zero_stop_distance_is_invalid():
    result = risk_engine.compute_position_size(equity=10_000, entry_price=100.0, stop_price=100.0, risk_per_trade_pct=0.2, cfg_leverage=LeverageConfig())
    assert not result.valid


def test_take_profit_plan_direction_for_short_vs_long():
    cfg = TakeProfitConfig(tp1_r_multiple=1.0)
    short_plan = risk_engine.compute_take_profit_plan(entry_price=100.0, stop_price=105.0, cfg=cfg, is_short=True)
    assert short_plan.tp1_price == pytest.approx(95.0)  # 1R below entry for a short
    long_plan = risk_engine.compute_take_profit_plan(entry_price=100.0, stop_price=95.0, cfg=cfg, is_short=False)
    assert long_plan.tp1_price == pytest.approx(105.0)


def test_resolve_symbolic_tp_target_picks_nearest_for_short_and_falls_back():
    target = risk_engine.resolve_symbolic_tp_target(
        "pump_avwap_or_breakout_area", {"pump_avwap": 90.0, "breakout_level": 92.0}, entry_price=100.0, is_short=True
    )
    assert target == pytest.approx(90.0)  # min() for a short target
    fallback = risk_engine.resolve_symbolic_tp_target("pump_avwap_or_breakout_area", {}, entry_price=100.0, is_short=True)
    assert fallback == pytest.approx(100.0)


def test_stop_after_tp1_moves_to_breakeven():
    assert risk_engine.stop_after_tp1(entry_price=100.0, current_stop_price=105.0, policy="breakeven_or_reduce_risk") == pytest.approx(100.0)
    assert risk_engine.stop_after_tp1(entry_price=100.0, current_stop_price=105.0, policy="something_unrecognized") == pytest.approx(105.0)


def test_trail_stop_never_loosens_for_short():
    assert risk_engine.trail_stop(current_stop_price=100.0, structural_reference_price=98.0, is_short=True) == pytest.approx(98.0)
    assert risk_engine.trail_stop(current_stop_price=100.0, structural_reference_price=102.0, is_short=True) == pytest.approx(100.0)


# ---- exit_engine.py --------------------------------------------------


def test_current_r_multiple_short():
    trade = exit_engine.TradeState(entry_price=100.0, entry_ts=0, stop_price=105.0, initial_risk_per_unit=5.0, mode="standard", is_short=True)
    assert exit_engine.current_r_multiple(trade, current_price=95.0) == pytest.approx(1.0)
    assert exit_engine.current_r_multiple(trade, current_price=102.5) == pytest.approx(-0.5)


def test_check_time_stop_no_progress_triggers_after_window():
    trade = exit_engine.TradeState(entry_price=100.0, entry_ts=0, stop_price=105.0, initial_risk_per_unit=5.0, mode="standard")
    cfg = TimeStopConfig(standard_minutes=60, exit_if_no_progress_r=0.25)
    # before window elapses -> no decision yet even with zero progress
    fired, reason = exit_engine.check_time_stop(trade, current_ts=30 * 60_000, current_price=100.0, oi_reaccelerated=False, cfg=cfg)
    assert not fired
    # after window, price barely moved -> exit
    fired, reason = exit_engine.check_time_stop(trade, current_ts=61 * 60_000, current_price=99.0, oi_reaccelerated=False, cfg=cfg)
    assert fired
    assert "no progress" in reason


def test_check_time_stop_sufficient_progress_does_not_fire():
    trade = exit_engine.TradeState(entry_price=100.0, entry_ts=0, stop_price=105.0, initial_risk_per_unit=5.0, mode="standard")
    cfg = TimeStopConfig(standard_minutes=60, exit_if_no_progress_r=0.25)
    fired, _ = exit_engine.check_time_stop(trade, current_ts=61 * 60_000, current_price=95.0, oi_reaccelerated=False, cfg=cfg)
    assert not fired


def test_check_time_stop_oi_reacceleration_forces_exit_even_with_progress():
    trade = exit_engine.TradeState(entry_price=100.0, entry_ts=0, stop_price=105.0, initial_risk_per_unit=5.0, mode="standard")
    cfg = TimeStopConfig(standard_minutes=60, exit_if_no_progress_r=0.25, exit_if_oi_reaccelerates=True)
    fired, reason = exit_engine.check_time_stop(trade, current_ts=61 * 60_000, current_price=95.0, oi_reaccelerated=True, cfg=cfg)
    assert fired
    assert "OI" in reason


# ---- entry_engine.py --------------------------------------------------


def _catalyst_clear():
    return CatalystStatus(True, True, "yes", False, "KNOWN", False, [])


def test_evaluate_entry_fires_when_all_signals_present():
    cfg = AppConfig()
    signals = {
        "structure_break": True, "failed_reclaim": True, "lower_high": True,
        "cvd_divergence": True, "sweep_rejection": True, "new_local_low": True, "avwap_loss": True,
    }
    market = entry_engine.MarketConditionInputs()
    result = entry_engine.evaluate_entry(
        pump_score=80.0, exhaustion_score=80.0, signals=signals, catalyst_status=_catalyst_clear(), market=market, cfg=cfg
    )
    assert result.valid
    assert result.mode == "standard"  # aggressive is disabled by default (spec) -> falls through to standard


def test_evaluate_entry_falls_back_to_standard_when_aggressive_disabled():
    cfg = AppConfig()
    assert cfg.entry_modes.aggressive.enabled is False  # spec default
    signals = {
        "structure_break": True, "failed_reclaim": True, "lower_high": True,
        "cvd_divergence": True, "sweep_rejection": True, "new_local_low": True, "avwap_loss": True,
    }
    result = entry_engine.evaluate_entry(
        pump_score=80.0, exhaustion_score=80.0, signals=signals,
        catalyst_status=_catalyst_clear(), market=entry_engine.MarketConditionInputs(), cfg=cfg,
    )
    assert result.valid
    assert result.mode == "standard"


def test_evaluate_entry_score_alone_never_bypasses_missing_structure_signals():
    cfg = AppConfig()
    signals = {k: False for k in ("structure_break", "failed_reclaim", "lower_high", "cvd_divergence", "sweep_rejection", "new_local_low", "avwap_loss")}
    result = entry_engine.evaluate_entry(
        pump_score=100.0, exhaustion_score=100.0, signals=signals,
        catalyst_status=_catalyst_clear(), market=entry_engine.MarketConditionInputs(), cfg=cfg,
    )
    assert not result.valid


def test_evaluate_entry_blocked_by_major_positive_news():
    cfg = AppConfig()
    signals = {"structure_break": True, "failed_reclaim": True, "lower_high": True, "cvd_divergence": True, "sweep_rejection": True, "new_local_low": True, "avwap_loss": True}
    blocked_catalyst = CatalystStatus(False, False, "no", True, "KNOWN", False, ["positive catalyst"])
    result = entry_engine.evaluate_entry(
        pump_score=90.0, exhaustion_score=90.0, signals=signals,
        catalyst_status=blocked_catalyst, market=entry_engine.MarketConditionInputs(), cfg=cfg,
    )
    assert not result.valid
    assert any("news" in r for r in result.reasons)


def test_evaluate_entry_blocked_by_no_trade_filter_price_still_hh_hl():
    cfg = AppConfig()
    signals = {"structure_break": True, "failed_reclaim": True, "lower_high": True, "cvd_divergence": True, "sweep_rejection": True, "new_local_low": True, "avwap_loss": True}
    market = entry_engine.MarketConditionInputs(price_still_hh_hl=True)
    result = entry_engine.evaluate_entry(
        pump_score=90.0, exhaustion_score=90.0, signals=signals,
        catalyst_status=_catalyst_clear(), market=market, cfg=cfg,
    )
    assert not result.valid
    assert "price_still_hh_hl" in result.reasons


# ---- state_machine.py --------------------------------------------------


def test_state_machine_full_happy_path():
    event = state_machine.PumpReversalEvent(symbol="BTCUSDT", exchange="bybit")

    state_machine.step_event(event, state_machine.StepContext(ts=1, pump_detected=True, pump_class="SHORT_SQUEEZE"))
    assert event.state == state_machine.PumpReversalState.WATCHING
    assert event.pump_class == "SHORT_SQUEEZE"

    state_machine.step_event(event, state_machine.StepContext(ts=2, exhaustion_state="EXHAUSTION_FORMING"))
    assert event.state == state_machine.PumpReversalState.EXHAUSTION_FORMING

    state_machine.step_event(event, state_machine.StepContext(ts=3, exhaustion_state="ARMED_SHORT"))
    assert event.state == state_machine.PumpReversalState.ARMED_SHORT

    state_machine.step_event(event, state_machine.StepContext(ts=4, exhaustion_state="ARMED_SHORT", entry_signal_valid=True, entry_mode="standard"))
    assert event.state == state_machine.PumpReversalState.MANAGING
    assert event.entry_mode == "standard"

    state_machine.step_event(event, state_machine.StepContext(ts=5, exit_decision="TP"))
    assert event.state == state_machine.PumpReversalState.CLOSED

    # IDLE->PUMP_DETECTED->WATCHING->EXHAUSTION_FORMING->ARMED_SHORT->SHORT_TRIGGERED->MANAGING->CLOSED
    assert len(event.history) == 7
    visited = [t.to_state for t in event.history]
    assert visited == [
        state_machine.PumpReversalState.PUMP_DETECTED,
        state_machine.PumpReversalState.WATCHING,
        state_machine.PumpReversalState.EXHAUSTION_FORMING,
        state_machine.PumpReversalState.ARMED_SHORT,
        state_machine.PumpReversalState.SHORT_TRIGGERED,
        state_machine.PumpReversalState.MANAGING,
        state_machine.PumpReversalState.CLOSED,
    ]


def test_state_machine_invalidation_from_watching():
    event = state_machine.PumpReversalEvent(symbol="XUSDT", exchange="binance")
    state_machine.step_event(event, state_machine.StepContext(ts=1, pump_detected=True))
    state_machine.step_event(event, state_machine.StepContext(ts=2, invalidated=True, invalidation_reason="price made new HH, structure never broke"))
    assert event.state == state_machine.PumpReversalState.INVALIDATED


def test_state_machine_locks_regardless_of_other_signals():
    event = state_machine.PumpReversalEvent(symbol="YUSDT", exchange="bybit")
    state_machine.step_event(event, state_machine.StepContext(ts=1, pump_detected=True))
    state_machine.step_event(event, state_machine.StepContext(ts=2, symbol_locked=True, exhaustion_state="ARMED_SHORT"))
    assert event.state == state_machine.PumpReversalState.LOCKED


def test_state_machine_no_trade_filters_block_forming_to_armed():
    event = state_machine.PumpReversalEvent(symbol="ZUSDT", exchange="bybit")
    state_machine.step_event(event, state_machine.StepContext(ts=1, pump_detected=True))
    state_machine.step_event(event, state_machine.StepContext(ts=2, exhaustion_state="EXHAUSTION_FORMING"))
    state_machine.step_event(event, state_machine.StepContext(ts=3, exhaustion_state="ARMED_SHORT", no_trade_reasons=["price_still_hh_hl"]))
    assert event.state == state_machine.PumpReversalState.EXHAUSTION_FORMING  # blocked from advancing to ARMED_SHORT
