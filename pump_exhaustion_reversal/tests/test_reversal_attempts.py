"""Stage 2A: retry-eligibility logic (_is_retryable_invalidation) and the
state machine's new attempt_invalidated transition (re-arm instead of
terminate)."""

from __future__ import annotations

from pump_exhaustion_reversal.backtest.simulator import _is_retryable_invalidation
from pump_exhaustion_reversal.config import AppConfig
from pump_exhaustion_reversal.strategy.state_machine import (
    PumpReversalEvent,
    PumpReversalState,
    StepContext,
    step_event,
)


def test_default_config_retries_only_new_pump_high():
    cfg = AppConfig()
    assert _is_retryable_invalidation("new_pump_high", cfg) is True
    assert _is_retryable_invalidation("bullish_hl_reformation", cfg) is False
    assert _is_retryable_invalidation("strong_reclaim", cfg) is False


def test_unknown_reason_code_is_never_retryable():
    cfg = AppConfig()
    assert _is_retryable_invalidation("something_else", cfg) is False


def test_reversal_attempts_disabled_blocks_all_retries():
    cfg = AppConfig(reversal_attempts={"enabled": False})
    assert _is_retryable_invalidation("new_pump_high", cfg) is False


def test_all_three_reasons_retryable_when_explicitly_enabled():
    cfg = AppConfig(
        reversal_attempts={
            "retry_on_new_pump_high": True,
            "retry_on_bullish_hl_reformation": True,
            "retry_on_strong_reclaim": True,
        }
    )
    assert _is_retryable_invalidation("new_pump_high", cfg) is True
    assert _is_retryable_invalidation("bullish_hl_reformation", cfg) is True
    assert _is_retryable_invalidation("strong_reclaim", cfg) is True


def _armed_event() -> PumpReversalEvent:
    event = PumpReversalEvent(symbol="TESTUSDT", exchange="bybit")
    step_event(event, StepContext(ts=0, pump_detected=True, pump_class="SPOT_ORGANIC"))
    step_event(event, StepContext(ts=1, exhaustion_state="ARMED_SHORT", no_trade_reasons=[]))
    assert event.state == PumpReversalState.ARMED_SHORT
    assert event.attempt_count == 1
    return event


def test_attempt_invalidated_reroutes_to_watching_not_invalidated():
    event = _armed_event()
    step_event(event, StepContext(ts=2, attempt_invalidated=True, invalidation_reason="price made a new pump high..."))
    assert event.state == PumpReversalState.WATCHING
    assert event.attempt_count == 2
    assert event.history[-1].reason == "price made a new pump high..."


def test_attempt_invalidated_can_fire_again_for_a_third_attempt():
    event = _armed_event()
    step_event(event, StepContext(ts=2, attempt_invalidated=True, invalidation_reason="retry 1"))
    step_event(event, StepContext(ts=3, exhaustion_state="ARMED_SHORT", no_trade_reasons=[]))
    step_event(event, StepContext(ts=3, exhaustion_state="ARMED_SHORT", no_trade_reasons=[]))
    assert event.state == PumpReversalState.ARMED_SHORT
    step_event(event, StepContext(ts=4, attempt_invalidated=True, invalidation_reason="retry 2"))
    assert event.state == PumpReversalState.WATCHING
    assert event.attempt_count == 3


def test_invalidated_wins_over_attempt_invalidated_when_both_set():
    """Defensive contract: real simulator.py code sets exactly one of these,
    never both - but if it somehow did, terminal invalidation must win as
    the safer default, not a silent retry."""
    event = _armed_event()
    step_event(event, StepContext(ts=2, invalidated=True, attempt_invalidated=True, invalidation_reason="both set"))
    assert event.state == PumpReversalState.INVALIDATED
    assert event.attempt_count == 1  # did not increment


def test_plain_invalidated_still_terminates_as_before():
    event = _armed_event()
    step_event(event, StepContext(ts=2, invalidated=True, invalidation_reason="reached event horizon without a valid trigger"))
    assert event.state == PumpReversalState.INVALIDATED
    assert event.attempt_count == 1
