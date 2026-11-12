"""Event-level state machine (spec section 20). Every transition is
persisted with a timestamp + feature snapshot - needed for backtest
audit/debugging and for a future Stage 2 live scanner alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PumpReversalState(str, Enum):
    IDLE = "IDLE"
    PUMP_DETECTED = "PUMP_DETECTED"
    WATCHING = "WATCHING"
    EXHAUSTION_FORMING = "EXHAUSTION_FORMING"
    ARMED_SHORT = "ARMED_SHORT"
    SHORT_TRIGGERED = "SHORT_TRIGGERED"
    MANAGING = "MANAGING"
    CLOSED = "CLOSED"
    INVALIDATED = "INVALIDATED"
    LOCKED = "LOCKED"


ACTIVE_STATES = frozenset(
    {
        PumpReversalState.PUMP_DETECTED,
        PumpReversalState.WATCHING,
        PumpReversalState.EXHAUSTION_FORMING,
        PumpReversalState.ARMED_SHORT,
        PumpReversalState.SHORT_TRIGGERED,
        PumpReversalState.MANAGING,
    }
)
TERMINAL_STATES = frozenset({PumpReversalState.CLOSED, PumpReversalState.INVALIDATED, PumpReversalState.LOCKED})
_WATCH_PHASE_STATES = frozenset(
    {PumpReversalState.WATCHING, PumpReversalState.EXHAUSTION_FORMING, PumpReversalState.ARMED_SHORT}
)


@dataclass
class Transition:
    ts: int
    from_state: PumpReversalState
    to_state: PumpReversalState
    reason: str
    feature_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class PumpReversalEvent:
    """One pump event's lifecycle for one symbol. A new pump after a prior
    one closes/invalidates is a separate PumpReversalEvent - this class does
    not itself track symbol-lock across events (see the backtest simulator).
    """

    symbol: str
    exchange: str
    state: PumpReversalState = PumpReversalState.IDLE
    history: list[Transition] = field(default_factory=list)
    pump_class: str | None = None
    entry_mode: str | None = None
    # Stage 2A: how many exhaustion-watch cycles this event has gone through.
    # Starts at 1 (the initial watch); a retryable dynamic invalidation (see
    # StepContext.attempt_invalidated) increments this and loops back to
    # WATCHING instead of terminating - see backtest/simulator.py, which owns
    # the retry-vs-terminal decision (this class stays cfg-agnostic).
    attempt_count: int = 1

    def transition_to(self, new_state: PumpReversalState, ts: int, reason: str, feature_snapshot: dict | None = None) -> None:
        self.history.append(Transition(ts, self.state, new_state, reason, feature_snapshot or {}))
        self.state = new_state

    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass
class StepContext:
    ts: int
    pump_detected: bool = False
    pump_class: str | None = None
    exhaustion_state: str = "WATCHING"  # exhaustion_detector.classify_exhaustion_state() output
    no_trade_reasons: list[str] = field(default_factory=list)
    entry_signal_valid: bool = False
    entry_mode: str | None = None
    invalidated: bool = False
    invalidation_reason: str | None = None
    # Stage 2A: a dynamic post-break invalidation the caller has decided is
    # retryable (attempt budget not yet exhausted) - loops the event back to
    # WATCHING instead of terminating. Mutually exclusive with `invalidated`
    # in practice (the caller sets one or the other, never both); if both are
    # somehow set, `invalidated` wins (checked first below) as the safer,
    # more conservative default.
    attempt_invalidated: bool = False
    exit_decision: str | None = None  # None | "TP" | "SL" | "TIME_STOP" | "MANUAL"
    symbol_locked: bool = False
    feature_snapshot: dict = field(default_factory=dict)


def step_event(event: PumpReversalEvent, ctx: StepContext) -> None:
    """Advances `event` by exactly one bar's worth of decisions, mutating it
    in place. Mirrors spec section 20/26's transition table + pseudocode -
    PUMP_DETECTED->WATCHING and SHORT_TRIGGERED->MANAGING each collapse into
    the same step as their trigger, matching spec's own pseudocode.
    """
    if ctx.symbol_locked and event.state not in (PumpReversalState.SHORT_TRIGGERED, PumpReversalState.MANAGING):
        if event.state != PumpReversalState.LOCKED:
            event.transition_to(PumpReversalState.LOCKED, ctx.ts, "symbol locked after max failed attempts", ctx.feature_snapshot)
        return

    if event.state == PumpReversalState.IDLE:
        if ctx.pump_detected:
            event.transition_to(PumpReversalState.PUMP_DETECTED, ctx.ts, "pump_score >= min_pump_score", ctx.feature_snapshot)
            event.pump_class = ctx.pump_class
            event.transition_to(PumpReversalState.WATCHING, ctx.ts, "pump classified, snapshot saved", ctx.feature_snapshot)
        return

    if event.state in _WATCH_PHASE_STATES:
        if ctx.invalidated:
            event.transition_to(
                PumpReversalState.INVALIDATED, ctx.ts, ctx.invalidation_reason or "setup invalidated", ctx.feature_snapshot
            )
            return

        if ctx.attempt_invalidated:
            event.attempt_count += 1
            event.transition_to(
                PumpReversalState.WATCHING, ctx.ts,
                ctx.invalidation_reason or f"reversal attempt invalidated, retrying (attempt {event.attempt_count})",
                ctx.feature_snapshot,
            )
            return

        if event.state == PumpReversalState.WATCHING and ctx.exhaustion_state in ("EXHAUSTION_FORMING", "ARMED_SHORT"):
            event.transition_to(PumpReversalState.EXHAUSTION_FORMING, ctx.ts, "exhaustion score >= watch threshold", ctx.feature_snapshot)

        if event.state == PumpReversalState.EXHAUSTION_FORMING and ctx.exhaustion_state == "ARMED_SHORT" and not ctx.no_trade_reasons:
            event.transition_to(
                PumpReversalState.ARMED_SHORT, ctx.ts, "exhaustion score >= armed threshold, no-trade filters clear", ctx.feature_snapshot
            )

        if event.state == PumpReversalState.ARMED_SHORT and ctx.entry_signal_valid:
            event.transition_to(PumpReversalState.SHORT_TRIGGERED, ctx.ts, f"entry mode {ctx.entry_mode} hard gates passed", ctx.feature_snapshot)
            event.entry_mode = ctx.entry_mode
            event.transition_to(PumpReversalState.MANAGING, ctx.ts, "simulated fill", ctx.feature_snapshot)
        return

    if event.state == PumpReversalState.MANAGING:
        if ctx.exit_decision:
            event.transition_to(PumpReversalState.CLOSED, ctx.ts, f"exit: {ctx.exit_decision}", ctx.feature_snapshot)
        return
