"""Time stop + trailing-exit decisions (spec sections 17-18). Pure functions
of trade state, reused identically by the backtest simulator and (Stage 2)
live position monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TimeStopConfig


@dataclass
class TradeState:
    entry_price: float
    entry_ts: int
    stop_price: float
    initial_risk_per_unit: float  # |entry_price - initial_stop_price|, fixed at entry
    mode: str  # "aggressive" | "standard" | "conservative"
    is_short: bool = True
    tp1_hit: bool = False


def current_r_multiple(trade: TradeState, current_price: float) -> float:
    if not trade.initial_risk_per_unit:
        return 0.0
    move = (trade.entry_price - current_price) if trade.is_short else (current_price - trade.entry_price)
    return move / trade.initial_risk_per_unit


_TIME_STOP_MINUTES_FIELD_BY_MODE = {
    "aggressive": "aggressive_minutes",
    "standard": "standard_minutes",
    "conservative": "conservative_minutes",
}


def check_time_stop(
    trade: TradeState,
    current_ts: int,
    current_price: float,
    oi_reaccelerated: bool,
    cfg: TimeStopConfig,
) -> tuple[bool, str | None]:
    """spec section 18: after the mode's time window, exit/reduce if price
    hasn't moved +exit_if_no_progress_r in favor, or if OI is reaccelerating
    (a sign the pump's original buyers/shorts are re-engaging against us).
    """
    if not cfg.enabled:
        return False, None
    field = _TIME_STOP_MINUTES_FIELD_BY_MODE.get(trade.mode, "standard_minutes")
    window_minutes = getattr(cfg, field)
    elapsed_minutes = (current_ts - trade.entry_ts) / 60_000
    if elapsed_minutes < window_minutes:
        return False, None

    r = current_r_multiple(trade, current_price)
    if r < cfg.exit_if_no_progress_r:
        return True, f"no progress: {r:.2f}R < {cfg.exit_if_no_progress_r}R after {window_minutes}min"
    if cfg.exit_if_oi_reaccelerates and oi_reaccelerated:
        return True, "OI reaccelerating against the position"
    return False, None
