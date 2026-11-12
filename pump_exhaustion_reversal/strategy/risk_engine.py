"""Structural stops + position sizing + TP ladder (spec sections 15-17).

Pure math, no I/O - reused identically by the backtest simulator (Stage 1)
and a future live risk engine (Stage 2). Risk is defined by position size
and stop distance, never by leverage (spec section 16): leverage here is
only ever a mechanical consequence of notional/equity, capped, never a
risk lever in its own right.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LeverageConfig, RiskConfig, StopsConfig, TakeProfitConfig


@dataclass
class StopResult:
    stop_price: float
    stop_distance_bps: float
    widened_to_min: bool
    exceeds_max_stop: bool  # caller should reduce size or skip the trade, not silently move the stop


def compute_structural_stop(
    entry_price: float,
    stop_reference_price: float,
    atr_value: float,
    cfg: StopsConfig,
    is_short: bool = True,
) -> StopResult:
    """spec section 15: stop = structural reference level +/- atr_buffer_multiplier*ATR,
    with min/max_stop_bps as a floor (widen a too-tight stop, which would
    guarantee a noise stop-out) and a ceiling flag (a too-wide stop is NOT
    silently tightened to an unjustified level - spec: "reduce position; or
    skip trade" - that decision belongs to the caller/position sizing step).
    """
    buffer = cfg.atr_buffer_multiplier * atr_value
    raw_stop = stop_reference_price + buffer if is_short else stop_reference_price - buffer
    distance_bps = abs(raw_stop - entry_price) / entry_price * 10_000
    sign = 1 if is_short else -1

    if distance_bps < cfg.min_stop_bps:
        final_bps = float(cfg.min_stop_bps)
        stop_price = entry_price * (1 + sign * final_bps / 10_000)
        widened = True
    else:
        stop_price = raw_stop
        final_bps = distance_bps
        widened = False

    return StopResult(
        stop_price=stop_price,
        stop_distance_bps=final_bps,
        widened_to_min=widened,
        exceeds_max_stop=final_bps > cfg.max_stop_bps,
    )


@dataclass
class PositionSizeResult:
    valid: bool
    notional: float = 0.0
    quantity: float = 0.0
    risk_dollars: float = 0.0
    stop_distance_pct: float = 0.0
    leverage_needed: float = 0.0
    capped_by_leverage: bool = False
    reason: str | None = None


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade_pct: float,
    cfg_leverage: LeverageConfig,
) -> PositionSizeResult:
    """spec section 16: risk_dollars = equity * risk_pct; notional =
    risk_dollars / stop_distance_pct. risk_per_trade_pct is a PERCENT (e.g.
    0.20 means 0.20% of equity, matching config.risk.initial_risk_per_trade_pct's
    scale) - not a fraction.
    """
    stop_distance_pct = abs(stop_price - entry_price) / entry_price if entry_price else 0.0
    if stop_distance_pct <= 0:
        return PositionSizeResult(valid=False, reason="zero or invalid stop distance")

    risk_dollars = equity * (risk_per_trade_pct / 100.0)
    notional = risk_dollars / stop_distance_pct
    leverage_needed = notional / equity if equity > 0 else float("inf")

    capped = False
    max_notional = equity * cfg_leverage.default_max_leverage
    if notional > max_notional:
        notional = max_notional
        capped = True
        leverage_needed = cfg_leverage.default_max_leverage

    return PositionSizeResult(
        valid=True,
        notional=notional,
        quantity=notional / entry_price,
        risk_dollars=risk_dollars,
        stop_distance_pct=stop_distance_pct,
        leverage_needed=min(leverage_needed, cfg_leverage.default_max_leverage),
        capped_by_leverage=capped,
    )


@dataclass
class TakeProfitPlan:
    tp1_price: float
    tp1_close_pct: float
    tp2_target_desc: str
    tp2_close_pct: float
    tp3_target_desc: str
    tp3_close_pct: float
    move_stop_after_tp1: str


def compute_take_profit_plan(entry_price: float, stop_price: float, cfg: TakeProfitConfig, is_short: bool = True) -> TakeProfitPlan:
    r_distance = abs(entry_price - stop_price)
    sign = -1 if is_short else 1
    tp1 = entry_price + sign * r_distance * cfg.tp1_r_multiple
    return TakeProfitPlan(
        tp1_price=tp1,
        tp1_close_pct=cfg.tp1_close_pct,
        tp2_target_desc=cfg.tp2_target,
        tp2_close_pct=cfg.tp2_close_pct,
        tp3_target_desc=cfg.tp3_target,
        tp3_close_pct=cfg.tp3_close_pct,
        move_stop_after_tp1=cfg.move_stop_after_tp1,
    )


def resolve_symbolic_tp_target(description: str, context: dict, entry_price: float, is_short: bool = True) -> float:
    """Resolves spec's symbolic TP2/TP3 descriptions ("pump_avwap_or_breakout_area",
    "0.5_pump_retracement_or_trailing_structure") using event context the
    backtest simulator/live scanner supplies (pump_avwap, breakout_level,
    pump_high, pump_low). Falls back to entry_price (no incremental target
    beyond TP1) when the needed context is missing - callers should read
    that as "trail structurally instead", per spec, not as a real target.
    """
    if description == "pump_avwap_or_breakout_area":
        candidates = [v for v in (context.get("pump_avwap"), context.get("breakout_level")) if v is not None]
        if not candidates:
            return entry_price
        return min(candidates) if is_short else max(candidates)
    if description == "0.5_pump_retracement_or_trailing_structure":
        pump_high, pump_low = context.get("pump_high"), context.get("pump_low")
        if pump_high is None or pump_low is None:
            return entry_price
        midpoint = pump_high - 0.5 * (pump_high - pump_low)
        return midpoint
    return entry_price


def stop_after_tp1(entry_price: float, current_stop_price: float, policy: str) -> float:
    """spec section 17: after TP1, move stop to breakeven (never worse for the position)."""
    if policy in ("breakeven_or_reduce_risk", "breakeven"):
        return entry_price
    return current_stop_price


def trail_stop(current_stop_price: float, structural_reference_price: float, is_short: bool = True) -> float:
    """Ratchets the stop toward the structural reference - never loosens it."""
    if is_short:
        return min(current_stop_price, structural_reference_price)
    return max(current_stop_price, structural_reference_price)
