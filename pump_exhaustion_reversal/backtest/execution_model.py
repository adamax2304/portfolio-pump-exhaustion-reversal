"""Fee/slippage/funding/participation-cap simulation (spec section 22.4).
Pure math - the simulator decides WHEN a fill happens (next-bar-open,
lookahead-safe); this module only decides the PRICE/cost mechanics.
"""

from __future__ import annotations

import numpy as np

from ..config import ExecutionModelConfig


def compute_slippage_bps(cfg: ExecutionModelConfig, high_volatility: bool) -> float:
    return cfg.default_slippage_bps * (cfg.high_vol_slippage_multiplier if high_volatility else 1.0)


def apply_slippage(quoted_price: float, slippage_bps: float, side: str) -> float:
    """side='sell' (opening a short / closing a long) fills LOWER than quoted;
    side='buy' (closing a short / opening a long) fills HIGHER - slippage
    always works against the trader, in either direction."""
    if side == "sell":
        return quoted_price * (1 - slippage_bps / 10_000)
    if side == "buy":
        return quoted_price * (1 + slippage_bps / 10_000)
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def compute_fee(notional: float, cfg: ExecutionModelConfig, is_maker: bool = False) -> float:
    bps = cfg.maker_fee_bps if is_maker else cfg.taker_fee_bps
    return notional * bps / 10_000


def compute_funding_cost(notional: float, funding_rate: float, is_short: bool) -> float:
    """Positive funding_rate: longs pay shorts. A short RECEIVES when
    funding_rate>0 (negative cost = a credit) and pays when it's negative."""
    sign = -1 if is_short else 1
    return notional * funding_rate * sign


def max_fillable_notional(bar_quote_volume: float | None, cfg: ExecutionModelConfig) -> float:
    if bar_quote_volume is None or np.isnan(bar_quote_volume):
        return float("inf")  # unknown volume -> don't fabricate a constraint; liquidity gates live upstream
    return bar_quote_volume * cfg.max_volume_participation_pct / 100.0


def apply_participation_cap(desired_notional: float, bar_quote_volume: float | None, cfg: ExecutionModelConfig) -> dict:
    cap = max_fillable_notional(bar_quote_volume, cfg)
    filled = min(desired_notional, cap)
    return {
        "filled_notional": filled,
        "requested_notional": desired_notional,
        "fill_ratio": (filled / desired_notional) if desired_notional > 0 else 1.0,
        "capped": filled < desired_notional,
    }


def simulate_entry_fill(next_bar_open: float, cfg: ExecutionModelConfig, high_volatility: bool) -> dict:
    """entry_type='next_bar_open_or_limit_retest' - Stage 1 implements the
    next-bar-open half (always lookahead-safe: strictly after the signal
    bar's close); limit-retest fills are a documented simplification, not
    modeled here (see README limitations)."""
    slippage_bps = compute_slippage_bps(cfg, high_volatility)
    fill_price = apply_slippage(next_bar_open, slippage_bps, side="sell")  # opening a short = selling
    return {"fill_price": fill_price, "slippage_bps": slippage_bps}


def simulate_stop_fill(stop_price: float, cfg: ExecutionModelConfig) -> dict:
    fill_price = apply_slippage(stop_price, cfg.stop_slippage_bps, side="buy")  # closing a short = buying
    return {"fill_price": fill_price, "slippage_bps": cfg.stop_slippage_bps}


def simulate_tp_fill(tp_price: float, cfg: ExecutionModelConfig, high_volatility: bool) -> dict:
    slippage_bps = compute_slippage_bps(cfg, high_volatility)
    fill_price = apply_slippage(tp_price, slippage_bps, side="buy")
    return {"fill_price": fill_price, "slippage_bps": slippage_bps}
