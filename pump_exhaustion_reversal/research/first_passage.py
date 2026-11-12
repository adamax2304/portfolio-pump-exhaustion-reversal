"""Stage 2A Part 2: first-passage / triple-barrier labeling - replaces
research/forward_study.py's eventual-peak-drawdown target for predictive-
value work. That target answered "did the coin ever fall 20% from some
future peak" - true 77-92% of the time regardless of any exhaustion signal,
not a tradable question. This module asks the actual trading question: from
a given anchor bar, does price hit the SHORT thesis's profit barrier or its
adverse barrier first (or does neither happen within the horizon)?

Same-candle ambiguity (both barriers touchable within one bar) resolves
adverse-first - the same "stop wins ties" convention already used in
backtest/simulator.py's _simulate_trade.

Deliberately does not import anything from backtest/simulator.py or
strategy/* - this operates purely on already-saved EventRecord fields and
raw OHLCV, mirroring research/forward_study.py's own lookahead-safety
framing (see research/__init__.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import structure as struct_feat

# Spec-listed label families to test WITHOUT picking a "winner" in advance -
# each is (label, profit_r_multiple | None, profit_pct | None, adverse_r_multiple | None, adverse_pct | None, horizon_hours).
# R-based families use resolve_r_based_barriers; %-based families use fixed
# price-fraction barriers directly - both funnel into the same
# first_passage_outcome() barrier walk.
LABEL_FAMILIES: dict[str, dict] = {
    "fast_1h": {"kind": "r", "profit_r": 1.5, "adverse_r": 1.0, "horizon_hours": 1},
    "standard_4h": {"kind": "r", "profit_r": 2.0, "adverse_r": 1.0, "horizon_hours": 4},
    "swing_8h": {"kind": "r", "profit_r": 3.0, "adverse_r": 1.0, "horizon_hours": 8},
    "pct_test_4h": {"kind": "pct", "profit_pct": 0.10, "adverse_pct": 0.05, "horizon_hours": 4},
    "deep_reversal_24h": {"kind": "pct", "profit_pct": 0.20, "adverse_pct": 0.08, "horizon_hours": 24},
}

# Phase 2B's own barrier-family grid (sensitivity analysis, not optimization -
# report every one of these, never pick a "best" from the full sample).
PHASE_2B_BARRIER_FAMILIES: dict[str, dict] = {
    "1.5R_1R_1h": {"kind": "r", "profit_r": 1.5, "adverse_r": 1.0, "horizon_hours": 1},
    "2R_1R_2h": {"kind": "r", "profit_r": 2.0, "adverse_r": 1.0, "horizon_hours": 2},
    "2R_1R_4h": {"kind": "r", "profit_r": 2.0, "adverse_r": 1.0, "horizon_hours": 4},
    "2R_1R_8h": {"kind": "r", "profit_r": 2.0, "adverse_r": 1.0, "horizon_hours": 8},
    "3R_1R_4h": {"kind": "r", "profit_r": 3.0, "adverse_r": 1.0, "horizon_hours": 4},
    "3R_1R_8h": {"kind": "r", "profit_r": 3.0, "adverse_r": 1.0, "horizon_hours": 8},
}


def compute_atr_stop_distance_pct(ohlcv: pd.DataFrame, anchor_idx: int, atr_length: int) -> float:
    """A uniform, anchor-agnostic stop-distance basis (ATR-multiple, not a
    structural swing level) - deliberately simple so a first-passage
    comparison ACROSS different candidate anchor timestamps isolates the
    effect of WHEN the anchor fires, not a different stop-sizing rule at
    each one (a structural stop's reference level would itself depend on
    which anchor/attempt is active - see backtest/simulator.py's
    _PostPumpInputs - which would confound that comparison)."""
    if anchor_idx <= 0 or anchor_idx >= len(ohlcv):
        return np.nan
    atr_series = struct_feat.atr(ohlcv, atr_length).bfill()
    atr_value = atr_series.iloc[anchor_idx]
    if pd.isna(atr_value):
        atr_value = (ohlcv["high"] - ohlcv["low"]).iloc[: anchor_idx + 1].mean()
    anchor_price = float(ohlcv["close"].iloc[anchor_idx])
    if not anchor_price or pd.isna(atr_value):
        return np.nan
    return float(atr_value) / anchor_price


def resolve_r_based_barriers(stop_distance_pct: float, profit_r_multiple: float, adverse_r_multiple: float = 1.0) -> tuple[float, float]:
    """adverse_barrier_pct = adverse_r_multiple * stop_distance_pct (1R by
    definition when adverse_r_multiple=1.0); profit_barrier_pct =
    profit_r_multiple * stop_distance_pct."""
    return profit_r_multiple * stop_distance_pct, adverse_r_multiple * stop_distance_pct


def _empty_outcome(outcome: str, bars: int, right_censored: bool) -> dict:
    return {
        "outcome": outcome, "bars": bars, "right_censored": right_censored,
        "mae_pct": np.nan, "mfe_pct": np.nan, "return_pct_at_exit": np.nan,
    }


def first_passage_outcome(
    ohlcv: pd.DataFrame, anchor_idx: int, profit_barrier_pct: float, adverse_barrier_pct: float, horizon_bars: int
) -> dict:
    """SHORT thesis: from anchor_idx's close, walk forward bar by bar. Profit
    barrier sits BELOW the anchor close by profit_barrier_pct; adverse
    barrier sits ABOVE it by adverse_barrier_pct. Returns which was touched
    first ("profit"/"adverse"/"timeout"), how many bars it took, and
    right_censored (True if the bounded horizon was cut short by running out
    of actual data before horizon_bars elapsed - the "timeout" in that case
    may understate what would eventually have happened).

    Also tracks, over EVERY bar actually walked (regardless of which barrier
    eventually wins, if any): mae_pct (max adverse excursion - the highest
    high reached, as a fraction above anchor_price - short-thesis "how far
    against me did it get") and mfe_pct (max favorable excursion - the
    lowest low reached, as a fraction below anchor_price), plus
    return_pct_at_exit (anchor_price vs the exit bar's close, positive =
    favorable for the short) - mainly useful for "timeout" outcomes, where
    profit/adverse are fixed by definition but the actual unrealized
    position at the horizon's end is not."""
    n = len(ohlcv)
    if anchor_idx < 0 or anchor_idx >= n or pd.isna(profit_barrier_pct) or pd.isna(adverse_barrier_pct):
        return _empty_outcome("timeout", 0, True)

    anchor_price = float(ohlcv["close"].iloc[anchor_idx])
    if not anchor_price:
        return _empty_outcome("timeout", 0, True)
    profit_price = anchor_price * (1 - profit_barrier_pct)
    adverse_price = anchor_price * (1 + adverse_barrier_pct)

    end_idx = min(n - 1, anchor_idx + horizon_bars)
    right_censored = end_idx < anchor_idx + horizon_bars and end_idx == n - 1

    mae_pct = 0.0
    mfe_pct = 0.0
    for i in range(anchor_idx + 1, end_idx + 1):
        low = float(ohlcv["low"].iloc[i])
        high = float(ohlcv["high"].iloc[i])
        close = float(ohlcv["close"].iloc[i])
        mae_pct = max(mae_pct, (high - anchor_price) / anchor_price)
        mfe_pct = max(mfe_pct, (anchor_price - low) / anchor_price)
        hit_adverse = high >= adverse_price
        hit_profit = low <= profit_price
        if hit_adverse:  # same-candle ambiguity resolved conservatively: adverse wins ties (matches _simulate_trade's "stop wins")
            return {
                "outcome": "adverse", "bars": i - anchor_idx, "right_censored": False,
                "mae_pct": mae_pct, "mfe_pct": mfe_pct, "return_pct_at_exit": (anchor_price - close) / anchor_price,
            }
        if hit_profit:
            return {
                "outcome": "profit", "bars": i - anchor_idx, "right_censored": False,
                "mae_pct": mae_pct, "mfe_pct": mfe_pct, "return_pct_at_exit": (anchor_price - close) / anchor_price,
            }

    exit_close = float(ohlcv["close"].iloc[end_idx]) if end_idx > anchor_idx else anchor_price
    return {
        "outcome": "timeout", "bars": end_idx - anchor_idx, "right_censored": right_censored,
        "mae_pct": mae_pct, "mfe_pct": mfe_pct, "return_pct_at_exit": (anchor_price - exit_close) / anchor_price,
    }


def label_event_across_families(
    ohlcv: pd.DataFrame, anchor_idx: int, bar_interval_ms: int, atr_length: int = 14,
    families: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Runs every entry of `families` (defaults to LABEL_FAMILIES; pass
    PHASE_2B_BARRIER_FAMILIES for the Phase 2B sensitivity grid) for one
    (ohlcv, anchor_idx) - the per-candidate-timestamp, per-family building
    block for the diagnostic scripts. Returns {family_name:
    first_passage_outcome() result}."""
    families = LABEL_FAMILIES if families is None else families
    stop_distance_pct = compute_atr_stop_distance_pct(ohlcv, anchor_idx, atr_length)
    results: dict[str, dict] = {}
    for name, spec in families.items():
        horizon_bars = max(round(spec["horizon_hours"] * 3_600_000 / bar_interval_ms), 1)
        if spec["kind"] == "r":
            if pd.isna(stop_distance_pct):
                results[name] = _empty_outcome("timeout", 0, True)
                continue
            profit_pct, adverse_pct = resolve_r_based_barriers(stop_distance_pct, spec["profit_r"], spec["adverse_r"])
        else:
            profit_pct, adverse_pct = spec["profit_pct"], spec["adverse_pct"]
        results[name] = first_passage_outcome(ohlcv, anchor_idx, profit_pct, adverse_pct, horizon_bars)
    return results


def summarize_outcomes(outcomes: list[dict]) -> dict:
    """Aggregate P(profit)/P(adverse)/P(timeout) and median bars-to-outcome
    over a list of first_passage_outcome()-shaped dicts (one label family,
    many events). Phase 2B point 1: also breaks out raw N per outcome plus
    avg/median return AT TIMEOUT and avg MAE/MFE across ALL walked events
    (regardless of which outcome they resolved to) - a bare P(profit) hides
    whether "not profit" mostly means "stopped out" or "went nowhere"."""
    empty = {
        "n": 0, "n_profit": 0, "n_adverse": 0, "n_timeout": 0, "p_profit": np.nan, "p_adverse": np.nan, "p_timeout": np.nan,
        "median_bars_to_profit": np.nan, "median_bars_to_adverse": np.nan,
        "avg_return_at_timeout_pct": np.nan, "median_return_at_timeout_pct": np.nan,
        "avg_mae_pct": np.nan, "avg_mfe_pct": np.nan,
    }
    if not outcomes:
        return empty
    n = len(outcomes)
    profit = [o for o in outcomes if o["outcome"] == "profit"]
    adverse = [o for o in outcomes if o["outcome"] == "adverse"]
    timeout = [o for o in outcomes if o["outcome"] == "timeout"]
    timeout_returns = [o["return_pct_at_exit"] for o in timeout if pd.notna(o.get("return_pct_at_exit"))]
    mae_vals = [o["mae_pct"] for o in outcomes if pd.notna(o.get("mae_pct"))]
    mfe_vals = [o["mfe_pct"] for o in outcomes if pd.notna(o.get("mfe_pct"))]
    return {
        "n": n,
        "n_profit": len(profit), "n_adverse": len(adverse), "n_timeout": len(timeout),
        "p_profit": len(profit) / n,
        "p_adverse": len(adverse) / n,
        "p_timeout": len(timeout) / n,
        "median_bars_to_profit": float(np.median([o["bars"] for o in profit])) if profit else np.nan,
        "median_bars_to_adverse": float(np.median([o["bars"] for o in adverse])) if adverse else np.nan,
        "avg_return_at_timeout_pct": float(np.mean(timeout_returns)) if timeout_returns else np.nan,
        "median_return_at_timeout_pct": float(np.median(timeout_returns)) if timeout_returns else np.nan,
        "avg_mae_pct": float(np.mean(mae_vals)) if mae_vals else np.nan,
        "avg_mfe_pct": float(np.mean(mfe_vals)) if mfe_vals else np.nan,
    }
