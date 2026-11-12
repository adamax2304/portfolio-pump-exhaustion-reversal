"""Phase 3B.1 point 6: how much move is actually LEFT after a first-alert
signal fires - not just a bars-to-resolution number. Needs raw OHLCV per
event (the flat snapshot table alone cannot answer this), so this is
deliberately a separate, small module the orchestrator calls once per
(event, first-alert bar) pair - never inside a per-bar feature-computation
loop, and never imported by anything in pump_dynamics.py/features.py/
labels.py (same "future-looking code stays physically separate from
anything causal" boundary labels.py's own module docstring already
established for the forward-looking label family).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_remaining_opportunity(
    ohlcv: pd.DataFrame, anchor_idx: int, event_high_so_far: float, horizon_bars: int, reversal_drawdown_pct: float = 0.10,
) -> dict:
    """From anchor_idx forward (the first-alert bar), walks up to
    horizon_bars tracking a causal running peak (starting at
    event_high_so_far, updated as new highs occur) and the drawdown from
    that running peak.

    `additional_upside_before_reversal_pct`: how much further ABOVE
    event_high_so_far price still went before the drawdown-from-running-
    peak first reached reversal_drawdown_pct (0.0 if it never exceeded the
    anchor's own high before reversing, or before the horizon ran out).

    `remaining_downside_pct`: the FULL max drawdown from the running peak
    observed across the WHOLE horizon (not stopped early at the reversal
    threshold) - "how much room was there to fall in total", which can
    exceed reversal_drawdown_pct.

    `bars_to_reversal`: bars from anchor_idx to the FIRST bar where the
    reversal_drawdown_pct threshold was reached, or None if never reached
    within the horizon (right_censored=True in that case)."""
    n = len(ohlcv)
    if anchor_idx < 0 or anchor_idx >= n or not event_high_so_far or pd.isna(event_high_so_far):
        return {"valid": False}

    end_idx = min(n - 1, anchor_idx + horizon_bars)
    running_peak = float(event_high_so_far)
    peak_at_reversal_confirmation = None
    bars_to_reversal = None
    max_drawdown_from_running_peak = 0.0

    for i in range(anchor_idx, end_idx + 1):
        high_i = float(ohlcv["high"].iloc[i])
        low_i = float(ohlcv["low"].iloc[i])
        if high_i > running_peak:
            running_peak = high_i
        drawdown_i = (running_peak - low_i) / running_peak if running_peak else 0.0
        if drawdown_i > max_drawdown_from_running_peak:
            max_drawdown_from_running_peak = drawdown_i
        if bars_to_reversal is None and drawdown_i >= reversal_drawdown_pct:
            bars_to_reversal = i - anchor_idx
            peak_at_reversal_confirmation = running_peak

    final_peak = peak_at_reversal_confirmation if peak_at_reversal_confirmation is not None else running_peak
    additional_upside_pct = (final_peak - event_high_so_far) / event_high_so_far
    right_censored = bars_to_reversal is None and end_idx < anchor_idx + horizon_bars

    return {
        "valid": True,
        "additional_upside_before_reversal_pct": float(additional_upside_pct),
        "remaining_downside_pct": float(max_drawdown_from_running_peak),
        "bars_to_reversal": bars_to_reversal,
        "reversal_confirmed": bars_to_reversal is not None,
        "right_censored": bool(right_censored),
    }


def compute_remaining_opportunity_for_first_alerts(
    first_alerts: pd.DataFrame, ohlcv_by_key: dict, horizon_bars: int, reversal_drawdown_pct: float = 0.10,
    event_col: str = "event_id", abs_idx_col: str = "snapshot_abs_idx", high_col: str = "event_high_so_far",
    key_cols: tuple[str, str] = ("exchange", "symbol"),
) -> pd.DataFrame:
    """Runs compute_remaining_opportunity for every row of `first_alerts`
    (one row per event - see model_evaluation.first_alert_per_event),
    looking up each row's raw OHLCV via `ohlcv_by_key[(exchange, symbol)]`
    (the caller loads these once, same pattern as the snapshot-building
    orchestrators). Rows whose (exchange, symbol) key is missing from
    ohlcv_by_key are skipped (not crashed on) and simply absent from the
    result - a live screener re-running this analysis incrementally would
    see the same "some symbols not loaded yet" gap."""
    rows = []
    for _, row in first_alerts.iterrows():
        key = tuple(row[c] for c in key_cols)
        ohlcv = ohlcv_by_key.get(key)
        if ohlcv is None:
            continue
        result = compute_remaining_opportunity(ohlcv, int(row[abs_idx_col]), float(row[high_col]), horizon_bars, reversal_drawdown_pct)
        if not result.get("valid"):
            continue
        result[event_col] = row[event_col]
        rows.append(result)
    return pd.DataFrame(rows)
