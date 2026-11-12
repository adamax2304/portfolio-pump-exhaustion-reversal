"""Phase 4A points 5-7: three causal, pre-registered entry-timing detectors
run AFTER an already-selected V2 confirmation alert (screener_v2's ranking
model has ALREADY told us this pump belongs to the strong-reversal cohort -
these detectors only decide WHEN, within that already-selected candidate, to
actually get short; they do not themselves search for candidates or claim
any ranking power of their own).

Each detector takes the raw post-alert OHLCV and returns a dict with
`valid`, and when valid, `signal_bar_idx` (the LAST bar of the pattern - the
caller enters on signal_bar_idx+1's open, via research.entry_edge.
resolve_entry_price) and `stop_reference_price` (fed into
strategy.risk_engine.compute_structural_stop, NOT used directly as a stop -
this module never touches min/max stop widening, ATR buffering, fees, or R
arithmetic; that is the orchestrator's job, reusing research/entry_edge.py
and strategy/risk_engine.py unchanged).

One frozen parameterization per family (point 6's "не оптимизировать
десятки X/Y") - these constants are fixed BEFORE running on real data and
are not adjusted after seeing results.
"""

from __future__ import annotations

import pandas as pd

# Family B (retest/rebound-then-lower-high): a rebound of at least this many
# ATR off the post-alert running low counts as "a real bounce happened" -
# then a stall of this many bars without a new (rebound-local) high confirms
# the lower-high/rejection.
FAMILY_B_REBOUND_MIN_ATR = 0.5
FAMILY_B_STALL_BARS = 2

# Family C (failed continuation): price must come back within this many ATR
# of the alert bar's own running high to count as "testing" it (a genuine
# retest attempt, not merely drifting nearby); if it exceeds the running
# high by more than this fraction, that is real continuation and the SHORT
# thesis is invalidated for this event (no Family C entry) rather than
# forced through. A stall of this many bars without a new (attempt-local)
# high then confirms the rejection.
FAMILY_C_APPROACH_TOLERANCE_ATR = 1.0
FAMILY_C_INVALIDATION_EXTENSION_PCT = 0.05
FAMILY_C_STALL_BARS = 2


def detect_family_a_immediate(alert_abs_idx: int, event_high_so_far: float) -> dict:
    """Family A: no wait at all - the alert bar itself is the signal.
    stop_reference_price is the causal running high AS OF the alert bar
    (event_high_so_far) - the simplest, most defensible 'recent high'
    reference, already causal by construction (see snapshot_builder.py)."""
    if not event_high_so_far or pd.isna(event_high_so_far):
        return {"valid": False, "reason": "no_event_high"}
    return {"valid": True, "signal_bar_idx": alert_abs_idx, "stop_reference_price": float(event_high_so_far)}


def detect_family_b_retest(
    ohlcv: pd.DataFrame, alert_abs_idx: int, atr_series: pd.Series, max_wait_bars: int,
    rebound_min_atr: float = FAMILY_B_REBOUND_MIN_ATR, stall_bars: int = FAMILY_B_STALL_BARS,
) -> dict:
    """Family B: after the alert, wait for a causal bounce off a post-alert
    low, then wait for that bounce to stall (stop making new highs) -
    signal fires on the stall-confirmation bar, stop_reference_price is the
    rebound's own peak."""
    n = len(ohlcv)
    start = alert_abs_idx + 1
    end = min(n - 1, alert_abs_idx + max_wait_bars)
    if start > end:
        return {"valid": False, "reason": "no_bars_after_alert"}

    # running_low is seeded from bar `start` alone (nothing to rebound off of
    # yet at that very bar); every SUBSEQUENT bar's high is checked against
    # the low established by STRICTLY EARLIER bars before that bar's own low
    # is folded in - otherwise a single bar's own high-low range could look
    # like "a rebound off its own low", which is just normal bar noise, not
    # a real bounce off an already-established post-alert low.
    running_low = float(ohlcv["low"].iloc[start])
    rebound_confirmed_idx = None
    rebound_peak = None
    for i in range(start + 1, end + 1):
        low_i, high_i = float(ohlcv["low"].iloc[i]), float(ohlcv["high"].iloc[i])
        atr_i = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else float("nan")
        if pd.notna(atr_i) and atr_i > 0 and high_i - running_low >= rebound_min_atr * atr_i:
            rebound_confirmed_idx = i
            rebound_peak = high_i
            break
        if low_i < running_low:
            running_low = low_i
    if rebound_confirmed_idx is None:
        return {"valid": False, "reason": "no_rebound_within_horizon"}

    bars_since_new_high = 0
    for i in range(rebound_confirmed_idx + 1, end + 1):
        high_i = float(ohlcv["high"].iloc[i])
        if high_i > rebound_peak:
            rebound_peak = high_i
            bars_since_new_high = 0
        else:
            bars_since_new_high += 1
            if bars_since_new_high >= stall_bars:
                return {
                    "valid": True, "signal_bar_idx": i, "stop_reference_price": float(rebound_peak),
                    "rebound_low": float(running_low), "rebound_confirmed_bar_idx": rebound_confirmed_idx,
                }
    return {"valid": False, "reason": "no_stall_confirmed_within_horizon"}


def detect_family_c_failed_continuation(
    ohlcv: pd.DataFrame, alert_abs_idx: int, event_high_so_far: float, atr_series: pd.Series, max_wait_bars: int,
    approach_tolerance_atr: float = FAMILY_C_APPROACH_TOLERANCE_ATR,
    invalidation_extension_pct: float = FAMILY_C_INVALIDATION_EXTENSION_PCT, stall_bars: int = FAMILY_C_STALL_BARS,
) -> dict:
    """Family C: after the alert, wait for price to test (approach, within
    tolerance) the alert bar's own running high, then fail to extend and
    stall - signal fires on the stall-confirmation bar, stop_reference_price
    is the peak actually reached during the failed attempt. If price instead
    extends MEANINGFULLY past the running high (a genuine breakout, not a
    mere approach), the short thesis is invalidated for this event - no
    Family C entry is forced through a real continuation."""
    if not event_high_so_far or pd.isna(event_high_so_far):
        return {"valid": False, "reason": "no_event_high"}
    n = len(ohlcv)
    start = alert_abs_idx + 1
    end = min(n - 1, alert_abs_idx + max_wait_bars)
    if start > end:
        return {"valid": False, "reason": "no_bars_after_alert"}

    invalidation_price = event_high_so_far * (1 + invalidation_extension_pct)
    attempt_confirmed_idx = None
    attempt_peak = None
    for i in range(start, end + 1):
        high_i = float(ohlcv["high"].iloc[i])
        if high_i >= invalidation_price:
            return {"valid": False, "reason": "continuation_invalidated_thesis"}
        atr_i = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else float("nan")
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        approach_price = event_high_so_far - approach_tolerance_atr * atr_i
        if high_i >= approach_price:
            attempt_confirmed_idx = i
            attempt_peak = high_i
            break
    if attempt_confirmed_idx is None:
        return {"valid": False, "reason": "no_continuation_attempt_within_horizon"}

    bars_since_new_high = 0
    for i in range(attempt_confirmed_idx + 1, end + 1):
        high_i = float(ohlcv["high"].iloc[i])
        if high_i >= invalidation_price:
            return {"valid": False, "reason": "continuation_invalidated_thesis"}
        if high_i > attempt_peak:
            attempt_peak = high_i
            bars_since_new_high = 0
        else:
            bars_since_new_high += 1
            if bars_since_new_high >= stall_bars:
                return {
                    "valid": True, "signal_bar_idx": i, "stop_reference_price": float(attempt_peak),
                    "attempt_confirmed_bar_idx": attempt_confirmed_idx,
                }
    return {"valid": False, "reason": "no_rejection_confirmed_within_horizon"}
