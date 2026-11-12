"""Phase 3A points 4-6: causal forward labels for each snapshot. NOT
eventual-peak-drawdown (research/forward_study.py's target, already shown to
be statistically empty in Phase 1.5 - true 77-92% of the time regardless of
signal). NOT the old exhaustion score (kept as a feature/reference only, see
point 6 - ground truth is forward PRICE outcomes exclusively).

Reuses research/first_passage.first_passage_outcome directly for the
competing-outcome family (its "profit"/"adverse"/"timeout" barrier walk,
same-candle-ambiguity convention, and MAE/MFE tracking are EXACTLY the
REVERSAL/CONTINUATION/NEUTRAL race this spec asks for - profit==price fell
first==REVERSAL, adverse==price rose first==CONTINUATION, timeout==NEUTRAL -
no need to re-derive the barrier walk). Same-candle convention (point 4):
if both barriers are touchable in one bar, CONTINUATION (the outcome
unfavorable to reversal) wins - this IS first_passage_outcome's existing
"adverse wins ties" rule, just relabeled for this question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..research import first_passage as fp

# Point 4: primary + two pre-registered secondary label families - never
# adjusted after looking at results.
LABEL_FAMILIES: dict[str, dict] = {
    "primary": {"reversal_pct": 0.10, "continuation_pct": 0.05, "horizon_hours": 4.0},
    "fast": {"reversal_pct": 0.075, "continuation_pct": 0.05, "horizon_hours": 2.0},
    "deep": {"reversal_pct": 0.15, "continuation_pct": 0.075, "horizon_hours": 8.0},
}

_OUTCOME_RENAME = {"profit": "REVERSAL", "adverse": "CONTINUATION", "timeout": "NEUTRAL"}

# Point 5: PEAK_HAZARD - frozen research parameters, not tuned per-result.
PEAK_HAZARD_NO_NEW_HIGH_TOLERANCE_PCT = 0.02
PEAK_HAZARD_DRAWDOWN_THRESHOLD_PCT = 0.075
PEAK_HAZARD_HORIZON_HOURS = 2.0

# Phase 3A.1 point 4: REVERSAL_FROM_HIGH - anchored to event_high_so_far
# (already causally known at snapshot time), NOT to the snapshot's own
# current price. Same reversal_pct/horizon as "primary" for direct
# comparability - a deliberate choice, not tuned per-result.
REVERSAL_FROM_HIGH_DRAWDOWN_PCT = 0.10
REVERSAL_FROM_HIGH_EXTENSION_PCT = 0.05
REVERSAL_FROM_HIGH_HORIZON_HOURS = 4.0


def compute_high_anchored_label(
    ohlcv: pd.DataFrame, anchor_idx: int, event_high_so_far: float, horizon_bars: int,
    drawdown_pct: float = REVERSAL_FROM_HIGH_DRAWDOWN_PCT, extension_pct: float = REVERSAL_FROM_HIGH_EXTENSION_PCT,
) -> dict:
    """Phase 3A.1 point 4: fixes the positional bias in the ORIGINAL
    (snapshot-price-anchored) competing-outcome label. A snapshot already
    8% below the event high needs only a further ~2.2% drop to be "10% below
    the high" - the price-anchored label instead asks it for a FRESH 10%
    drop from its OWN (already depressed) price, mechanically making
    "REVERSAL" look less likely the further a pullback has already
    progressed, even with zero real information content. This label asks
    the SAME question (is price 10% below the peak) regardless of how far
    into the move the snapshot already is - "REVERSAL" fires the moment
    price is AT OR BELOW drawdown_pct-below-the-peak (immediately, if
    already there at the anchor bar), "CONTINUATION" fires if price
    instead extends extension_pct above the peak first. Same-candle
    convention unchanged: CONTINUATION (unfavorable to reversal) wins ties.

    Returns outcome=None (not a label) if event_high_so_far is missing -
    there is nothing to anchor to."""
    n = len(ohlcv)
    if anchor_idx < 0 or anchor_idx >= n or not event_high_so_far or pd.isna(event_high_so_far):
        return {"outcome": None, "bars": 0, "right_censored": True}

    drawdown_price = event_high_so_far * (1 - drawdown_pct)
    extension_price = event_high_so_far * (1 + extension_pct)
    anchor_low = float(ohlcv["low"].iloc[anchor_idx])
    anchor_high = float(ohlcv["high"].iloc[anchor_idx])
    if anchor_low <= drawdown_price or anchor_high >= extension_price:
        # already resolved AT this snapshot's own bar (e.g. deep into an
        # already-realized drawdown) - not a meaningful "will THIS resolve
        # forward" data point; exclude rather than fabricate a same-bar answer.
        return {"outcome": None, "bars": 0, "right_censored": False}

    end_idx = min(n - 1, anchor_idx + horizon_bars)
    right_censored = end_idx < anchor_idx + horizon_bars and end_idx == n - 1
    for i in range(anchor_idx + 1, end_idx + 1):
        low, high = float(ohlcv["low"].iloc[i]), float(ohlcv["high"].iloc[i])
        hit_extension = high >= extension_price
        hit_drawdown = low <= drawdown_price
        if hit_extension and hit_drawdown:
            hit_drawdown = False
        if hit_extension:
            return {"outcome": "CONTINUATION", "bars": i - anchor_idx, "right_censored": False}
        if hit_drawdown:
            return {"outcome": "REVERSAL", "bars": i - anchor_idx, "right_censored": False}
    return {"outcome": "NEUTRAL", "bars": end_idx - anchor_idx, "right_censored": right_censored}


def compute_competing_outcome_label(ohlcv: pd.DataFrame, anchor_idx: int, reversal_pct: float, continuation_pct: float, horizon_bars: int) -> dict:
    raw = fp.first_passage_outcome(ohlcv, anchor_idx, profit_barrier_pct=reversal_pct, adverse_barrier_pct=continuation_pct, horizon_bars=horizon_bars)
    raw["outcome"] = _OUTCOME_RENAME[raw["outcome"]]
    return raw


def label_snapshot_across_families(ohlcv: pd.DataFrame, anchor_idx: int, bar_interval_ms: int, families: dict[str, dict] | None = None) -> dict[str, dict]:
    """Runs every family in LABEL_FAMILIES (or a caller-supplied override,
    e.g. for a unit test) for one snapshot's anchor bar."""
    families = LABEL_FAMILIES if families is None else families
    out: dict[str, dict] = {}
    for name, spec in families.items():
        horizon_bars = max(round(spec["horizon_hours"] * 3_600_000 / bar_interval_ms), 1)
        out[name] = compute_competing_outcome_label(ohlcv, anchor_idx, spec["reversal_pct"], spec["continuation_pct"], horizon_bars)
    return out


def compute_peak_hazard_label(
    ohlcv: pd.DataFrame, anchor_idx: int, event_high_so_far: float, bar_interval_ms: int,
    no_new_high_tolerance_pct: float = PEAK_HAZARD_NO_NEW_HIGH_TOLERANCE_PCT,
    drawdown_threshold_pct: float = PEAK_HAZARD_DRAWDOWN_THRESHOLD_PCT,
    horizon_hours: float = PEAK_HAZARD_HORIZON_HOURS,
) -> dict:
    """Point 5: is THIS pump leg close to done? Positive iff, within the
    horizon: (a) price never meaningfully exceeds the running high as of
    this snapshot (event_high_so_far - a forward-looking check against a
    value that was itself already known causally at snapshot time), AND
    (b) price still gives a genuine max-drawdown-from-its-own-peak of at
    least drawdown_threshold_pct somewhere in the window (a real
    peak-then-drop, not just "the close N bars later happens to be lower" -
    which could miss an interim new high followed by the drop, or credit a
    slow drift that never actually reversed hard)."""
    horizon_bars = max(round(horizon_hours * 3_600_000 / bar_interval_ms), 1)
    n = len(ohlcv)
    if anchor_idx < 0 or anchor_idx >= n or not event_high_so_far or pd.isna(event_high_so_far):
        return {"peak_hazard": None, "bars": 0, "right_censored": True, "max_high_extension_pct": np.nan, "max_drawdown_pct": np.nan}

    end_idx = min(n - 1, anchor_idx + horizon_bars)
    right_censored = end_idx < anchor_idx + horizon_bars and end_idx == n - 1
    window_high = ohlcv["high"].iloc[anchor_idx + 1 : end_idx + 1]
    window_low = ohlcv["low"].iloc[anchor_idx + 1 : end_idx + 1]
    if len(window_high) == 0:
        return {"peak_hazard": None, "bars": 0, "right_censored": True, "max_high_extension_pct": np.nan, "max_drawdown_pct": np.nan}

    max_price_in_window = float(window_high.max())
    max_high_extension_pct = (max_price_in_window - event_high_so_far) / event_high_so_far
    running_peak = window_high.cummax()
    drawdown_series = (running_peak - window_low) / running_peak.replace(0, np.nan)
    max_drawdown_pct = float(drawdown_series.max()) if drawdown_series.notna().any() else np.nan

    no_meaningful_new_high = max_high_extension_pct <= no_new_high_tolerance_pct
    genuine_drawdown = pd.notna(max_drawdown_pct) and max_drawdown_pct >= drawdown_threshold_pct
    peak_hazard = bool(no_meaningful_new_high and genuine_drawdown)
    return {
        "peak_hazard": peak_hazard, "bars": len(window_high), "right_censored": right_censored,
        "max_high_extension_pct": float(max_high_extension_pct), "max_drawdown_pct": max_drawdown_pct,
    }
