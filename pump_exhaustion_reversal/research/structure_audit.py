"""Structure-break diagnostic audit (Stage 1.5 research/diagnostic phase, see
research/__init__.py's own no-lookahead-into-live boundary note). Two
independent, pure-read analyses over already-computed EventRecord fields -
no re-simulation, no re-classification, no trading logic:

1. new_pump_high_timing_distribution: how many bars elapse, after a
   structure break, before backtest/simulator.py's own
   _post_break_invalidation_reason() fires its "new pump high" case (the
   pump resumed beyond the pre-break swing high and invalidated the setup) -
   i.e. how much of FailedReclaimConfig.max_bars_after_break's post-break
   waiting window is typically "wasted" watching a setup that a resumed pump
   was always going to kill anyway, versus how often that check only fires
   very late.
2. classify_pivot_significance / broken_hl_significance_distribution: buckets
   the ATR-multiple magnitude of the higher-low that broke to trigger a
   structure break (features/structure.py's compute_swing_magnitude output,
   joined by the caller) into MICRO/LOCAL/MEANINGFUL/MAJOR, to see whether
   structure breaks worth attention tend to come from breaking a
   barely-there swing vs a genuinely major one.

The MICRO/LOCAL/MEANINGFUL/MAJOR ATR-multiple cutpoints below are a
diagnostic-only categorization for this report. They intentionally live
ONLY in this module - never in config.py or any YAML config file - so this
categorization can never silently become a production strategy parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Simple, round ATR-multiple cutpoints for classify_pivot_significance, chosen relative to
# StructureConfig.min_swing_distance_atr's own default (0.35 ATR - the minimum a swing needs to even
# be confirmed at all):
#   MICRO_HL    (< 0.5 ATR):  barely past compute_structure's own confirmation gate - noise-adjacent.
#   LOCAL_HL    (0.5-1.5 ATR): a small, local pullback - roughly "within one ATR-ish".
#   MEANINGFUL_HL (1.5-4 ATR): a multi-ATR swing - a genuinely meaningful move, not noise.
#   MAJOR_HL    (>= 4 ATR):   outsized relative to recent volatility - a major structural level.
# Each bucket's own lower bound is inclusive (half-open [lower, upper) ranges), so a value sitting
# exactly on a cutpoint belongs to the bucket named by that cutpoint (e.g. exactly 0.5 -> LOCAL_HL).
MICRO_HL_MAX_ATR = 0.5
LOCAL_HL_MAX_ATR = 1.5
MEANINGFUL_HL_MAX_ATR = 4.0


def new_pump_high_timing_distribution(events: list) -> dict:
    """Among `events` whose invalidation_reason contains "new pump high" AND
    have both structure_break_bar_local_idx and dynamic_invalidation_bar_local_idx
    set, compute lag_bars = dynamic_invalidation_bar_local_idx -
    structure_break_bar_local_idx for each and summarize the distribution.

    n_excluded_missing_index counts events that matched the "new pump high"
    substring but were dropped for missing one or both bar indices; it is
    computed independently of the lag sample, so it stays a real, accurate
    count even when n_matched == 0 (only the fields that actually need a
    non-empty lag sample - the mean/percentiles/bucket counts - fall back to
    NaN/0 in that case, matching this codebase's existing convention
    elsewhere, e.g. research/stats.py's bootstrap_ci, of reporting a real
    n/count alongside a NaN statistic rather than fabricating everything).
    """
    matched_reason = [e for e in events if e.invalidation_reason and "new pump high" in e.invalidation_reason]
    matched = [
        e
        for e in matched_reason
        if e.structure_break_bar_local_idx is not None and e.dynamic_invalidation_bar_local_idx is not None
    ]
    n_excluded_missing_index = len(matched_reason) - len(matched)

    lags = np.array(
        [e.dynamic_invalidation_bar_local_idx - e.structure_break_bar_local_idx for e in matched], dtype=float
    )
    n_matched = len(lags)

    result = {"n_matched": n_matched, "n_excluded_missing_index": n_excluded_missing_index}
    if n_matched == 0:
        result.update(
            {
                "mean_bars": np.nan, "p25_bars": np.nan, "p50_bars": np.nan, "p75_bars": np.nan,
                "p90_bars": np.nan, "p95_bars": np.nan,
                "within_1_bar": 0, "within_2_bars": 0, "within_3_bars": 0, "within_6_bars": 0,
                "within_12_bars": 0, "within_24_bars": 0, "over_24_bars": 0,
            }
        )
        return result

    result["mean_bars"] = float(np.mean(lags))
    result["p25_bars"] = float(np.percentile(lags, 25))
    result["p50_bars"] = float(np.percentile(lags, 50))
    result["p75_bars"] = float(np.percentile(lags, 75))
    result["p90_bars"] = float(np.percentile(lags, 90))
    result["p95_bars"] = float(np.percentile(lags, 95))
    # Overlapping ("at most N bars") coverage buckets, NOT mutually-exclusive histogram bins - each
    # event counts toward every bucket its lag_bars qualifies for at once (a lag of 2 bars counts
    # toward within_2_bars AND within_3_bars AND within_6_bars etc. simultaneously). This deliberately
    # answers "how many of these would a timeout of N bars have already caught", not "how many events'
    # lag falls in this specific range".
    result["within_1_bar"] = int(np.sum(lags <= 1))
    result["within_2_bars"] = int(np.sum(lags <= 2))
    result["within_3_bars"] = int(np.sum(lags <= 3))
    result["within_6_bars"] = int(np.sum(lags <= 6))
    result["within_12_bars"] = int(np.sum(lags <= 12))
    result["within_24_bars"] = int(np.sum(lags <= 24))
    result["over_24_bars"] = int(np.sum(lags > 24))
    return result


def classify_pivot_significance(swing_distance_atr: float) -> str:
    """MICRO_HL/LOCAL_HL/MEANINGFUL_HL/MAJOR_HL per the module-level cutpoints
    above; UNKNOWN when swing_distance_atr is NaN (or otherwise missing)."""
    if pd.isna(swing_distance_atr):
        return "UNKNOWN"
    if swing_distance_atr < MICRO_HL_MAX_ATR:
        return "MICRO_HL"
    if swing_distance_atr < LOCAL_HL_MAX_ATR:
        return "LOCAL_HL"
    if swing_distance_atr < MEANINGFUL_HL_MAX_ATR:
        return "MEANINGFUL_HL"
    return "MAJOR_HL"


def broken_hl_significance_distribution(events_with_magnitude: pd.DataFrame) -> pd.Series:
    """`events_with_magnitude`: one row per event, already carrying a
    swing_distance_atr column - the magnitude (in ATR units) of whichever
    higher-low was broken to trigger that event's structure break. This
    function does NOT perform that join itself: the caller is responsible
    for having joined it from features/structure.py's compute_swing_magnitude
    output at each event's own structure_break_bar_local_idx row (this
    module only ever reads already-computed EventRecord-level data - it has
    no access to the per-symbol raw OHLC/ATR needed to recompute
    compute_swing_magnitude itself)."""
    classes = events_with_magnitude["swing_distance_atr"].apply(classify_pivot_significance)
    return classes.rename("pivot_significance").value_counts()


def render_structure_audit_report(new_high_timing: dict, broken_hl_distribution: pd.Series | None) -> str:
    lines = ["# Structure Audit", ""]
    lines.append("## New-pump-high invalidation timing (structure break -> dynamic invalidation, in bars)")
    lines.append("")
    lines.append(
        f"n_matched={new_high_timing['n_matched']}, n_excluded_missing_index={new_high_timing['n_excluded_missing_index']}"
    )
    lines.append("")
    if new_high_timing["n_matched"] == 0:
        lines.append("No matched 'new pump high' invalidations with both bar indices present - nothing to report.")
    else:
        lines.append("| Stat | Bars |")
        lines.append("|---|---|")
        for label, key in (
            ("Mean", "mean_bars"), ("P25", "p25_bars"), ("P50 (median)", "p50_bars"),
            ("P75", "p75_bars"), ("P90", "p90_bars"), ("P95", "p95_bars"),
        ):
            lines.append(f"| {label} | {new_high_timing[key]:.2f} |")
        lines.append("")
        lines.append(
            "Overlapping ('at most N bars') coverage - each event counts toward every bucket its lag "
            "qualifies for, not just one bin:"
        )
        lines.append("")
        lines.append("| <=1 bar | <=2 bars | <=3 bars | <=6 bars | <=12 bars | <=24 bars | >24 bars |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(
            f"| {new_high_timing['within_1_bar']} | {new_high_timing['within_2_bars']} | "
            f"{new_high_timing['within_3_bars']} | {new_high_timing['within_6_bars']} | "
            f"{new_high_timing['within_12_bars']} | {new_high_timing['within_24_bars']} | "
            f"{new_high_timing['over_24_bars']} |"
        )
    lines.append("")
    lines.append("## Broken higher-low significance distribution (ATR-multiple of the swing that broke)")
    lines.append("")
    if broken_hl_distribution is None:
        lines.append(
            "Not computed for this run - requires joining features/structure.py's compute_swing_magnitude "
            "swing_distance_atr onto each event at its structure_break_bar_local_idx row (see "
            "broken_hl_significance_distribution's docstring); the caller did not supply that join."
        )
    else:
        lines.append("| Class | N |")
        lines.append("|---|---|")
        for cls, n in broken_hl_distribution.items():
            lines.append(f"| {cls} | {n} |")
    return "\n".join(lines)
