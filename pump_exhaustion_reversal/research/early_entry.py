"""Stage 2A Phase 2C: earlier, causal entry-family research. Phase 2B showed
first_exhaustion_signal (36.6% first-passage P(profit)) outperforming
structure_break (29.9%), and failed_reclaim (37.5%) only a small further lift
over first_exhaustion_signal despite a ~34.5-bar median wait - raising the
question of whether waiting for a full structure-break-then-failed-reclaim
sequence gives away more edge (via worse entry price/timing) than it adds via
higher hit rate. This module tests specific, pre-defined earlier entry rules
(ENTRY A: exhaustion-signal combinations; ENTRY B: climax rejection) instead
of the current failed-reclaim-only anchor (ENTRY C, unchanged benchmark).

Detection reuses backtest/simulator.py's OWN _build_event_inputs /
_build_post_pump_inputs / _build_exhaustion_inputs_for_attempt and
strategy/exhaustion_detector.compute_exhaustion_flags_at directly - "first bar
where combination X holds" is computed with IDENTICAL logic to what a live/
backtest scan would compute, not a research-only reimplementation of the
flags. Still diagnostic-only: nothing here is imported by backtest/simulator.py
or strategy/* (see research/__init__.py's lookahead-safety framing), and nothing
here opens a position - it only locates a bar index a later research script
turns into a first-passage/cost-model evaluation via research/entry_edge.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.simulator import DEFAULT_MAX_EVENT_HORIZON_BARS, EventRecord, _build_event_inputs, _build_exhaustion_inputs_for_attempt, _build_post_pump_inputs
from ..config import AppConfig
from ..data.normalizer import parse_timeframe_ms
from ..strategy.exhaustion_detector import compute_exhaustion_flags_at

# Point 6's fixed, pre-defined combination list - a small set with a stated
# microstructure rationale, not a parameter search. Each requires ALL listed
# flags true on the SAME closed bar.
ENTRY_A_COMBINATIONS: dict[str, tuple[str, ...]] = {
    "cvd_plus_volume_decay": ("cvd_bearish_divergence", "volume_decay"),
    "cvd_plus_oi_rollover": ("cvd_bearish_divergence", "oi_rollover"),
    "cvd_plus_sweep": ("cvd_bearish_divergence", "sweep_rejection"),
    "oi_plus_volume_decay": ("oi_rollover", "volume_decay"),
    "cvd_plus_volume_decay_plus_sweep": ("cvd_bearish_divergence", "volume_decay", "sweep_rejection"),
    "cvd_plus_oi_plus_volume_decay": ("cvd_bearish_divergence", "oi_rollover", "volume_decay"),
}

# Combinations that require OI data to mean anything - point 8: don't mix
# "OI missing" with "OI no rollover" when reporting these specifically.
ENTRY_A_OI_DEPENDENT: frozenset[str] = frozenset({"cvd_plus_oi_rollover", "oi_plus_volume_decay", "cvd_plus_oi_plus_volume_decay"})

# Entry B: climax rejection - sweep/rejection plus at least one of CVD
# divergence / volume decay. "No major catalyst" is applied by the caller via
# EventRecord.news_flag (the only historical catalyst signal actually
# captured - see Phase 1's documented sparse-news-coverage caveat), not here.
ENTRY_B_REQUIRED: tuple[str, ...] = ("sweep_rejection",)
ENTRY_B_ANY_OF: tuple[str, ...] = ("cvd_bearish_divergence", "volume_decay")

ALL_EARLY_ENTRY_FAMILY_NAMES: tuple[str, ...] = tuple(ENTRY_A_COMBINATIONS) + ("entry_b",)


@dataclass
class EarlyEntrySignal:
    event_id: str
    attempt_number: int
    entry_family: str  # one of ALL_EARLY_ENTRY_FAMILY_NAMES
    signal_wide_idx: int  # WIDE (df_slice-local) coordinate - same system as EventRecord.climax_local_idx etc.
    climax_wide_idx: int
    oi_dependent: bool


def _walk_attempt_window(exhaustion_inputs, cfg: AppConfig, n_bars: int) -> dict[str, int | None]:
    """Single bar-by-bar walk (post-pump-local coordinates) computing
    exhaustion flags ONCE per bar and checking every ENTRY_A_COMBINATIONS
    entry plus Entry B against that one flags snapshot - far cheaper than
    re-walking the window once per combination, and guarantees every
    combination is evaluated against the exact same per-bar flag state."""
    found: dict[str, int | None] = dict.fromkeys(ALL_EARLY_ENTRY_FAMILY_NAMES, None)
    remaining = set(found)
    for i in range(n_bars):
        if not remaining:
            break
        flags = compute_exhaustion_flags_at(i, exhaustion_inputs, cfg).as_dict()
        for name in list(remaining):
            if name == "entry_b":
                required, any_of = ENTRY_B_REQUIRED, ENTRY_B_ANY_OF
            else:
                required, any_of = ENTRY_A_COMBINATIONS[name], ()
            if not all(flags.get(r) is True for r in required):
                continue
            if any_of and not any(flags.get(a) is True for a in any_of):
                continue
            found[name] = i
            remaining.discard(name)
    return found


def _detect_idx_for(event: EventRecord, ohlcv: pd.DataFrame) -> int | None:
    idx = int(np.searchsorted(ohlcv["timestamp"].to_numpy(), event.pump_detected_ts, side="left"))
    return idx if idx < len(ohlcv) else None


def find_early_entry_signals(
    event: EventRecord, ohlcv: pd.DataFrame, feature_frame: pd.DataFrame, timeframe: str, cfg: AppConfig,
) -> tuple[list[EarlyEntrySignal], int | None]:
    """Detects Entry A (all 6 combinations) and Entry B on EVERY reversal
    attempt this event actually had (event.attempt_climax_local_idxs, already
    max-2-attempts-capped and re-anchored per Stage 2A's attempt/event split -
    a new pump high invalidates only the current attempt, not re-implemented
    here, just walked). Returns (signals, pump_start_idx) - pump_start_idx is
    needed by the caller to translate signal_wide_idx/climax_wide_idx into an
    absolute `ohlcv` position (same WIDE-coordinate convention as
    lifecycle_diagnostics._anchor_idx_for)."""
    bar_interval_ms = parse_timeframe_ms(timeframe)
    detect_idx = _detect_idx_for(event, ohlcv)
    if detect_idx is None:
        return [], None
    horizon_bars = min(DEFAULT_MAX_EVENT_HORIZON_BARS, len(ohlcv) - 1 - detect_idx)
    if horizon_bars <= 0:
        return [], None
    inputs = _build_event_inputs(ohlcv, feature_frame, detect_idx, timeframe, cfg, horizon_bars)

    climax_wide_idxs = [int(c) for c in event.attempt_climax_local_idxs if pd.notna(c)]
    if not climax_wide_idxs and pd.notna(event.climax_local_idx):
        climax_wide_idxs = [int(event.climax_local_idx)]

    signals: list[EarlyEntrySignal] = []
    for attempt_number, climax_wide_idx in enumerate(climax_wide_idxs, start=1):
        if attempt_number == 1 and climax_wide_idx == inputs.post_pump.climax_idx:
            post_pump = inputs.post_pump
            exhaustion_inputs = inputs.exhaustion_inputs
        else:
            post_pump = _build_post_pump_inputs(inputs.df_slice, inputs.cvd_slice, climax_wide_idx, bar_interval_ms, cfg)
            exhaustion_inputs = _build_exhaustion_inputs_for_attempt(
                post_pump, inputs.volume_decayed, inputs.liq_depletion_ratio, inputs.avwap_loss_columns, inputs.funding_overheated
            )
        n_bars = len(post_pump.post_pump_df)
        found = _walk_attempt_window(exhaustion_inputs, cfg, n_bars)
        for name, local_idx in found.items():
            if local_idx is None:
                continue
            signals.append(
                EarlyEntrySignal(
                    event_id=event.event_id, attempt_number=attempt_number, entry_family=name,
                    signal_wide_idx=post_pump.structure_start_idx + local_idx, climax_wide_idx=climax_wide_idx,
                    oi_dependent=name in ENTRY_A_OI_DEPENDENT,
                )
            )
    return signals, inputs.pump_start_idx


def oi_coverage_adequate(feature_frame: pd.DataFrame, start_idx: int, end_idx: int, min_fraction: float = 0.8) -> bool:
    """Point 8: an OI-dependent combination's result is only interpretable on
    events where OI data actually covered most of the post-climax window
    being evaluated - never mix "OI missing" (data never existed there) with
    "OI didn't roll over" (data existed and said no)."""
    if "open_interest" not in feature_frame.columns:
        return False
    window = feature_frame["open_interest"].iloc[max(0, start_idx) : end_idx + 1]
    if len(window) == 0:
        return False
    return float(window.notna().mean()) >= min_fraction
