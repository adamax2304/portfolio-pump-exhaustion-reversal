"""Stage 2A Part 4: the lean 6-table diagnostic set the user asked for,
replacing Stage 1.5's 10-report spread for this follow-up pass. This module
covers tables 1 (lifecycle funnel), 3 (first-passage per signal), 4 (per
combination), and 5 (continuation risk per candidate anchor) - table 2
reuses research/exhaustion_audit.hit_rate_by_component unchanged, table 6
reuses research/underlying_classification.py unchanged.

Everything here anchors first-passage outcomes at one of several candidate
timestamps (pump_detected, climax, first exhaustion signal, structure break,
failed reclaim) purely to compare predictive power ACROSS anchors - the
actual point of Stage 2A's re-test. Reads already-saved EventRecord fields
and raw OHLCV only; never imported by backtest/simulator.py or strategy/*
(see research/__init__.py's lookahead-safety framing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord
from ..strategy.exhaustion_detector import ExhaustionFlags
from . import first_passage as fp
from . import stats as research_stats

COMPONENT_NAMES: tuple[str, ...] = tuple(ExhaustionFlags().as_dict())

# Candidate anchor timestamps to compare predictive power across (spec's own
# list): pump_detected is the baseline; the other four require the field to
# actually be populated on a given event (None -&gt; that event is skipped for
# that anchor, not treated as "anchor at bar 0").
ANCHOR_NAMES: tuple[str, ...] = ("pump_detected", "climax", "first_exhaustion_signal", "structure_break", "failed_reclaim")

DEFAULT_LABEL_FAMILY = "standard_4h"


def lifecycle_funnel(events: list[EventRecord], armed_threshold: float) -> dict:
    """Table 1: Pump -> Climax -> Exhaustion -> Structure Break -> Failed
    Reclaim -> Attempt (>1) -> Trade counts. armed_threshold is
    cfg.exhaustion_thresholds.armed (peak score reaching it is used as an
    "exhaustion-forming-or-beyond" proxy, not a re-simulation of the state
    machine's own gating)."""
    n = len(events)
    if n == 0:
        return {"n_pump_detected": 0}
    # pd.notna() (not `is not None`) throughout: dict[str, int | None] and
    # int | None fields round-trip through parquet with None sometimes
    # becoming float('nan') rather than None (verified empirically - see
    # project plan) - `is not None` would silently over-count in that case.
    return {
        "n_pump_detected": n,
        "n_climax_found": sum(1 for e in events if pd.notna(e.climax_local_idx)),
        "n_exhaustion_forming_or_beyond": sum(1 for e in events if e.exhaustion_score_peak >= armed_threshold),
        "n_structure_break": sum(1 for e in events if pd.notna(e.structure_break_bar_local_idx)),
        "n_failed_reclaim": sum(
            1 for e in events if pd.notna(e.exhaustion_component_first_true_bar.get("failed_reclaim_lower_high"))
        ),
        "n_multi_attempt": sum(1 for e in events if e.attempt_count > 1),
        "n_trade": sum(1 for e in events if e.trade is not None),
    }


def render_lifecycle_funnel(funnel: dict) -> str:
    if funnel.get("n_pump_detected", 0) == 0:
        return "# Lifecycle Funnel\n\nNo events."
    lines = ["# Lifecycle Funnel", "", "| Stage | N | % of pump_detected |", "|---|---|---|"]
    total = funnel["n_pump_detected"]
    stage_labels = [
        ("Pump detected", "n_pump_detected"), ("Climax found", "n_climax_found"),
        ("Exhaustion forming or beyond", "n_exhaustion_forming_or_beyond"), ("Structure break", "n_structure_break"),
        ("Failed reclaim", "n_failed_reclaim"), ("Multi-attempt (retried)", "n_multi_attempt"), ("Trade", "n_trade"),
    ]
    for label, key in stage_labels:
        n = funnel[key]
        lines.append(f"| {label} | {n} | {n / total:.1%} |")
    return "\n".join(lines)


def _anchor_idx_for(event: EventRecord, anchor: str, ohlcv: pd.DataFrame, pump_start_idx: int) -> int | None:
    """Resolves one anchor name to a POSITIONAL bar index in `ohlcv` (safe to
    pass straight to .iloc regardless of whatever index labels `ohlcv`
    actually carries - matches research/forward_study.py's own
    np.searchsorted pattern for the same "locate this timestamp" problem;
    `ohlcv.index[mask][0]` is NOT positional in general and must not be used
    here). structure_break/failed_reclaim/first_exhaustion_signal are stored
    as wide (df_slice-local, pump_start_idx-relative) bar indices on
    EventRecord - translated back to absolute ohlcv position here."""
    if anchor == "pump_detected":
        idx = int(np.searchsorted(ohlcv["timestamp"].to_numpy(), event.pump_detected_ts, side="left"))
        return idx if idx < len(ohlcv) else None
    if anchor == "climax":
        if pd.isna(event.climax_local_idx):
            return None
        return pump_start_idx + int(event.climax_local_idx)
    if anchor == "structure_break":
        if pd.isna(event.structure_break_bar_local_idx):
            return None
        return pump_start_idx + int(event.structure_break_bar_local_idx)
    if anchor == "first_exhaustion_signal":
        # dict[str, int | None] fields round-trip through parquet's map type
        # with None becoming float('nan') (not None) for a missing entry
        # (verified empirically - see project plan) - pd.notna() catches
        # both representations; a bare `is not None` check would let NaN
        # slip through and poison min() below.
        first_bars = [b for b in event.exhaustion_component_first_true_bar.values() if pd.notna(b)]
        if not first_bars:
            return None
        return pump_start_idx + int(min(first_bars))
    if anchor == "failed_reclaim":
        bar = event.exhaustion_component_first_true_bar.get("failed_reclaim_lower_high")
        if pd.isna(bar):
            return None
        return pump_start_idx + int(bar)
    raise ValueError(f"unknown anchor {anchor!r}")


def _pump_start_idx_for(event: EventRecord, ohlcv: pd.DataFrame, lookback_bars: int) -> int:
    detect_idx = int(np.searchsorted(ohlcv["timestamp"].to_numpy(), event.pump_detected_ts, side="left"))
    detect_idx = min(detect_idx, len(ohlcv) - 1) if len(ohlcv) else 0
    return max(0, detect_idx - lookback_bars)


def compute_first_passage_by_anchor(
    events: list[EventRecord], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame], bar_interval_ms: int,
    lookback_bars: int, label_family: str = DEFAULT_LABEL_FAMILY,
) -> pd.DataFrame:
    """Table 5 building block: one row per (event, anchor) with that
    anchor's first-passage outcome for `label_family` - the direct test of
    "does structure_break/failed_reclaim add predictive power over a bare
    pump_detected anchor". Also carries each of the 10 exhaustion
    components' fired-or-not flag (as of pump_detected, whole-event - see
    EventRecord.exhaustion_component_first_true_bar) for
    compute_first_passage_by_component below."""
    rows = []
    for event in events:
        key = (event.exchange, event.symbol)
        ohlcv = ohlcv_by_symbol.get(key)
        if ohlcv is None:
            continue
        pump_start_idx = _pump_start_idx_for(event, ohlcv, lookback_bars)
        for anchor in ANCHOR_NAMES:
            anchor_idx = _anchor_idx_for(event, anchor, ohlcv, pump_start_idx)
            if anchor_idx is None or anchor_idx >= len(ohlcv):
                continue
            all_families = fp.label_event_across_families(ohlcv, anchor_idx, bar_interval_ms)
            outcome = all_families[label_family]
            row = {"event_id": event.event_id, "anchor": anchor, "pump_class": event.pump_class, **outcome}
            for name in COMPONENT_NAMES:
                row[name] = event.exhaustion_component_first_true_bar.get(name) is not None
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_anchor(first_passage_df: pd.DataFrame) -> pd.DataFrame:
    """Table 5: P(profit)/P(adverse)/P(timeout) per candidate anchor."""
    if first_passage_df.empty:
        return pd.DataFrame(columns=["anchor", "n", "p_profit", "p_adverse", "p_timeout"])
    rows = []
    for anchor, group in first_passage_df.groupby("anchor"):
        summary = fp.summarize_outcomes(group.to_dict("records"))
        rows.append({"anchor": anchor, **summary})
    order = {name: i for i, name in enumerate(ANCHOR_NAMES)}
    return pd.DataFrame(rows).sort_values("anchor", key=lambda s: s.map(order)).reset_index(drop=True)


def first_passage_single_feature_counterfactual(first_passage_df: pd.DataFrame, anchor: str = "pump_detected") -> pd.DataFrame:
    """Table 3: for each of the 10 components, P(profit before adverse |
    component ever fired) vs P(... | never fired) - restricted to one
    anchor's rows (pump_detected by default) so every event contributes
    exactly once."""
    subset = first_passage_df[first_passage_df["anchor"] == anchor]
    if subset.empty:
        return pd.DataFrame(columns=["feature", "n_with_condition", "n_without_condition", "p_outcome_given_condition", "p_outcome_given_not_condition"])
    outcome = subset["outcome"] == "profit"
    rows = []
    for name in COMPONENT_NAMES:
        result = research_stats.conditional_probability(subset[name].astype(bool), outcome)
        result["feature"] = name
        rows.append(result)
    return pd.DataFrame(rows)[["feature", "n_with_condition", "n_without_condition", "p_outcome_given_condition", "p_outcome_given_not_condition"]]


def first_passage_combination_counterfactual(
    first_passage_df: pd.DataFrame, combinations: list[tuple[str, ...]], anchor: str = "pump_detected"
) -> pd.DataFrame:
    """Table 4: same as the single-feature version, but for feature
    combinations (all listed components must have fired)."""
    subset = first_passage_df[first_passage_df["anchor"] == anchor]
    if subset.empty:
        return pd.DataFrame(columns=["combination", "n_with_condition", "n_without_condition", "p_outcome_given_condition", "p_outcome_given_not_condition"])
    outcome = subset["outcome"] == "profit"
    rows = []
    for combo in combinations:
        condition = subset[list(combo)].all(axis=1)
        result = research_stats.conditional_probability(condition, outcome)
        result["combination"] = " + ".join(combo)
        rows.append(result)
    return pd.DataFrame(rows)[["combination", "n_with_condition", "n_without_condition", "p_outcome_given_condition", "p_outcome_given_not_condition"]]


def _fmt_pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.1%}"


def render_first_passage_report(
    by_anchor: pd.DataFrame, single_feature: pd.DataFrame, combinations: pd.DataFrame, label_family: str
) -> str:
    lines = [f"# First-Passage Diagnostics (label family: {label_family})", ""]
    lines.append("## Table 5: outcome by candidate anchor timestamp")
    lines.append("")
    lines.append("| Anchor | N | P(profit) | P(adverse) | P(timeout) | Median bars to profit |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in by_anchor.iterrows():
        median_bars = "n/a" if pd.isna(row.get("median_bars_to_profit")) else f"{row['median_bars_to_profit']:.1f}"
        lines.append(
            f"| {row['anchor']} | {row['n']} | {_fmt_pct(row['p_profit'])} | {_fmt_pct(row['p_adverse'])} "
            f"| {_fmt_pct(row['p_timeout'])} | {median_bars} |"
        )
    lines.append("")
    lines.append("## Table 3: single-feature counterfactual (P(profit) given feature fired vs not, at pump_detected)")
    lines.append("")
    lines.append("| Feature | N fired | N not fired | P(profit\\|fired) | P(profit\\|not fired) |")
    lines.append("|---|---|---|---|---|")
    for _, row in single_feature.iterrows():
        lines.append(
            f"| {row['feature']} | {row['n_with_condition']} | {row['n_without_condition']} "
            f"| {_fmt_pct(row['p_outcome_given_condition'])} | {_fmt_pct(row['p_outcome_given_not_condition'])} |"
        )
    lines.append("")
    lines.append("## Table 4: feature combinations")
    lines.append("")
    lines.append("| Combination | N fired | N not fired | P(profit\\|fired) | P(profit\\|not fired) |")
    lines.append("|---|---|---|---|---|")
    for _, row in combinations.iterrows():
        lines.append(
            f"| {row['combination']} | {row['n_with_condition']} | {row['n_without_condition']} "
            f"| {_fmt_pct(row['p_outcome_given_condition'])} | {_fmt_pct(row['p_outcome_given_not_condition'])} |"
        )
    return "\n".join(lines)
