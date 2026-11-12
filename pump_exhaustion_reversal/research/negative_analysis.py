"""Spec sections 23/24: False Negative / True Negative analysis. Joins
already-computed EventRecord fields with research/forward_study.py's
forward-path output (joined by event_id) - purely retrospective labeling to
see what the strategy's own exhaustion/entry logic did or didn't catch,
never used for a trading decision (forward_study's own fields already carry
that lookahead-safety boundary - this module only reads them).

Expects a forward_paths_df matching research/forward_study.py's
compute_forward_paths_for_events() output: one row per event_id, with at
least columns `event_id`, `maximum_additional_upside`, and
`drawdown_from_eventual_peak_{1h,4h,8h,24h}`.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from ..backtest.simulator import EventRecord
from ..strategy.exhaustion_detector import ExhaustionFlags

COMPONENT_NAMES: tuple[str, ...] = tuple(ExhaustionFlags().as_dict())


def _events_by_id(events: list[EventRecord]) -> dict[str, EventRecord]:
    return {e.event_id: e for e in events}


def _categorize_reason(reason: str | float | None) -> str:
    if reason is None or (isinstance(reason, float) and pd.isna(reason)):
        return "still open / no invalidation recorded"
    if "new pump high" in reason:
        return "new_pump_high"
    if "higher low" in reason:
        return "bullish_hl_reform"
    if "reclaim:" in reason:
        return "strong_reclaim"
    if reason == "reached event horizon without a valid trigger":
        return "timeout"
    return "other"


def find_false_negatives(
    events: list[EventRecord], forward_paths_df: pd.DataFrame, drawdown_threshold: float = 0.20, within_hours: int = 24
) -> pd.DataFrame:
    """Events where a real, sizeable reversal happened
    (drawdown_from_eventual_peak_{within_hours}h >= drawdown_threshold) but
    the strategy's own lifecycle never produced a trade (event.trade is
    None). within_hours must be one of 1, 4, 8, 24 (forward_study's computed
    horizons)."""
    col = f"drawdown_from_eventual_peak_{within_hours}h"
    if col not in forward_paths_df.columns:
        raise ValueError(f"forward_paths_df is missing expected column {col!r}")
    events_by_id = _events_by_id(events)
    candidates = forward_paths_df[forward_paths_df[col] >= drawdown_threshold]
    rows = []
    for _, fp_row in candidates.iterrows():
        event = events_by_id.get(fp_row["event_id"])
        if event is None or event.trade is not None:
            continue
        fired = [name for name in COMPONENT_NAMES if event.exhaustion_component_first_true_bar.get(name) is not None]
        not_fired = [name for name in COMPONENT_NAMES if name not in fired]
        rows.append(
            {
                "event_id": event.event_id, "symbol": event.symbol, "exchange": event.exchange,
                "pump_class": event.pump_class, "pump_score": event.pump_score,
                "exhaustion_score_peak": event.exhaustion_score_peak, "final_state": event.final_state,
                "invalidation_reason": event.invalidation_reason,
                "drawdown_from_eventual_peak": fp_row[col],
                "features_fired": fired, "features_not_fired": not_fired,
            }
        )
    return pd.DataFrame(rows)


def find_true_negatives(
    events: list[EventRecord], forward_paths_df: pd.DataFrame, continuation_threshold: float = 0.20
) -> pd.DataFrame:
    """Events where the strategy correctly did NOT trade (event.trade is
    None) and price went on to continue up by at least continuation_threshold
    (maximum_additional_upside) - a correct rejection, not a missed
    opportunity."""
    events_by_id = _events_by_id(events)
    candidates = forward_paths_df[forward_paths_df["maximum_additional_upside"] >= continuation_threshold]
    rows = []
    for _, fp_row in candidates.iterrows():
        event = events_by_id.get(fp_row["event_id"])
        if event is None or event.trade is not None:
            continue
        rows.append(
            {
                "event_id": event.event_id, "symbol": event.symbol, "exchange": event.exchange,
                "pump_class": event.pump_class, "exhaustion_score_peak": event.exhaustion_score_peak,
                "final_state": event.final_state, "invalidation_reason": event.invalidation_reason,
                "maximum_additional_upside": fp_row["maximum_additional_upside"],
            }
        )
    return pd.DataFrame(rows)


def summarize_false_negatives(false_negatives_df: pd.DataFrame) -> dict:
    if false_negatives_df.empty:
        return {"n": 0}
    not_fired_counts: Counter = Counter()
    for feats in false_negatives_df["features_not_fired"]:
        not_fired_counts.update(feats)
    return {
        "n": len(false_negatives_df),
        "by_pump_class": false_negatives_df["pump_class"].value_counts().to_dict(),
        "by_invalidation_reason_category": false_negatives_df["invalidation_reason"].map(_categorize_reason).value_counts().to_dict(),
        "most_commonly_missing_features": dict(not_fired_counts.most_common(5)),
        "median_exhaustion_score_peak": float(false_negatives_df["exhaustion_score_peak"].median()),
    }


def summarize_true_negatives(true_negatives_df: pd.DataFrame) -> dict:
    if true_negatives_df.empty:
        return {"n": 0}
    return {
        "n": len(true_negatives_df),
        "by_invalidation_reason_category": true_negatives_df["invalidation_reason"].map(_categorize_reason).value_counts().to_dict(),
        "median_maximum_additional_upside": float(true_negatives_df["maximum_additional_upside"].median()),
    }


def render_negative_analysis_reports(
    false_negatives_df: pd.DataFrame, false_negative_summary: dict,
    true_negatives_df: pd.DataFrame, true_negative_summary: dict,
) -> tuple[str, str]:
    """Returns (false_negative_report_md, true_negative_report_md) - two
    separate markdown strings, matching the two separate deliverable files."""
    fn_lines = ["# False Negative Analysis", "", f"Real reversals the strategy did not trade: n={false_negative_summary.get('n', 0)}", ""]
    if false_negative_summary.get("n", 0) > 0:
        fn_lines.append(f"Median exhaustion_score_peak among these: {false_negative_summary['median_exhaustion_score_peak']:.1f}")
        fn_lines.append("")
        fn_lines.append("By pump class: " + ", ".join(f"{k}={v}" for k, v in false_negative_summary["by_pump_class"].items()))
        fn_lines.append("")
        fn_lines.append(
            "By what actually ended the event: "
            + ", ".join(f"{k}={v}" for k, v in false_negative_summary["by_invalidation_reason_category"].items())
        )
        fn_lines.append("")
        fn_lines.append(
            "Most commonly missing features: "
            + ", ".join(f"{k}={v}" for k, v in false_negative_summary["most_commonly_missing_features"].items())
        )
        fn_lines.append("")
        fn_lines.append("## Individual events")
        fn_lines.append("")
        fn_lines.append("| Symbol | Exchange | Pump class | Exhaustion peak | Drawdown from peak | Features NOT fired |")
        fn_lines.append("|---|---|---|---|---|---|")
        for _, row in false_negatives_df.iterrows():
            fn_lines.append(
                f"| {row['symbol']} | {row['exchange']} | {row['pump_class']} | {row['exhaustion_score_peak']:.1f} "
                f"| {row['drawdown_from_eventual_peak']:.1%} | {', '.join(row['features_not_fired'])} |"
            )

    tn_lines = ["# True Negative Analysis", "", f"Correct rejections (no trade, pump continued): n={true_negative_summary.get('n', 0)}", ""]
    if true_negative_summary.get("n", 0) > 0:
        tn_lines.append(f"Median maximum additional upside among these: {true_negative_summary['median_maximum_additional_upside']:.1%}")
        tn_lines.append("")
        tn_lines.append(
            "By what actually ended the event (i.e. what protected us): "
            + ", ".join(f"{k}={v}" for k, v in true_negative_summary["by_invalidation_reason_category"].items())
        )
        tn_lines.append("")
        tn_lines.append("## Individual events")
        tn_lines.append("")
        tn_lines.append("| Symbol | Exchange | Additional upside | What ended the event |")
        tn_lines.append("|---|---|---|---|")
        for _, row in true_negatives_df.iterrows():
            tn_lines.append(f"| {row['symbol']} | {row['exchange']} | {row['maximum_additional_upside']:.1%} | {row['invalidation_reason']} |")

    return "\n".join(fn_lines), "\n".join(tn_lines)
