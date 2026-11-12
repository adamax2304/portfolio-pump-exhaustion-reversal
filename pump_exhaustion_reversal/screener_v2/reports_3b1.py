"""Phase 3B.1's 7 deliverables - markdown rendering only, all computation
lives in logistic_model.py / model_evaluation.py / remaining_opportunity.py
/ validation.py. Reuses reports.py's _fmt_pct/_fmt_num and reports_3b.py's
_render_decile_curve_table rather than duplicating them.
"""

from __future__ import annotations

import pandas as pd

from .reports import _fmt_num, _fmt_pct
from .reports_3b import _render_decile_curve_table


def render_v1_vs_v2_comparison(v1_curve: pd.DataFrame, v2_curve: pd.DataFrame, v1_metrics: dict, v2_metrics: dict, v1_precision: dict, v2_precision: dict, v1_boot: dict, v2_boot: dict) -> str:
    """Deliverable: v1_vs_v2_comparison.md (point 3)."""
    lines = [
        "# V1 vs V2 Comparison", "",
        "V1 = Phase 3B's FEATURE_SET_V1 (includes oi_regime and bars_since_last_new_high). V2 = FEATURE_SET_V2 "
        "(pullback_depth_atr, rejection_atr, lower_high_strength, realized_vol_15m, current_return_from_start, "
        "upper_wick_pct only - oi_regime and bars_since_last_new_high dropped per Phase 3B's own ablation and "
        "correlation evidence). Both fit on the IDENTICAL temporal folds/event-weighting/symbol-grouping/forward-lock.", "",
        "## Score decile curve", "",
        "### V1", "", _render_decile_curve_table(v1_curve), "",
        "### V2", "", _render_decile_curve_table(v2_curve), "",
        "## Precision@K (first-alert / unique)", "",
        "| Model | K=1 | K=3 | K=5 |", "|---|---|---|---|",
        f"| V1 | {_fmt_pct(v1_precision.get(1, {}).get('precision_among_unique_alerts'))} | {_fmt_pct(v1_precision.get(3, {}).get('precision_among_unique_alerts'))} | {_fmt_pct(v1_precision.get(5, {}).get('precision_among_unique_alerts'))} |",
        f"| V2 | {_fmt_pct(v2_precision.get(1, {}).get('precision_among_unique_alerts'))} | {_fmt_pct(v2_precision.get(3, {}).get('precision_among_unique_alerts'))} | {_fmt_pct(v2_precision.get(5, {}).get('precision_among_unique_alerts'))} |",
        "",
        "## Classification metrics (shown for completeness only - not the deciding criteria)", "",
        "| Model | N resolved | ROC-AUC | PR-AUC | Brier |", "|---|---|---|---|---|",
        f"| V1 | {v1_metrics.get('n_resolved')} | {_fmt_num(v1_metrics.get('roc_auc'))} | {_fmt_num(v1_metrics.get('pr_auc'))} | {_fmt_num(v1_metrics.get('brier_score'))} |",
        f"| V2 | {v2_metrics.get('n_resolved')} | {_fmt_num(v2_metrics.get('roc_auc'))} | {_fmt_num(v2_metrics.get('pr_auc'))} | {_fmt_num(v2_metrics.get('brier_score'))} |",
        "",
        "## Top-10% bucket bootstrap 95% CI", "",
        f"- V1 event-cluster: {_fmt_pct(v1_boot.get('event', {}).get('statistic'))} [{_fmt_pct(v1_boot.get('event', {}).get('ci_low'))}, {_fmt_pct(v1_boot.get('event', {}).get('ci_high'))}]",
        f"- V1 symbol-cluster: {_fmt_pct(v1_boot.get('symbol', {}).get('statistic'))} [{_fmt_pct(v1_boot.get('symbol', {}).get('ci_low'))}, {_fmt_pct(v1_boot.get('symbol', {}).get('ci_high'))}]",
        f"- V2 event-cluster: {_fmt_pct(v2_boot.get('event', {}).get('statistic'))} [{_fmt_pct(v2_boot.get('event', {}).get('ci_low'))}, {_fmt_pct(v2_boot.get('event', {}).get('ci_high'))}]",
        f"- V2 symbol-cluster: {_fmt_pct(v2_boot.get('symbol', {}).get('statistic'))} [{_fmt_pct(v2_boot.get('symbol', {}).get('ci_low'))}, {_fmt_pct(v2_boot.get('symbol', {}).get('ci_high'))}]",
    ]
    return "\n".join(lines)


def render_predump_ranking(bucket_curve: pd.DataFrame, cumulative_curve: pd.DataFrame) -> str:
    """Deliverable: predump_ranking_by_drawdown_bucket.md (points 4-5) - the
    central question of this phase: does V2's ranking edge exist BEFORE a
    meaningful drawdown from the event high has already happened, or only
    after."""
    lines = [
        "# Pre-Dump Ranking by Drawdown Bucket", "",
        "Points 4-5: V2's ranking edge, split by how far price already sits below the event's own running high at "
        "the snapshot being scored. The top-X% threshold in each bucket is computed WITHIN that bucket's own scored "
        "population, not globally - a live screener ranking 'still near the highs' candidates only compares them "
        "against other still-near-the-highs candidates. If lift is only real once drawdown is already large, this is "
        "a CONFIRMATION screener, not an EARLY-WARNING one.", "",
        "## Discrete buckets", "",
    ]
    for bucket, group in bucket_curve.groupby("bucket", sort=False):
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append(_render_decile_curve_table(group.drop(columns=["bucket"])))
        lines.append("")
    lines.append("## Cumulative pre-dump cohorts (pooled, more N)")
    lines.append("")
    for bucket, group in cumulative_curve.groupby("bucket", sort=False):
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append(_render_decile_curve_table(group.drop(columns=["bucket"])))
        lines.append("")
    return "\n".join(lines)


def render_first_alert_results(counts_by_quantile: dict, precision_by_quantile: dict, cooldown_note: str) -> str:
    """Deliverable: first_alert_event_level_results.md (points 7-9)."""
    lines = [
        "# First-Alert Event-Level Results", "",
        "Point 7: one alert per event, at the FIRST snapshot it enters each threshold - later snapshots of the same "
        "event in the same bucket are not separate opportunities. Point 9: snapshot-level Precision@K (bar-by-bar) "
        "vs first-alert Precision@K (the live-like, production-relevant number) side by side.", "",
        "| Threshold | N events alerted | N true (resolved REVERSAL) | N false | First-alert precision |",
        "|---|---|---|---|---|",
    ]
    for q, c in counts_by_quantile.items():
        lines.append(f"| top {1 - q:.0%} | {c['n_alerted']} | {c['n_true']} | {c['n_false']} | {_fmt_pct(c['precision'])} |")
    lines.append("")
    lines.append("## Snapshot vs first-alert Precision@K (from precision_at_k, K=1/3/5)")
    lines.append("")
    lines.append("| K | Snapshot (bar-level) Precision@K | First-alert Precision@K |")
    lines.append("|---|---|---|")
    for k, p in precision_by_quantile.items():
        lines.append(f"| {k} | {_fmt_pct(p['snapshot'])} | {_fmt_pct(p['first_alert'])} |")
    lines.append("")
    lines.append("## Point 8 cooldown sanity check")
    lines.append("")
    lines.append(cooldown_note)
    return "\n".join(lines)


def render_remaining_opportunity(stats_by_quantile: dict) -> str:
    """Deliverable: remaining_opportunity_analysis.md (point 6)."""
    lines = [
        "# Remaining Opportunity Analysis", "",
        "Point 6: for each event's FIRST entry into a score threshold, how much move was actually still ahead - not "
        "just a bars-to-resolution count. additional_upside_before_reversal_pct: how much further above the "
        "event's high price still climbed before a genuine reversal (10% drawdown from a causally-tracked running "
        "peak) was confirmed. remaining_downside_pct: the FULL max drawdown from that running peak across the whole "
        "horizon (not stopped early at the reversal threshold - the true extent of the eventual move). "
        "drawdown_already_realized_pct: how far price had ALREADY fallen from the high by the moment the signal "
        "itself fired.", "",
        "| Threshold | N events | Median upside-before-reversal | Median remaining-downside | Median drawdown already realized at signal | Median bars-to-reversal (min) | Fraction reversal confirmed within horizon |",
        "|---|---|---|---|---|---|---|",
    ]
    for q, s in stats_by_quantile.items():
        lines.append(
            f"| top {1 - q:.0%} | {s['n']} | {_fmt_pct(s['median_upside'])} | {_fmt_pct(s['median_downside'])} "
            f"| {_fmt_pct(s['median_drawdown_at_signal'])} | {_fmt_num(s['median_bars_to_reversal_minutes'], 1)} "
            f"| {_fmt_pct(s['fraction_confirmed'])} |"
        )
    return "\n".join(lines)


def render_symbol_holdout_v2(v1_curve: pd.DataFrame, v2_curve: pd.DataFrame, n_train: int, n_val: int, n_holdout_symbols: int) -> str:
    """Deliverable: symbol_holdout_v2.md (point 10)."""
    lines = [
        "# Symbol Holdout V2", "",
        f"Same unseen-symbol holdout as Phase 3B (train={n_train} events, val={n_val} events, {n_holdout_symbols} "
        "held-out symbols never seen in training), re-run with V2's feature set - one of the strongest pieces of "
        "evidence in Phase 3B (the model generalized to entirely new tokens, not just later dates for known ones).", "",
        "## V1 (for reference)", "", _render_decile_curve_table(v1_curve), "",
        "## V2", "", _render_decile_curve_table(v2_curve),
    ]
    return "\n".join(lines)


def render_coefficient_stability_v2(stability_df: pd.DataFrame, fold_keys: list, notes: str) -> str:
    """Deliverable: coefficient_stability_v2.md (point 11)."""
    lines = [
        "# Coefficient Stability V2", "",
        "Point 11: same per-fold coefficient check as Phase 3B, on V2's smaller feature set. Expectation: "
        "pullback_depth_atr / rejection_atr / lower_high_strength keep a consistent sign now that the collinear "
        "bars_since_last_new_high has been removed.", "",
        "| Feature | " + " | ".join(f"Fold {k}" for k in fold_keys) + " | Mean | Std | Sign consistent |",
        "|---|" + "---|" * len(fold_keys) + "---|---|---|",
    ]
    for _, row in stability_df.iterrows():
        fold_cells = " | ".join(_fmt_num(row.get(f"fold_{k}"), 3) for k in fold_keys)
        lines.append(f"| {row['feature']} | {fold_cells} | {_fmt_num(row.get('mean_coefficient'), 3)} | {_fmt_num(row.get('std_coefficient'), 3)} | {row.get('sign_consistent')} |")
    lines.append("")
    lines.append(notes)
    return "\n".join(lines)


def render_phase_3b1_summary(
    v2_better_than_v1: str, early_warning_edge: str, confirmation_edge: str, symbol_generalization: str, go_no_go: str,
    criteria_table: list[dict], reasoning: str,
) -> str:
    """Deliverable: phase_3b1_summary.md (point 14's master synthesis)."""
    lines = [
        "# Phase 3B.1 Summary - Clean Baseline + Early-Signal Validation", "",
        f"**V2 BETTER THAN V1: {v2_better_than_v1}**", f"**EARLY WARNING EDGE: {early_warning_edge}**",
        f"**CONFIRMATION EDGE: {confirmation_edge}**", f"**SYMBOL GENERALIZATION: {symbol_generalization}**",
        f"**GO / NO-GO TO BOOSTING: {go_no_go}**", "",
        "## Point 13 GO criteria checklist (all must hold together)", "",
        "| Criterion | Threshold | Observed | Pass |", "|---|---|---|---|",
    ]
    for row in criteria_table:
        lines.append(f"| {row['criterion']} | {row['threshold']} | {row['observed']} | {row['pass']} |")
    lines.append("")
    lines.append("## Reasoning")
    lines.append("")
    lines.append(reasoning)
    lines.append("")
    lines.append("No PnL, no TP/SL, no position sizing. No gradient boosting run in this pass regardless of verdict - a separate explicit go-ahead is required.")
    return "\n".join(lines)
