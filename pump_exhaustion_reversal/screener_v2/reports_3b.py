"""Phase 3B's 10 deliverables - markdown rendering only, all computation
lives in model_features.py / logistic_model.py / walkforward_folds.py /
model_evaluation.py / ranking.py / validation.py. Reuses reports.py's
_fmt_pct/_fmt_num rather than duplicating them.
"""

from __future__ import annotations

import pandas as pd

from .reports import _fmt_num, _fmt_pct


def render_feature_correlation_and_dedup(corr_matrix: pd.DataFrame, dedup_result: dict, feature_set_v1: list[str], known_aliases: dict) -> str:
    """Deliverable: feature_correlation_and_dedup.md (point 1)."""
    lines = [
        "# Feature Correlation and Deduplication", "",
        f"Spearman correlation among Phase 3A.1's KEEP+WEAK candidates, in that priority order - the FIRST feature "
        f"in any cluster with |corr| > {dedup_result['threshold']} against an already-kept feature is retained, later "
        "members of the same cluster are dropped against it. This decides which single representative stands in for "
        "an aliased family (point 1's explicit rule: no feature gets triple-counted).", "",
        "## Correlation matrix", "",
    ]
    cols = list(corr_matrix.columns)
    lines.append("| | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for idx in cols:
        row_vals = " | ".join(_fmt_num(corr_matrix.loc[idx, c], 2) for c in cols)
        lines.append(f"| {idx} | {row_vals} |")
    lines.append("")

    lines.append("## Dedup result")
    lines.append("")
    lines.append(f"Kept ({len(dedup_result['kept'])}): " + ", ".join(dedup_result["kept"]))
    lines.append("")
    if dedup_result["dropped"]:
        lines.append("| Dropped feature | Redundant with | |corr| |")
        lines.append("|---|---|---|")
        for f, info in dedup_result["dropped"].items():
            lines.append(f"| {f} | {info['redundant_with']} | {_fmt_num(info['abs_correlation'], 2)} |")
    else:
        lines.append("(nothing dropped - no pair exceeded the threshold)")
    lines.append("")

    lines.append("## Known alias families (checked against the measured correlation above)")
    lines.append("")
    for canonical, aliases in known_aliases.items():
        for alias in aliases:
            corr_val = corr_matrix.loc[canonical, alias] if canonical in corr_matrix.index and alias in corr_matrix.columns else None
            lines.append(f"- `{alias}` vs canonical `{canonical}`: corr = {_fmt_num(corr_val, 3)}")
    lines.append("")

    lines.append("## FEATURE_SET_V1 (pre-registered, used for the baseline model)")
    lines.append("")
    lines.append(", ".join(f"`{f}`" for f in feature_set_v1))
    consistency = [f for f in feature_set_v1 if f in dedup_result["dropped"]]
    if consistency:
        lines.append("")
        lines.append(f"**Inconsistency flag:** {', '.join(consistency)} appear in FEATURE_SET_V1 but were flagged as redundant by the automatic dedup above - kept anyway per the user's explicit manual specification, disclosed here rather than silently overridden.")
    return "\n".join(lines)


def _render_decile_curve_table(curve_df: pd.DataFrame) -> str:
    lines = ["| Top % | N snapshots | N resolved | N events | N symbols | P(reversal\\|resolved) | Directional lift |", "|---|---|---|---|---|---|---|"]
    for _, row in curve_df.iterrows():
        lines.append(
            f"| {row['top_pct']:.0%} | {row['n_snapshots']} | {row['n_resolved']} | {row['n_events']} | {row['n_unique_symbols']} "
            f"| {_fmt_pct(row['p_reversal_given_resolved'])} | {_fmt_num(row['directional_lift'], 2)} |"
        )
    return "\n".join(lines)


def render_temporal_walkforward_results(fold_summaries: list[dict], symbol_holdout_summary: dict) -> str:
    """Deliverable: temporal_walkforward_results.md (points 7-8). Each fold
    summary: {"fold", "n_train_events", "n_val_events", "curve": DataFrame}."""
    lines = [
        "# Temporal Walk-Forward Results", "",
        "Point 7: expanding-window folds by event (never random, never a snapshot split across train/val within one "
        "event) - train always precedes its own validation slice chronologically, later folds' train sets are "
        "supersets of earlier ones'. Point 8: a separate SYMBOL-holdout fold - entire symbols withheld from training "
        "- checks whether the model generalizes to tokens it has never seen at all, not just to later dates for "
        "already-seen tokens.", "",
    ]
    for fs in fold_summaries:
        lines.append(f"## Temporal fold {fs['fold']} (train={fs['n_train_events']} events, val={fs['n_val_events']} events)")
        lines.append("")
        lines.append(_render_decile_curve_table(fs["curve"]))
        lines.append("")
    lines.append(f"## Symbol-holdout fold (train={symbol_holdout_summary['n_train_events']} events, val={symbol_holdout_summary['n_val_events']} events, {len(symbol_holdout_summary['holdout_symbols'])} held-out symbols)")
    lines.append("")
    lines.append(_render_decile_curve_table(symbol_holdout_summary["curve"]))
    return "\n".join(lines)


def render_score_decile_curve(pooled_curve: pd.DataFrame, per_fold_curves: dict[int, pd.DataFrame]) -> str:
    """Deliverable: score_decile_curve.md (point 11)."""
    lines = [
        "# Score Decile Curve", "",
        "Point 11: pooled out-of-fold validation scores (every row scored ONLY by a model that never saw it in "
        "training) across all temporal folds - the primary ranking-quality readout. Looking for a MONOTONIC curve "
        "(tighter top-X% -> higher lift), not one lucky bucket.", "",
        "## Pooled (all temporal folds)", "",
        _render_decile_curve_table(pooled_curve), "",
        "## Per-fold (consistency check - each fold should show the same rising shape)", "",
    ]
    for fold, curve in per_fold_curves.items():
        lines.append(f"### Fold {fold}")
        lines.append("")
        lines.append(_render_decile_curve_table(curve))
        lines.append("")
    return "\n".join(lines)


def render_high_state_ranking(all_state_on_high_state: pd.DataFrame, high_state_only_on_high_state: pd.DataFrame, n_high_state_val_rows: int, n_total_val_rows: int) -> str:
    """Deliverable: high_state_ranking.md (point 10) - the single most
    important comparison per the user's framing: if the model only works
    once price has already fallen a long way, it is not useful as a
    screener."""
    lines = [
        "# High-State Ranking (Point 10)", "",
        f"Restricted to validation snapshots where price is still within the top 10% of the event's own range so far "
        f"({n_high_state_val_rows} / {n_total_val_rows} = {n_high_state_val_rows / n_total_val_rows:.1%} of all pooled "
        "validation rows) - exactly the population a live screener actually needs to work on, since by the time price "
        "has fallen far it is no longer 'catching' anything.", "",
        "## ALL_STATE model (trained on all states), evaluated ONLY on high-state validation rows", "",
        _render_decile_curve_table(all_state_on_high_state), "",
        "## HIGH_STATE_ONLY model (trained ONLY on high-state rows), evaluated on the SAME high-state validation rows", "",
        _render_decile_curve_table(high_state_only_on_high_state),
    ]
    return "\n".join(lines)


def render_precision_at_k(result: dict) -> str:
    """Deliverable: precision_at_k.md (point 13)."""
    lines = [
        "# Precision@K", "",
        f"Point 13: simulates the screener's real job - at each of {result['n_timestamps_used']} timestamps with >=2 "
        f"genuinely concurrent scored candidates (spanning {result['n_days_span']:.0f} days), ranks by score and "
        "checks the top-K. An 'alert' counts once per (event, K), the first time that event enters the top-K - not "
        "once per bar it stays there.", "",
        "| K | Mean Precision@K (per timestamp) | Unique events alerted | True alerts | False alerts | Precision (unique alerts) | Alerts/day | False alerts/day |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, stats in result["by_k"].items():
        lines.append(
            f"| {k} | {_fmt_pct(stats['mean_precision_at_k_per_timestamp'])} | {stats['n_unique_events_alerted']} "
            f"| {stats['n_true_alerts']} | {stats['n_false_alerts']} | {_fmt_pct(stats['precision_among_unique_alerts'])} "
            f"| {_fmt_num(stats['alerts_per_day'], 2)} | {_fmt_num(stats['false_alerts_per_day'], 2)} |"
        )
    return "\n".join(lines)


def render_lead_time_analysis(results_by_quantile: dict[float, dict]) -> str:
    """Deliverable: lead_time_analysis.md (point 14, on the model SCORE
    rather than a raw feature - reuses ranking.lead_time_analysis exactly,
    including its disclosed scope: remaining-downside / max-additional-
    upside would need re-touching raw OHLCV per event and are not computed
    here, same disclosed simplification as Phase 3A.1's point 17)."""
    lines = [
        "# Lead Time Analysis", "",
        "Point 14: for each entry threshold (top 20%/10%/5% of the model SCORE), the FIRST snapshot per event to "
        "cross it, then how the forward competing-outcome label resolved from that exact bar. A genuine lead time "
        "means REVERSAL, when it happens, happens SOME BARS AFTER the signal fired - not on the same bar the drop "
        "was already visible.", "",
        "| Top quantile | N events signaled | N resolved REVERSAL at first entry | Fraction resolved REVERSAL | Median time to reversal (min) | Fraction already >=5% below high at signal |",
        "|---|---|---|---|---|---|",
    ]
    for q, r in results_by_quantile.items():
        if not r.get("valid"):
            lines.append(f"| top {1 - q:.0%} | {r.get('n_events_with_signal', 0)} | - | - | - | - |")
            continue
        lines.append(
            f"| top {1 - q:.0%} | {r['n_events_with_signal']} | {r['n_first_entries_resolved_reversal']} "
            f"| {_fmt_pct(r['fraction_first_entries_resolved_reversal'])} | {_fmt_num(r.get('median_time_to_reversal_minutes'), 1)} "
            f"| {_fmt_pct(r.get('fraction_already_deep_in_drawdown_at_signal'))} |"
        )
    return "\n".join(lines)


def render_coefficient_stability(stability_df: pd.DataFrame, fold_keys: list) -> str:
    """Deliverable: coefficient_stability.md (point 15)."""
    lines = [
        "# Coefficient Stability", "",
        "Point 15: each temporal fold's OWN fitted coefficient (standardized scale) per feature. A feature whose "
        "sign flips between folds is a bad feature even with a good aggregate fit.", "",
        "| Feature | " + " | ".join(f"Fold {k}" for k in fold_keys) + " | Mean | Std | Sign consistent |",
        "|---|" + "---|" * len(fold_keys) + "---|---|---|",
    ]
    for _, row in stability_df.iterrows():
        fold_cells = " | ".join(_fmt_num(row.get(f"fold_{k}"), 3) for k in fold_keys)
        lines.append(f"| {row['feature']} | {fold_cells} | {_fmt_num(row.get('mean_coefficient'), 3)} | {_fmt_num(row.get('std_coefficient'), 3)} | {row.get('sign_consistent')} |")
    return "\n".join(lines)


def render_feature_ablation(ablation_results: dict[str, dict]) -> str:
    """Deliverable: feature_ablation.md (point 16)."""
    lines = [
        "# Feature Ablation", "",
        "Point 16: four pre-specified variants, marginal lift only - a variant is never preferred just because a "
        "train-set metric improved.", "",
        "| Variant | N features | Top-10% lift | Top-20% lift | N resolved (pooled OOS) |",
        "|---|---|---|---|---|",
    ]
    for name, r in ablation_results.items():
        lines.append(f"| {name} | {r['n_features']} | {_fmt_num(r['top10_lift'], 2)} | {_fmt_num(r['top20_lift'], 2)} | {r['n_resolved']} |")
    return "\n".join(lines)


def render_calibration_and_bootstrap(calibration_df: pd.DataFrame, event_bootstrap: dict, symbol_bootstrap: dict, cross_exchange_note: str | None = None) -> str:
    """Deliverable: calibration_and_bootstrap.md (points 21-22, plus a
    short point-17 cross-exchange diagnostic note)."""
    lines = [
        "# Calibration and Bootstrap", "",
        "Point 21: predicted probability vs actual reversal rate, among resolved pooled-OOS validation snapshots. "
        "Point 22: event-cluster AND symbol-cluster bootstrap 95% CI on the top-10% bucket's reversal rate - the "
        "single number the summary's GO/NO-GO criteria hinges on most.", "",
        "## Calibration", "",
        "| Bin | N | Mean predicted probability | Actual reversal rate |", "|---|---|---|---|",
    ]
    for _, row in calibration_df.iterrows():
        lines.append(f"| {row['bin']} | {row['n']} | {_fmt_pct(row['mean_predicted_probability'])} | {_fmt_pct(row['actual_reversal_rate'])} |")
    lines.append("")
    lines.append("## Top-10% bucket reversal rate - bootstrap 95% CI")
    lines.append("")
    lines.append(f"- Event-cluster bootstrap: {_fmt_pct(event_bootstrap.get('statistic'))} [{_fmt_pct(event_bootstrap.get('ci_low'))}, {_fmt_pct(event_bootstrap.get('ci_high'))}] (n_clusters={event_bootstrap.get('n_clusters')})")
    lines.append(f"- Symbol-cluster bootstrap: {_fmt_pct(symbol_bootstrap.get('statistic'))} [{_fmt_pct(symbol_bootstrap.get('ci_low'))}, {_fmt_pct(symbol_bootstrap.get('ci_high'))}] (n_clusters={symbol_bootstrap.get('n_clusters')})")
    if cross_exchange_note:
        lines.append("")
        lines.append("## Point 17 diagnostic: one_exchange_new_high_only")
        lines.append("")
        lines.append(cross_exchange_note)
    return "\n".join(lines)


def render_phase_3b_summary(
    ranking_edge: str, high_state_edge: str, lead_time_quality: str, go_no_go: str,
    stable_features: list[str], unstable_features: list[str], criteria_table: list[dict], reasoning: str,
) -> str:
    """Deliverable: phase_3b_logistic_summary.md (point 26's master synthesis)."""
    lines = [
        "# Phase 3B Logistic Baseline Summary", "",
        f"**RANKING EDGE: {ranking_edge}**", f"**HIGH-STATE EDGE: {high_state_edge}**",
        f"**LEAD-TIME QUALITY: {lead_time_quality}**", f"**GO / NO-GO TO BOOSTING: {go_no_go}**", "",
        "## Point 20 GO criteria checklist", "",
        "| Criterion | Threshold | Observed | Pass |", "|---|---|---|---|",
    ]
    for row in criteria_table:
        lines.append(f"| {row['criterion']} | {row['threshold']} | {row['observed']} | {row['pass']} |")
    lines.append("")
    lines.append(f"**Stable features** ({len(stable_features)}): " + (", ".join(f"`{f}`" for f in stable_features) if stable_features else "(none)"))
    lines.append("")
    lines.append(f"**Unstable features** ({len(unstable_features)}): " + (", ".join(f"`{f}`" for f in unstable_features) if unstable_features else "(none)"))
    lines.append("")
    lines.append("## Reasoning")
    lines.append("")
    lines.append(reasoning)
    lines.append("")
    lines.append(
        "No PnL, no TP/SL, no position sizing anywhere in this phase (point 25). No gradient boosting run "
        "automatically regardless of this verdict (point 24) - that requires a separate, explicit go-ahead."
    )
    return "\n".join(lines)
