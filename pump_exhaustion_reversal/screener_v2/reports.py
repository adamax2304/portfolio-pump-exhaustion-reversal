"""Phase 3A deliverables 2, 3, 4, 5, 6, 7 - markdown rendering only (all
computation lives in diagnostics.py / ranking.py). Deliverable 1
(event_snapshot_dataset.parquet) is raw data, saved directly by the
orchestrating script - no render function needed here.
"""

from __future__ import annotations

import pandas as pd


def _fmt_pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.1%}"


def _fmt_num(v, digits: int = 3) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def render_data_quality_report(quality: dict) -> str:
    lines = [
        "# Snapshot Data Quality", "",
        f"N snapshots: {quality.get('n_snapshots', 0)}. N events: {quality.get('n_events', 0)}. "
        f"N symbols: {quality.get('n_symbols', 0)}. N exchanges: {quality.get('n_exchanges', 0)}.",
        "",
    ]
    per_event = quality.get("snapshots_per_event", {})
    if per_event:
        lines.append(
            f"Snapshots per event: mean={_fmt_num(per_event.get('mean'), 1)}, median={_fmt_num(per_event.get('50%'), 1)}, "
            f"min={_fmt_num(per_event.get('min'), 0)}, max={_fmt_num(per_event.get('max'), 0)} - "
            "the reason event weighting (point 3) exists: this spread alone would let a handful of long events dominate any pooled statistic."
        )
        lines.append("")

    fully_missing = quality.get("fully_missing_features", [])
    if fully_missing:
        lines.append(f"**Fully missing features (100% NaN, excluded from decile/effect-size analysis): {', '.join(fully_missing)}**")
        lines.append("")

    lines.append("## Per-feature NaN rate (worst first)")
    lines.append("")
    lines.append("| Feature | NaN rate |")
    lines.append("|---|---|")
    for feature, rate in quality.get("feature_nan_rates", {}).items():
        lines.append(f"| {feature} | {_fmt_pct(rate)} |")
    return "\n".join(lines)


def render_label_distribution_report(distribution_df: pd.DataFrame) -> str:
    lines = [
        "# Label Distribution", "",
        "Event-weighted vs snapshot-weighted rates side by side - point 3's long-event-domination check made visible. "
        "A family where these diverge a lot is being pulled around by a handful of long-lived events.", "",
        "| Family | N snapshots | N events | P(reversal) snap-w | P(reversal) evt-w | P(continuation) snap-w | P(continuation) evt-w | P(neutral) snap-w | P(neutral) evt-w |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in distribution_df.iterrows():
        lines.append(
            f"| {row['label_family']} | {row['n_snapshots']} | {row['n_events']} "
            f"| {_fmt_pct(row.get('snapshot_weighted_p_reversal'))} | {_fmt_pct(row.get('event_weighted_p_reversal'))} "
            f"| {_fmt_pct(row.get('snapshot_weighted_p_continuation'))} | {_fmt_pct(row.get('event_weighted_p_continuation'))} "
            f"| {_fmt_pct(row.get('snapshot_weighted_p_neutral'))} | {_fmt_pct(row.get('event_weighted_p_neutral'))} |"
        )
    return "\n".join(lines)


def render_decile_table(feature: str, decile_df: pd.DataFrame) -> str:
    lines = [f"### {feature}", "", "| Bin | N snapshots | N events | P(reversal) | P(continuation) | P(neutral) |", "|---|---|---|---|---|---|"]
    if decile_df.empty:
        lines.append("| (insufficient unique values for binning) | - | - | - | - | - |")
        return "\n".join(lines)
    for _, row in decile_df.sort_values("bin_order").iterrows():
        lines.append(f"| {row['bin']} | {row['n_snapshots']} | {row['n_events']} | {_fmt_pct(row['p_reversal'])} | {_fmt_pct(row['p_continuation'])} | {_fmt_pct(row['p_neutral'])} |")
    return "\n".join(lines)


def render_feature_decile_analysis(decile_tables: dict[str, pd.DataFrame], title: str = "Feature Decile Analysis") -> str:
    lines = [f"# {title}", "", "Looking for MONOTONIC trends across bins, not a single winning bucket - see each table's own P(reversal) column top-to-bottom.", ""]
    for feature, decile_df in decile_tables.items():
        lines.append(render_decile_table(feature, decile_df))
        lines.append("")
    return "\n".join(lines)


def render_effect_size_table(effect_sizes: list[dict]) -> str:
    lines = [
        "| Feature | N | Cliff's delta | Mann-Whitney p | Reversal lift (top/bottom tercile) | Continuation suppression | Top-tercile P(reversal) | 95% CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(effect_sizes, key=lambda x: -abs(x.get("cliffs_delta") or 0) if pd.notna(x.get("cliffs_delta")) else 0):
        if not r.get("valid"):
            continue
        ci = f"[{_fmt_pct(r.get('top_tercile_reversal_rate_ci_low'))}, {_fmt_pct(r.get('top_tercile_reversal_rate_ci_high'))}]"
        lines.append(
            f"| {r['feature']} | {r['n']} | {_fmt_num(r.get('cliffs_delta'))} | {_fmt_num(r.get('mann_whitney_p'))} "
            f"| {_fmt_num(r.get('reversal_lift_top_vs_bottom_tercile'), 2)} | {_fmt_num(r.get('continuation_suppression_top_vs_bottom_tercile'), 2)} "
            f"| {_fmt_pct(r.get('top_tercile_reversal_rate'))} | {ci} |"
        )
    return "\n".join(lines)


def render_pump_dynamics_feature_study(effect_sizes: list[dict], decile_tables: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Pump Dynamics Feature Study (Priority 1)", "",
        "The new hypotheses this phase is built around: pump DECELERATION, high-extension exhaustion, deepening "
        "pullbacks/weakening recoveries, declining participation efficiency, and failed continuation attempts - "
        "not RSI/structure-break/score>X levels. This report covers ONLY the pump_dynamics.py feature set.", "",
        "## Effect sizes (REVERSAL vs CONTINUATION snapshots)", "",
        render_effect_size_table(effect_sizes), "",
        "## Decile tables", "",
    ]
    for feature, decile_df in decile_tables.items():
        lines.append(render_decile_table(feature, decile_df))
        lines.append("")
    return "\n".join(lines)


def render_feature_stability_report(stability_by_feature: dict[str, tuple[pd.DataFrame, dict]]) -> str:
    lines = ["# Feature Stability", "", "Point 25: same effect-size check, split by period/symbol/exchange/regime - looking for features whose EFFECT DIRECTION survives across slices, not features that only work in one slice.", ""]
    for feature, (stability_df, verdict) in stability_by_feature.items():
        lines.append(f"## {feature}")
        lines.append("")
        lines.append(f"Stable (>=70% of slices agree on direction): **{verdict['stable']}** ({_fmt_pct(verdict['fraction_same_direction'])} of {verdict['n_slices']} slices agree).")
        lines.append("")
        if not stability_df.empty:
            lines.append("| Dimension | Value | N | Cliff's delta | Mann-Whitney p |")
            lines.append("|---|---|---|---|---|")
            for _, row in stability_df.iterrows():
                if not row.get("valid"):
                    continue
                lines.append(f"| {row['dimension']} | {row['value']} | {row['n']} | {_fmt_num(row.get('cliffs_delta'))} | {_fmt_num(row.get('mann_whitney_p'))} |")
        lines.append("")
    return "\n".join(lines)


def render_effect_size_table_directional(effect_sizes: list[dict]) -> str:
    """Directional counterpart to render_effect_size_table: reads
    directional_effect_size_report's field names (n_resolved,
    directional_lift_top_vs_bottom_tercile, top_tercile_reversal_share) -
    the RESOLVED-only population, never the raw population effect_size_report uses."""
    lines = [
        "| Feature | N resolved | Cliff's delta | Mann-Whitney p | Directional lift (top/bottom tercile) | Continuation suppression | Top-tercile P(reversal\\|resolved) | 95% CI | Monotonic |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(effect_sizes, key=lambda x: -abs(x.get("cliffs_delta") or 0) if pd.notna(x.get("cliffs_delta")) else 0):
        if not r.get("valid"):
            lines.append(f"| {r.get('feature', '?')} | {r.get('n_resolved', 0)} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        ci = f"[{_fmt_pct(r.get('top_tercile_reversal_share_ci_low'))}, {_fmt_pct(r.get('top_tercile_reversal_share_ci_high'))}]"
        lines.append(
            f"| {r['feature']} | {r['n_resolved']} | {_fmt_num(r.get('cliffs_delta'))} | {_fmt_num(r.get('mann_whitney_p'))} "
            f"| {_fmt_num(r.get('directional_lift_top_vs_bottom_tercile'), 2)} | {_fmt_num(r.get('continuation_suppression_top_vs_bottom_tercile'), 2)} "
            f"| {_fmt_pct(r.get('top_tercile_reversal_share'))} | {ci} | {r.get('monotonic')} |"
        )
    return "\n".join(lines)


def render_directional_vs_resolution_report(resolution_effects: dict[str, dict], directional_effects: dict[str, dict], decile_tables: dict[str, pd.DataFrame]) -> str:
    """Deliverable: directional_vs_resolution_analysis.md (Phase 3A.1 points 1-3)."""
    lines = [
        "# Directional vs Resolution Power", "",
        "Two separate questions, never mixed (points 1-2): RESOLUTION POWER (does the feature predict "
        "P(REVERSAL or CONTINUATION) vs NEUTRAL - i.e. \"something is about to happen\") vs DIRECTION POWER "
        "(among snapshots that DID resolve, does the feature favor REVERSAL over CONTINUATION specifically). "
        "A feature can be strong on one and weak on the other - Phase 3A's own top_candidate_features.md ranking "
        "used raw reversal lift, which conflates them. This is the re-ranking basis point 3 asks for.", "",
        "| Feature | Resolution Cliff's delta | Resolution p | Direction Cliff's delta | Direction p | Directional lift (top/bottom tercile) | Monotonic (direction) |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in resolution_effects:
        res, dirn = resolution_effects[f], directional_effects.get(f, {})
        res_delta = res.get("resolution_cliffs_delta") if res.get("valid") else None
        res_p = res.get("resolution_mann_whitney_p") if res.get("valid") else None
        dir_delta = dirn.get("cliffs_delta") if dirn.get("valid") else None
        dir_p = dirn.get("mann_whitney_p") if dirn.get("valid") else None
        lines.append(
            f"| {f} | {_fmt_num(res_delta)} | {_fmt_num(res_p)} | {_fmt_num(dir_delta)} | {_fmt_num(dir_p)} "
            f"| {_fmt_num(dirn.get('directional_lift_top_vs_bottom_tercile'), 2)} | {dirn.get('monotonic')} |"
        )
    lines.append("")
    lines.append("## Per-feature resolved-conditional decile tables")
    lines.append("")
    lines.append(
        "Point 1's exact requested format: N_resolved, P(reversal|resolved), P(continuation|resolved), "
        "reversal_to_continuation_ratio per bin, alongside the raw (unconditional) P(reversal)/P(continuation)/"
        "P(neutral) for comparison. A feature whose raw P(reversal) rises across bins while "
        "P(reversal|resolved) stays flat is resolution-power-only - exactly the concern raised about "
        "realized_vol_1h / volume_current_over_climax."
    )
    lines.append("")
    for f, decile_df in decile_tables.items():
        lines.append(f"### {f}")
        lines.append("")
        if decile_df.empty:
            lines.append("(insufficient unique values for binning)")
            lines.append("")
            continue
        lines.append("| Bin | N | N resolved | P(reversal) raw | P(reversal\\|resolved) | P(continuation\\|resolved) | Reversal:Continuation ratio |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in decile_df.sort_values("bin_order").iterrows():
            lines.append(
                f"| {row['bin']} | {row['n_snapshots']} | {row.get('n_resolved', 0)} | {_fmt_pct(row['p_reversal'])} "
                f"| {_fmt_pct(row.get('p_reversal_given_resolved'))} | {_fmt_pct(row.get('p_continuation_given_resolved'))} "
                f"| {_fmt_num(row.get('reversal_to_continuation_ratio'), 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_high_state_early_warning_report(censored_effects: dict[str, dict[float, dict]], high_state_effects: dict[str, dict], censor_thresholds: tuple[float, ...]) -> str:
    """Deliverable: high_state_early_warning_analysis.md (Phase 3A.1 points 5-6)."""
    lines = [
        "# High-State / Early-Warning Analysis", "",
        "Point 5: restricts to snapshots where price is STILL within a fixed threshold of the event's own running "
        "high - does a feature survive BEFORE any real reversal has visibly started? All three thresholds reported "
        "side by side, no \"best\" one picked. Point 6: the high-state cohort (price within the top 10% of the "
        "event's own range so far) - if a feature's directional edge survives on JUST this cohort, that is much "
        "stronger evidence than a lift that only shows up after price has already fallen a long way.", "",
        "## Pre-reversal censoring (directional Cliff's delta among RESOLVED snapshots)", "",
        "| Feature | " + " | ".join(f"<= {t:.0%} drawdown" for t in censor_thresholds) + " |",
        "|---|" + "---|" * len(censor_thresholds),
    ]
    for f, by_threshold in censored_effects.items():
        cells = []
        for t in censor_thresholds:
            r = by_threshold.get(t, {})
            cells.append(_fmt_num(r.get("cliffs_delta")) if r.get("valid") else "n/a")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## High-state cohort (price in top 10% of event's own range)")
    lines.append("")
    lines.append("| Feature | N resolved | Directional Cliff's delta | Direction p | Monotonic |")
    lines.append("|---|---|---|---|---|")
    for f, r in high_state_effects.items():
        if not r.get("valid"):
            lines.append(f"| {f} | {r.get('n_resolved', 0)} | n/a | n/a | n/a |")
            continue
        lines.append(f"| {f} | {r['n_resolved']} | {_fmt_num(r.get('cliffs_delta'))} | {_fmt_num(r.get('mann_whitney_p'))} | {r.get('monotonic')} |")
    return "\n".join(lines)


def render_feature_group_report(title: str, subtitle: str, directional_effects: dict[str, dict], decile_tables: dict[str, pd.DataFrame]) -> str:
    """Shared renderer for the two priority feature-group deliverables
    (pullback_deterioration_analysis.md, failed_continuation_efficiency.md) -
    directional power only; see directional_vs_resolution_analysis.md for
    the resolution-power comparison on the same features."""
    lines = [f"# {title}", "", subtitle, "", render_effect_size_table_directional(list(directional_effects.values())), "", "## Decile tables (resolved-conditional)", ""]
    for f, decile_df in decile_tables.items():
        lines.append(render_decile_table(f, decile_df))
        lines.append("")
    return "\n".join(lines)


def _render_grid_table(grid_df: pd.DataFrame) -> str:
    if grid_df.empty:
        return "(no data)"
    dim_cols = [c for c in grid_df.columns if c not in ("n_snapshots", "n_resolved", "low_n", "p_resolved", "p_reversal_given_resolved")]
    lines = ["| " + " | ".join(dim_cols) + " | N | N resolved | P(resolved) | P(reversal\\|resolved) | Low N |", "|" + "---|" * (len(dim_cols) + 5)]
    for _, row in grid_df.iterrows():
        dim_vals = " | ".join(str(row[c]) for c in dim_cols)
        lines.append(f"| {dim_vals} | {row['n_snapshots']} | {row['n_resolved']} | {_fmt_pct(row['p_resolved'])} | {_fmt_pct(row['p_reversal_given_resolved'])} | {row['low_n']} |")
    return "\n".join(lines)


def render_oi_regime_interactions_report(oi_regime_breakdown: pd.DataFrame, interaction_results: dict[str, pd.DataFrame]) -> str:
    """Deliverable: oi_regime_interactions.md (Phase 3A.1 points 10-11)."""
    lines = [
        "# OI Regime Interactions", "",
        "Point 10: oi_change_60m bucketed into 5 pre-registered regimes (thresholds fixed before looking at any "
        "result - never linearized, since Phase 3A found it U-shaped: both strong growth AND strong decline showed "
        "elevated reversal rates vs a flat middle). Point 11: a handful of PRE-SPECIFIED 2D interaction grids, "
        "never an exhaustive combinatorial search.", "",
        "## OI regime breakdown", "",
        _render_grid_table(oi_regime_breakdown), "",
    ]
    for name, grid_df in interaction_results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(_render_grid_table(grid_df))
        lines.append("")
    return "\n".join(lines)


def render_cross_exchange_exhaustion_report(coverage: dict, directional_effects: dict[str, dict], new_high_only_grid: pd.DataFrame, leader_lagger_grid: pd.DataFrame) -> str:
    """Deliverable: cross_exchange_exhaustion.md (Phase 3A.1 point 12, user's #3 priority)."""
    total = coverage.get("n_snapshots_total", 0)
    with_other = coverage.get("n_snapshots_with_other_exchange", 0)
    lines = [
        "# Cross-Exchange Exhaustion (Priority #3)", "",
        f"Snapshots with a matched cross-exchange counterpart: {with_other} / {total} "
        f"({with_other / total:.1%} of all snapshots)" + (f", from {coverage.get('n_linked_underlying_events', 0)} linked (both-present) underlying events." if total else "."),
        "Most events are expected to be solo (single-exchange) - Phase 3A's cross_exchange_audit already found this "
        "is the majority case given the Binance/Bybit symbol universe mismatch; this is the disclosed expected "
        "coverage limit, not a bug.", "",
        "## Divergence features (directional power, among snapshots with a cross-exchange match)", "",
        render_effect_size_table_directional(list(directional_effects.values())), "",
        "## one_exchange_new_high_only (non-confirmation on new highs)", "",
        _render_grid_table(new_high_only_grid), "",
        "## leader_lagger_state", "",
        _render_grid_table(leader_lagger_grid),
    ]
    return "\n".join(lines)


def render_phase_3a1_summary(summary_rows: list[dict], feature_interpretations: dict[str, str]) -> str:
    """Deliverable: phase_3a1_summary.md (Phase 3A.1 points 18/20) - the
    master FEATURE / RESOLUTION POWER / DIRECTION POWER / HIGH-STATE POWER /
    LEAD TIME / STABILITY / MICROSTRUCTURE INTERPRETATION / VERDICT table."""
    lines = [
        "# Phase 3A.1 Summary - Directional Separation Audit", "",
        "Per point 18's criteria: **KEEP** = real directional lift (not just resolution lift), directionally "
        "stable across period/exchange/regime/symbol/liquidity-bucket slices, AND the same-sign effect survives in "
        "the <=5%-drawdown pre-reversal-censored cohort. **WEAK** = directional lift present but fails exactly one "
        "of those. **REJECT** = no real directional lift, or fails both. No feature here has been used to change "
        "any trading logic - this is a candidate list for Phase 3B, not a signal.", "",
        "| Feature | Resolution Δ | Direction Δ | High-state (<=5%) Δ | Lead time (median min, REVERSAL first-entries) | Stable | Verdict | Microstructure interpretation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = {"KEEP": 0, "WEAK": 1, "REJECT": 2}
    sorted_rows = sorted(summary_rows, key=lambda r: order.get(r["verdict"]["verdict"], 3))
    for row in sorted_rows:
        f = row["feature"]
        res_delta = row["resolution"].get("resolution_cliffs_delta") if row["resolution"].get("valid") else None
        dir_delta = row["direction"].get("cliffs_delta") if row["direction"].get("valid") else None
        hs = row.get("high_state") or {}
        hs_delta = hs.get("cliffs_delta") if hs.get("valid") else None
        lt = row.get("lead_time") or {}
        lead_time_str = _fmt_num(lt.get("median_time_to_reversal_minutes"), 1) if lt.get("valid") else "n/a"
        interpretation = feature_interpretations.get(f, "")
        lines.append(
            f"| {f} | {_fmt_num(res_delta)} | {_fmt_num(dir_delta)} | {_fmt_num(hs_delta)} | {lead_time_str} "
            f"| {row['stability'].get('stable')} | **{row['verdict']['verdict']}** | {interpretation} |"
        )
    lines.append("")
    lines.append("## Verdict reasoning detail")
    lines.append("")
    for row in sorted_rows:
        lines.append(f"- **{row['feature']}** ({row['verdict']['verdict']}): {row['verdict']['reason']}")
    return "\n".join(lines)


def render_top_candidate_features(ranked: pd.DataFrame) -> str:
    lines = [
        "# Top Candidate Features", "",
        "Ranked by |Cliff's delta| among features that are BOTH statistically meaningful (top-tercile reversal-rate "
        "bootstrap CI excludes the overall baseline) AND directionally stable across >=70% of the period/symbol/regime "
        "slices tested (feature_stability.md). Being in this list is a candidate for Phase 3B - not a trading signal.", "",
        "| Rank | Feature | Cliff's delta | Reversal lift | Stable | Fraction same direction |",
        "|---|---|---|---|---|---|",
    ]
    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(f"| {i} | {row['feature']} | {_fmt_num(row['cliffs_delta'])} | {_fmt_num(row.get('reversal_lift_top_vs_bottom_tercile'), 2)} | {row['stable']} | {_fmt_pct(row['fraction_same_direction'])} |")
    if len(ranked) == 0:
        lines.append("| (none) | - | - | - | - | - |")
    return "\n".join(lines)
