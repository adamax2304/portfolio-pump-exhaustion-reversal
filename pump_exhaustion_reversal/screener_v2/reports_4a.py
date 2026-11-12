"""Phase 4A's 8 markdown deliverables (the 9th, confirmation_candidate_
dataset.parquet, is raw data saved directly by the orchestrator - no render
function needed). All computation lives in entry_families.py /
matched_controls.py / research/entry_edge.py / research/first_passage.py /
strategy/risk_engine.py.
"""

from __future__ import annotations

import pandas as pd

from .reports import _fmt_num, _fmt_pct


def _fmt_r(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:+.3f}R"


def render_expectancy_table(summary_by_barrier: dict[str, dict]) -> str:
    lines = [
        "| Barrier family | N | Win rate | Net EV | Gross EV | Fees | Entry slip | Exit slip | Funding | PF | Avg win | Avg loss | Avg MAE | Avg MFE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary_by_barrier.items():
        if s.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {s['n']} | {_fmt_pct(s.get('win_rate'))} | {_fmt_r(s.get('net_expectancy_r'))} "
            f"| {_fmt_r(s.get('gross_expectancy_r'))} | {_fmt_r(s.get('fees_r'))} | {_fmt_r(s.get('entry_slippage_r'))} "
            f"| {_fmt_r(s.get('exit_slippage_r'))} | {_fmt_r(s.get('funding_r'))} | {_fmt_num(s.get('profit_factor'), 2)} "
            f"| {_fmt_r(s.get('avg_win_r'))} | {_fmt_r(s.get('avg_loss_r'))} | {_fmt_num(s.get('avg_mae_r'), 2)} | {_fmt_num(s.get('avg_mfe_r'), 2)} |"
        )
    return "\n".join(lines)


def render_entry_family_results(
    family_name: str, description: str, n_candidates: int, n_valid_signals: int,
    summary_by_barrier: dict[str, dict], bootstrap_by_barrier: dict[str, dict],
    temporal_by_fold: dict, symbol_holdout: dict, score_bucket_summary: dict[str, dict],
) -> str:
    lines = [
        f"# Entry Family {family_name}", "", description, "",
        f"Candidates (FIRST_TOP10_ALERT_PER_EVENT): {n_candidates}. Valid signals produced by this family's detector: {n_valid_signals} "
        f"({n_valid_signals / n_candidates:.1%} of candidates - the rest never satisfied this family's pattern within the horizon, or the "
        "thesis was invalidated by real continuation).", "",
        "## Net expectancy by barrier family (max 1 attempt per event)", "",
        render_expectancy_table(summary_by_barrier), "",
        "## Bootstrap 95% CI (primary barrier family = STANDARD_2R_4H)", "",
    ]
    for cluster_kind, boot in bootstrap_by_barrier.items():
        if boot.get("n", 0) == 0:
            lines.append(f"- {cluster_kind}: n=0")
            continue
        wr = boot.get("win_rate_ci", {})
        ev = boot.get("net_expectancy_ci", {})
        lines.append(
            f"- {cluster_kind} (n={boot['n']}): win rate {_fmt_pct(wr.get('statistic'))} [{_fmt_pct(wr.get('ci_low'))}, {_fmt_pct(wr.get('ci_high'))}], "
            f"net EV {_fmt_r(ev.get('statistic'))} [{_fmt_r(ev.get('ci_low'))}, {_fmt_r(ev.get('ci_high'))}]"
        )
    lines.append("")
    lines.append("## Temporal fold stability (STANDARD_2R_4H)")
    lines.append("")
    lines.append("| Fold | N | Win rate | Net EV | PF |")
    lines.append("|---|---|---|---|---|")
    for fold, s in temporal_by_fold.items():
        if s.get("n", 0) == 0:
            lines.append(f"| {fold} | 0 | - | - | - |")
            continue
        lines.append(f"| {fold} | {s['n']} | {_fmt_pct(s.get('win_rate'))} | {_fmt_r(s.get('net_expectancy_r'))} | {_fmt_num(s.get('profit_factor'), 2)} |")
    lines.append("")
    lines.append("## Symbol holdout (STANDARD_2R_4H)")
    lines.append("")
    if symbol_holdout.get("n", 0) == 0:
        lines.append("(no valid signals among held-out-symbol candidates)")
    else:
        lines.append(f"N={symbol_holdout['n']}, win rate={_fmt_pct(symbol_holdout.get('win_rate'))}, net EV={_fmt_r(symbol_holdout.get('net_expectancy_r'))}, PF={_fmt_num(symbol_holdout.get('profit_factor'), 2)}")
    lines.append("")
    lines.append("## Score bucket -> trading EV (STANDARD_2R_4H)")
    lines.append("")
    lines.append("| Alert threshold | N | Win rate | Net EV | PF |")
    lines.append("|---|---|---|---|---|")
    for bucket, s in score_bucket_summary.items():
        if s.get("n", 0) == 0:
            lines.append(f"| {bucket} | 0 | - | - | - |")
            continue
        lines.append(f"| {bucket} | {s['n']} | {_fmt_pct(s.get('win_rate'))} | {_fmt_r(s.get('net_expectancy_r'))} | {_fmt_num(s.get('profit_factor'), 2)} |")
    return "\n".join(lines)


def render_tradeability_and_cost(stop_distance_summary: pd.DataFrame, cost_sensitivity_by_family: dict[str, pd.DataFrame], poor_tradeability_note: str) -> str:
    lines = [
        "# Tradeability and Cost Analysis", "",
        "Point 10-11: stop distance vs execution cost, kept SEPARATE from the ranking probability - a wide stop is "
        "not a reason to distrust V2's ranking, only a reason to flag that particular trade as hard to size well. "
        "Diagnostic only in this pass, not a new hard filter.", "",
        "## Stop distance by entry family", "",
        "| Family | N | Stop bps P10 | P25 | P50 | P75 | P90 | Round-trip cost / stop distance (median) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in stop_distance_summary.iterrows():
        lines.append(
            f"| {row['family']} | {row['n']} | {_fmt_num(row.get('stop_distance_bps_p10'), 0)} | {_fmt_num(row.get('stop_distance_bps_p25'), 0)} "
            f"| {_fmt_num(row.get('stop_distance_bps_p50'), 0)} | {_fmt_num(row.get('stop_distance_bps_p75'), 0)} | {_fmt_num(row.get('stop_distance_bps_p90'), 0)} "
            f"| {_fmt_pct(row.get('cost_to_stop_ratio_median'))} |"
        )
    lines.append("")
    lines.append(poor_tradeability_note)
    lines.append("")
    lines.append("## Cost sensitivity sweep (STANDARD_2R_4H, ZERO/LOW/BASE/HIGH cost profiles)")
    lines.append("")
    for family, df in cost_sensitivity_by_family.items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Profile | Net EV | N |")
        lines.append("|---|---|---|")
        for _, row in df.iterrows():
            lines.append(f"| {row['profile']} | {_fmt_r(row.get('net_expectancy_r'))} | {row.get('n', 0)} |")
        lines.append("")
    return "\n".join(lines)


def render_rank_vs_trade_ev(bucket_ev_by_family: dict[str, dict[str, dict]]) -> str:
    lines = [
        "# Rank vs Trade EV", "",
        "Point 16: does a higher V2 rank translate into a better trading outcome, not just a better classification "
        "metric? Looking for higher rank -> better net EV; if that does not hold, ranking probability is not "
        "translating into trading edge even where the entry mechanics themselves work.", "",
    ]
    for family, buckets in bucket_ev_by_family.items():
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| Alert threshold | N | Net EV | PF | Avg MAE | Avg MFE |")
        lines.append("|---|---|---|---|---|---|")
        for bucket, s in buckets.items():
            if s.get("n", 0) == 0:
                lines.append(f"| {bucket} | 0 | - | - | - | - |")
                continue
            lines.append(f"| {bucket} | {s['n']} | {_fmt_r(s.get('net_expectancy_r'))} | {_fmt_num(s.get('profit_factor'), 2)} | {_fmt_num(s.get('avg_mae_r'), 2)} | {_fmt_num(s.get('avg_mfe_r'), 2)} |")
        lines.append("")
    return "\n".join(lines)


def render_matched_control_analysis(candidate_summary: dict, control_summary: dict, family_name: str) -> str:
    lines = [
        "# Matched Control Analysis", "",
        "Point 15: the SAME entry algorithm run on control events matched by period/liquidity/magnitude/volatility "
        "but NOT top-10-ranked by V2. If entry mechanics alone are profitable on any similar falling pump, V2's "
        "ranking adds no value beyond the entry mechanic itself.", "",
        f"## {family_name} (STANDARD_2R_4H)", "",
        "| Cohort | N | Win rate | Net EV | PF |", "|---|---|---|---|---|",
        f"| V2 top-10 candidates | {candidate_summary.get('n', 0)} | {_fmt_pct(candidate_summary.get('win_rate'))} | {_fmt_r(candidate_summary.get('net_expectancy_r'))} | {_fmt_num(candidate_summary.get('profit_factor'), 2)} |",
        f"| Matched controls | {control_summary.get('n', 0)} | {_fmt_pct(control_summary.get('win_rate'))} | {_fmt_r(control_summary.get('net_expectancy_r'))} | {_fmt_num(control_summary.get('profit_factor'), 2)} |",
    ]
    return "\n".join(lines)


def render_temporal_symbol_robustness(by_family: dict[str, dict]) -> str:
    lines = ["# Temporal / Symbol Robustness (all families, STANDARD_2R_4H)", ""]
    for family, data in by_family.items():
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| Fold | N | Net EV | PF |")
        lines.append("|---|---|---|---|")
        for fold, s in data["temporal"].items():
            lines.append(f"| {fold} | {s.get('n', 0)} | {_fmt_r(s.get('net_expectancy_r'))} | {_fmt_num(s.get('profit_factor'), 2)} |")
        sh = data["symbol_holdout"]
        lines.append(f"| symbol_holdout | {sh.get('n', 0)} | {_fmt_r(sh.get('net_expectancy_r'))} | {_fmt_num(sh.get('profit_factor'), 2)} |")
        lines.append("")
    return "\n".join(lines)


def render_phase_4a_summary(
    best_family: str | None, outcome_label: str, go_no_go: str, per_family_headline: dict[str, dict],
    criteria_table: list[dict], reasoning: str,
) -> str:
    lines = [
        "# Phase 4A Summary - Ranking to Trading Edge", "",
        f"**BEST FAMILY: {best_family or 'none'}**", f"**OUTCOME: {outcome_label}**", f"**GO / NO-GO: {go_no_go}**", "",
        "## Entry family comparison (STANDARD_2R_4H, primary top-10 cohort)", "",
        "| Entry | N | Net EV | PF | WR | Avg win | Avg loss | Avg MAE | Avg MFE | Median entry drawdown-from-high |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for family, h in per_family_headline.items():
        if h.get("n", 0) == 0:
            lines.append(f"| {family} | 0 | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {family} | {h['n']} | {_fmt_r(h.get('net_expectancy_r'))} | {_fmt_num(h.get('profit_factor'), 2)} | {_fmt_pct(h.get('win_rate'))} "
            f"| {_fmt_r(h.get('avg_win_r'))} | {_fmt_r(h.get('avg_loss_r'))} | {_fmt_num(h.get('avg_mae_r'), 2)} | {_fmt_num(h.get('avg_mfe_r'), 2)} "
            f"| {_fmt_pct(h.get('median_entry_drawdown'))} |"
        )
    lines.append("")
    lines.append("## Point 26 GO criteria checklist")
    lines.append("")
    lines.append("| Criterion | Observed | Pass |")
    lines.append("|---|---|---|")
    for row in criteria_table:
        lines.append(f"| {row['criterion']} | {row['observed']} | {row['pass']} |")
    lines.append("")
    lines.append("## Reasoning")
    lines.append("")
    lines.append(reasoning)
    lines.append("")
    lines.append("No live scanner built. No real money. Forward sample after 2026-08-27 untouched.")
    return "\n".join(lines)
