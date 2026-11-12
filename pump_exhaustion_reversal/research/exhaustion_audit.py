"""Spec sections 17-21: Exhaustion Feature Funnel, Timing, Co-occurrence, and
Score Decomposition audit. Pure read of already-computed EventRecord fields
(exhaustion_component_first_true_bar / exhaustion_component_contribution_at_peak)
- no re-simulation, no trading logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord
from ..config import ExhaustionScoreWeights
from ..strategy.exhaustion_detector import ExhaustionFlags

COMPONENT_NAMES: tuple[str, ...] = tuple(ExhaustionFlags().as_dict())


def hit_rate_by_component(events: list[EventRecord]) -> pd.DataFrame:
    """Spec section 17: for each of the 10 components, what fraction of
    events ever had it fire (first_true_bar is not None) at all during their
    lifecycle - not just at the peak bar."""
    if not events:
        return pd.DataFrame(columns=["component", "n_fired", "hit_rate"])
    n = len(events)
    rows = [
        {
            "component": name,
            "n_fired": sum(1 for e in events if e.exhaustion_component_first_true_bar.get(name) is not None),
        }
        for name in COMPONENT_NAMES
    ]
    df = pd.DataFrame(rows)
    df["hit_rate"] = df["n_fired"] / n
    return df.sort_values("hit_rate", ascending=False).reset_index(drop=True)


def timing_by_component(events: list[EventRecord]) -> pd.DataFrame:
    """Spec section 18 ("time_from_pump_detection"): distribution of
    first_true_bar - the earliest bar (never earlier than the event's own
    local_detect_idx, since the live per-bar loop that captures this starts
    there) where each component was observed true. Effectively "bars since
    detection", clamped at 0 - a component whose underlying condition was
    ALREADY true at detection (e.g. a structure break that happened during
    the pre-detection lookback window) is recorded at bar 0, not its true
    earlier bar; that's the intended semantic for a post-detection timing
    report, but it means this field is NOT the right source for "structural
    break relative to nothing" questions - see EventRecord's separate
    structure_break_bar_local_idx (used by research/structure_audit.py) for
    the true, un-clamped structure-break bar. Computed only among the events
    where the component fired at all."""
    rows = []
    for name in COMPONENT_NAMES:
        bars = [e.exhaustion_component_first_true_bar.get(name) for e in events]
        bars = [float(b) for b in bars if b is not None]
        if not bars:
            rows.append({"component": name, "n": 0, "median_bar": np.nan, "p25_bar": np.nan, "p75_bar": np.nan})
            continue
        s = pd.Series(bars)
        rows.append({"component": name, "n": len(bars), "median_bar": s.median(), "p25_bar": s.quantile(0.25), "p75_bar": s.quantile(0.75)})
    return pd.DataFrame(rows)


def co_occurrence_matrix(events: list[EventRecord]) -> pd.DataFrame:
    """Spec section 19: pairwise co-occurrence - among events where component
    A ever fired (anywhere in its lifecycle, not necessarily the same bar as
    B), what fraction also had component B fire. Diagonal is NaN (not a
    meaningful self-comparison, not a real 0)."""
    matrix = pd.DataFrame(index=list(COMPONENT_NAMES), columns=list(COMPONENT_NAMES), dtype=float)
    if not events:
        return matrix
    fired = pd.DataFrame(
        {name: [e.exhaustion_component_first_true_bar.get(name) is not None for e in events] for name in COMPONENT_NAMES}
    )
    for a in COMPONENT_NAMES:
        denom = int(fired[a].sum())
        for b in COMPONENT_NAMES:
            if a == b:
                continue
            matrix.loc[a, b] = ((fired[a] & fired[b]).sum() / denom) if denom > 0 else np.nan
    return matrix


def signal_count_distribution(events: list[EventRecord]) -> pd.Series:
    """Spec section 19: how many of the 10 components ever fired, per event -
    a 0..10 histogram (index = count, value = number of events)."""
    counts = [
        sum(1 for name in COMPONENT_NAMES if e.exhaustion_component_first_true_bar.get(name) is not None)
        for e in events
    ]
    return pd.Series(counts, dtype=int).value_counts().sort_index()


def score_decomposition(events: list[EventRecord], weights: ExhaustionScoreWeights) -> pd.DataFrame:
    """Spec section 21: average weighted contribution of each component to
    exhaustion_score_peak. Since every component is boolean-based (raw value
    is 0.0, 100.0, or NaN via scoring.boolean_component()), a present
    component's weighted contribution is simply its configured weight when
    firing, 0 when not, and it is EXCLUDED from the average (not counted as
    0) on events where it was genuinely missing/inapplicable at the peak bar
    - consistent with this codebase's renormalization convention in
    strategy/scoring.py."""
    weight_map = weights.model_dump()
    rows = []
    for name in COMPONENT_NAMES:
        contributions = []
        for e in events:
            raw = e.exhaustion_component_contribution_at_peak.get(name)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            contributions.append(raw / 100.0 * weight_map[name])
        rows.append(
            {
                "component": name, "weight": weight_map[name], "n_present": len(contributions),
                "avg_weighted_contribution": float(np.mean(contributions)) if contributions else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("avg_weighted_contribution", ascending=False, na_position="last").reset_index(drop=True)


def render_exhaustion_audit_report(
    hit_rates: pd.DataFrame, timing: pd.DataFrame, co_occurrence: pd.DataFrame,
    signal_counts: pd.Series, decomposition: pd.DataFrame,
) -> str:
    lines = ["# Exhaustion Feature Audit", "", "## Hit rate by component (spec section 17)", ""]
    lines.append("| Component | N fired | Hit rate |")
    lines.append("|---|---|---|")
    for _, row in hit_rates.iterrows():
        lines.append(f"| {row['component']} | {row['n_fired']} | {row['hit_rate']:.1%} |")
    lines.append("")
    lines.append("## Timing: first-true bar within each event's slice, among events where it fired (spec section 18)")
    lines.append("")
    lines.append("| Component | N | Median bar | P25 | P75 |")
    lines.append("|---|---|---|---|---|")
    for _, row in timing.iterrows():
        def _fmt(v: float) -> str:
            return "n/a" if pd.isna(v) else f"{v:.0f}"
        lines.append(f"| {row['component']} | {row['n']} | {_fmt(row['median_bar'])} | {_fmt(row['p25_bar'])} | {_fmt(row['p75_bar'])} |")
    lines.append("")
    lines.append("## Signal count per event (spec section 19)")
    lines.append("")
    for count, n in signal_counts.items():
        lines.append(f"- {count} signals: {n} events")
    lines.append("")
    lines.append("## Score decomposition: average weighted contribution to exhaustion_score_peak (spec section 21)")
    lines.append("")
    lines.append("| Component | Weight | N present | Avg weighted contribution |")
    lines.append("|---|---|---|---|")
    for _, row in decomposition.iterrows():
        avg = "n/a" if pd.isna(row["avg_weighted_contribution"]) else f"{row['avg_weighted_contribution']:.1f}"
        lines.append(f"| {row['component']} | {row['weight']} | {row['n_present']} | {avg} |")
    lines.append("")
    lines.append("## Co-occurrence matrix (spec section 19) - P(row fired -> column also fired)")
    lines.append("")
    lines.append("| | " + " | ".join(co_occurrence.columns) + " |")
    lines.append("|---|" + "---|" * len(co_occurrence.columns))
    for idx in co_occurrence.index:
        cells = ["-" if pd.isna(co_occurrence.loc[idx, col]) else f"{co_occurrence.loc[idx, col]:.0%}" for col in co_occurrence.columns]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
