"""Spec sections 65/84: Research Summary - the final executive-conclusion
document. This module aggregates the KEY quantitative findings from every
other phase into one structured overview. The qualitative synthesis (is
something working / broken / merely under-tested) is deliberately left to a
human reviewing these numbers - see the project plan's explicit "do not make
unilateral strategy-change decisions" instruction (spec sections 64/83).
This module states facts pulled from the other modules' own already-computed
outputs, it does not render a verdict.
"""

from __future__ import annotations

import pandas as pd


def render_research_summary(
    exchange_event_counts: dict[str, int],
    class_distributions: dict[str, pd.DataFrame],
    oi_hypothesis: dict[str, dict],
    new_high_timing: dict,
    exhaustion_hit_rates: pd.DataFrame,
    false_negative_summary: dict,
    true_negative_summary: dict,
    logit_result: dict,
) -> str:
    lines = ["# Research Summary", "", "## Dataset", ""]
    for exchange, n in exchange_event_counts.items():
        lines.append(f"- {exchange}: {n} events")
    lines.append("")
    lines.append("## Key numbers by area")
    lines.append("")
    lines.append("### 1. Pump Detector / Classifier")
    for exchange, dist in class_distributions.items():
        if dist is None or dist.empty:
            continue
        top = dist.iloc[0]
        lines.append(f"- {exchange}: top class = {top['pump_class']} ({top['share']:.0%} of events, n={top['n']})")
    for exchange, bias in oi_hypothesis.items():
        if not bias:
            continue
        lines.append(
            f"- {exchange}: SPOT_ORGANIC share is {bias['spot_organic_share_no_signal']:.0%} of events with NO "
            f"OI/liquidation signal at all (n={bias['n_no_oi_or_liq_signal']}) vs "
            f"{bias['spot_organic_share_has_signal']:.0%} of events WITH at least one such signal "
            f"(n={bias['n_has_oi_or_liq_signal']})"
        )
    lines.append("")
    lines.append("### 2. Structure Detector")
    if new_high_timing.get("n_matched", 0) > 0:
        lines.append(
            f"- new_pump_high invalidation: median {new_high_timing['p50_bars']:.1f} bars after structure break "
            f"(n={new_high_timing['n_matched']}, excluded for missing index={new_high_timing['n_excluded_missing_index']}), "
            f"{new_high_timing['within_3_bars']} within 3 bars, {new_high_timing['within_12_bars']} within 12 bars"
        )
    else:
        lines.append("- new_pump_high timing: no matched events with both bar indices present")
    lines.append("")
    lines.append("### 3. Exhaustion Features")
    if exhaustion_hit_rates is not None and not exhaustion_hit_rates.empty:
        ranked = exhaustion_hit_rates.sort_values("hit_rate", ascending=False)
        lines.append("- most common: " + ", ".join(f"{r['component']} ({r['hit_rate']:.0%})" for _, r in ranked.head(3).iterrows()))
        lines.append("- least common: " + ", ".join(f"{r['component']} ({r['hit_rate']:.0%})" for _, r in ranked.tail(3).iterrows()))
    lines.append("")
    lines.append("### 4. False Negatives / True Negatives")
    lines.append(f"- False negatives (real >=20% reversal within 24h, no trade taken): n={false_negative_summary.get('n', 0)}")
    lines.append(f"- True negatives (no trade, pump correctly continued >=20% further): n={true_negative_summary.get('n', 0)}")
    lines.append("")
    lines.append("### 5. Logistic regression confirmatory check")
    if logit_result.get("sufficient_n") and "coefficients" in logit_result:
        lines.append(
            f"- n={logit_result['n']} (train={logit_result.get('n_train')}, OOS={logit_result.get('n_oos')}), "
            f"OOS accuracy={logit_result['oos_accuracy']:.0%}, McFadden pseudo-R2={logit_result['pseudo_r2_mcfadden']:.3f}"
        )
    else:
        lines.append(f"- not reportable: {logit_result.get('note', 'insufficient data')}")
    lines.append("")
    lines.append("## Acceptance criteria (spec section 66)")
    lines.append("")
    lines.append("- A (repeatable pump cohorts with above-baseline reversal probability): see forward_event_study.md and feature_predictive_value.md")
    lines.append("- B (observable pre-reversal features distinguishing reversal from continuation): see exhaustion_feature_audit.md and feature_predictive_value.md")
    lines.append(
        "- C (those features available without future leakage): yes by construction - the research/ package "
        "(forward_study.py, negative_analysis.py, predictive_value.py, dataset_builder.py, charts.py) is physically "
        "separate from backtest/simulator.py and strategy/*, and is never imported by either"
    )
    lines.append("")
    lines.append(
        "This document reports numbers pulled from the other reports, it does not render a verdict. The A/B/C "
        "judgment call and any per-feature GO/INVESTIGATE/REJECT decision (spec section 63) is for human review "
        "before any change to Strategy v1.1 - see project plan."
    )
    return "\n".join(lines)
