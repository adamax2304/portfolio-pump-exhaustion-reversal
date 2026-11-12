"""Spec sections 25/38/39/40: Feature Predictive Value. Counterfactual
conditional-probability study (P(bad outcome | feature) vs P(bad outcome |
not feature)) for each exhaustion component individually and for the
specific combinations spec section 25 lists, plus a labeled
reversal-vs-continuation effect-size comparison (spec section 37/38) and an
optional logistic-regression confirmatory check (spec section 39) with a
strict temporal train/OOS split (spec section 40) - never a random/shuffled
split, since these are time-ordered market events.

Everything here is diagnostic-only: it measures which already-computed
signals correlate with a REAL (forward-looking, research-only) outcome. It
does not feed anything back into strategy/entry_engine.py or change any
score/threshold, and per the project plan's explicit scope decision it uses
statsmodels.Logit only - no random forest / gradient boosting, since at this
project's event-count scale (low hundreds) a tree ensemble would reliably
overfit and produce a meaningless feature-importance ranking.

Expects a forward_paths_df matching research/forward_study.py's
compute_forward_paths_for_events() output (columns including
maximum_additional_upside, drawdown_from_eventual_peak_{1h,4h,8h,24h}).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord
from ..strategy.exhaustion_detector import ExhaustionFlags
from . import stats as research_stats

COMPONENT_NAMES: tuple[str, ...] = tuple(ExhaustionFlags().as_dict())

REVERSAL_THRESHOLDS = {"REVERSAL_10": 0.10, "REVERSAL_20": 0.20}
CONTINUATION_THRESHOLDS = {"CONTINUATION_20": 0.20, "CONTINUATION_50": 0.50}

# Spec section 25's explicit combination list.
FEATURE_COMBINATIONS: list[tuple[str, ...]] = [
    ("cvd_bearish_divergence", "oi_rollover"),
    ("cvd_bearish_divergence", "structure_break"),
    ("oi_rollover", "structure_break"),
    ("volume_decay", "cvd_bearish_divergence"),
    ("sweep_rejection", "structure_break"),
    ("structure_break", "failed_reclaim_lower_high"),
    ("cvd_bearish_divergence", "oi_rollover", "structure_break"),
    ("cvd_bearish_divergence", "oi_rollover", "structure_break", "failed_reclaim_lower_high"),
]

MIN_N_FOR_LOGIT = 40


def _fired(event: EventRecord, name: str) -> bool:
    return event.exhaustion_component_first_true_bar.get(name) is not None


def build_feature_outcome_frame(events: list[EventRecord], forward_paths_df: pd.DataFrame) -> pd.DataFrame:
    """One row per event (in forward_paths_df's row order - callers wanting a
    temporal split for logistic_regression_check must pass an already
    chronologically-sorted forward_paths_df): a boolean column per exhaustion
    component (whether it ever fired anywhere in the event's lifecycle) plus
    every outcome column from forward_paths_df, joined by event_id."""
    events_by_id = {e.event_id: e for e in events}
    rows = []
    for _, fp_row in forward_paths_df.iterrows():
        event = events_by_id.get(fp_row["event_id"])
        if event is None:
            continue
        row = {name: _fired(event, name) for name in COMPONENT_NAMES}
        row["event_id"] = fp_row["event_id"]
        row["pump_class"] = event.pump_class
        row["pump_score"] = event.pump_score
        row["exhaustion_score_peak"] = event.exhaustion_score_peak
        for col in fp_row.index:
            if col not in row:
                row[col] = fp_row[col]
        rows.append(row)
    return pd.DataFrame(rows)


def single_feature_counterfactual(frame: pd.DataFrame, outcome_col: str, outcome_threshold: float) -> pd.DataFrame:
    """For each of the 10 components: P(frame[outcome_col] >= threshold | fired) vs P(... | not fired)."""
    outcome = frame[outcome_col] >= outcome_threshold
    rows = []
    for name in COMPONENT_NAMES:
        result = research_stats.conditional_probability(frame[name].astype(bool), outcome)
        result["feature"] = name
        rows.append(result)
    return pd.DataFrame(rows)[["feature", "n_with_condition", "n_without_condition", "p_outcome_given_condition", "p_outcome_given_not_condition"]]


def combination_counterfactual(frame: pd.DataFrame, outcome_col: str, outcome_threshold: float) -> pd.DataFrame:
    """Same as single_feature_counterfactual but for spec section 25's
    specific feature combinations - ALL listed features must have fired for
    "condition" to be True (a conjunction, not "any of")."""
    outcome = frame[outcome_col] >= outcome_threshold
    rows = []
    for combo in FEATURE_COMBINATIONS:
        condition = frame[list(combo)].all(axis=1)
        result = research_stats.conditional_probability(condition, outcome)
        result["combination"] = " + ".join(combo)
        rows.append(result)
    return pd.DataFrame(rows)[["combination", "n_with_condition", "n_without_condition", "p_outcome_given_condition", "p_outcome_given_not_condition"]]


def label_reversal_continuation(frame: pd.DataFrame) -> pd.DataFrame:
    """Spec section 37: research-only labels added as new boolean columns.
    REVERSAL_10/REVERSAL_20 come from drawdown_from_eventual_peak_24h,
    CONTINUATION_20/CONTINUATION_50 from maximum_additional_upside. The two
    label families are NOT mutually exclusive by construction (they answer
    "did it eventually reverse" and "did it also run further first" about the
    same event) - a row can be both, and that is correct, not a bug."""
    out = frame.copy()
    for label, threshold in REVERSAL_THRESHOLDS.items():
        out[label] = out["drawdown_from_eventual_peak_24h"] >= threshold
    for label, threshold in CONTINUATION_THRESHOLDS.items():
        out[label] = out["maximum_additional_upside"] >= threshold
    return out


def compare_reversal_vs_continuation(
    labeled_frame: pd.DataFrame, numeric_col: str, reversal_label: str = "REVERSAL_20", continuation_label: str = "CONTINUATION_20"
) -> dict:
    """Spec section 38 effect-size comparison of a numeric column (e.g.
    pump_score, exhaustion_score_peak) between the reversal_label group and
    the continuation_label group."""
    reversal_group = labeled_frame.loc[labeled_frame[reversal_label], numeric_col]
    continuation_group = labeled_frame.loc[labeled_frame[continuation_label], numeric_col]
    return research_stats.effect_size_report(reversal_group, continuation_group, label_a=reversal_label, label_b=continuation_label)


def logistic_regression_check(labeled_frame: pd.DataFrame, target_col: str = "REVERSAL_20", train_frac: float = 0.7) -> dict:
    """Spec sections 39/40: one confirmatory statsmodels.Logit fit on the 10
    boolean exhaustion components, with a strict TEMPORAL split (first
    train_frac rows = train, remainder = OOS - `labeled_frame` must already
    be in chronological event order; this function does not sort it, so an
    unsorted input silently produces a meaningless split - that's the
    caller's responsibility, documented here rather than guessed at
    silently). Always framed as directional/hypothesis-generating, never a
    production decision rule."""
    import statsmodels.api as sm

    n = len(labeled_frame)
    if n < MIN_N_FOR_LOGIT:
        return {"n": n, "sufficient_n": False, "note": f"n={n} < {MIN_N_FOR_LOGIT} - too few events for a trustworthy logistic fit, reporting skipped"}

    x = sm.add_constant(labeled_frame[list(COMPONENT_NAMES)].astype(float))
    y = labeled_frame[target_col].astype(float)

    split_idx = int(n * train_frac)
    x_train, x_oos = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_oos = y.iloc[:split_idx], y.iloc[split_idx:]

    if y_train.nunique() < 2 or len(x_oos) == 0:
        return {"n": n, "sufficient_n": True, "note": "target has no variation in the train split, or the OOS split is empty - cannot fit/evaluate"}

    try:
        model = sm.Logit(y_train, x_train).fit(disp=0)
    except Exception as exc:  # perfect/quasi separation at small N is a real, expected failure mode here, not a bug
        return {"n": n, "sufficient_n": True, "note": f"logistic fit failed ({exc.__class__.__name__}) - likely perfect/quasi-separation at this N"}

    pred_prob = model.predict(x_oos)
    oos_accuracy = float(((pred_prob >= 0.5).astype(float) == y_oos).mean())

    return {
        "n": n, "sufficient_n": True, "n_train": len(x_train), "n_oos": len(x_oos),
        "coefficients": model.params.to_dict(), "p_values": model.pvalues.to_dict(),
        "pseudo_r2_mcfadden": float(model.prsquared), "oos_accuracy": oos_accuracy,
    }


def _fmt_pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


def _fmt_num(v, digits: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def render_predictive_value_report(
    single_feature_tables: dict[str, pd.DataFrame], combination_tables: dict[str, pd.DataFrame],
    reversal_vs_continuation: dict, logit_result: dict,
) -> str:
    lines = ["# Feature Predictive Value", "", "## Single-feature counterfactual: P(bad outcome | feature fired) vs P(... | not fired)", ""]
    for horizon_label, table in single_feature_tables.items():
        lines.append(f"### {horizon_label}")
        lines.append("")
        lines.append("| Feature | N fired | N not fired | P(outcome given fired) | P(outcome given not fired) |")
        lines.append("|---|---|---|---|---|")
        for _, row in table.iterrows():
            lines.append(
                f"| {row['feature']} | {row['n_with_condition']} | {row['n_without_condition']} "
                f"| {_fmt_pct(row['p_outcome_given_condition'])} | {_fmt_pct(row['p_outcome_given_not_condition'])} |"
            )
        lines.append("")
    lines.append("## Feature combinations (spec section 25)")
    lines.append("")
    for horizon_label, table in combination_tables.items():
        lines.append(f"### {horizon_label}")
        lines.append("")
        lines.append("| Combination | N fired | N not fired | P(outcome given fired) | P(outcome given not fired) |")
        lines.append("|---|---|---|---|---|")
        for _, row in table.iterrows():
            lines.append(
                f"| {row['combination']} | {row['n_with_condition']} | {row['n_without_condition']} "
                f"| {_fmt_pct(row['p_outcome_given_condition'])} | {_fmt_pct(row['p_outcome_given_not_condition'])} |"
            )
        lines.append("")
    lines.append("## Reversal vs continuation effect size (spec section 38)")
    lines.append("")
    lines.append(f"- {reversal_vs_continuation['label_a']} (n={reversal_vs_continuation['n_a']}) median: {_fmt_num(reversal_vs_continuation['median_a'])}")
    lines.append(f"- {reversal_vs_continuation['label_b']} (n={reversal_vs_continuation['n_b']}) median: {_fmt_num(reversal_vs_continuation['median_b'])}")
    lines.append(f"- Mann-Whitney p-value: {_fmt_num(reversal_vs_continuation['mann_whitney']['p_value'])}")
    lines.append(f"- Cliff's delta: {_fmt_num(reversal_vs_continuation['cliffs_delta'])}")
    lines.append("")
    lines.append("## Logistic regression confirmatory check (spec sections 39/40)")
    lines.append("")
    lines.append(
        "Hypothesis-generating only, NOT a production decision rule - single statsmodels.Logit fit on the "
        "10 boolean exhaustion components, strict temporal train/OOS split, no tree ensembles (too few events "
        "at this project's scale for that to be trustworthy - see project plan)."
    )
    lines.append("")
    if not logit_result.get("sufficient_n", False) or "coefficients" not in logit_result:
        lines.append(f"Not run / not reportable: {logit_result.get('note', 'insufficient data')}")
    else:
        lines.append(
            f"n={logit_result['n']} (train={logit_result['n_train']}, OOS={logit_result['n_oos']}), "
            f"McFadden pseudo-R2={logit_result['pseudo_r2_mcfadden']:.3f}, OOS accuracy={logit_result['oos_accuracy']:.1%}"
        )
        lines.append("")
        lines.append("| Feature | Coefficient | p-value |")
        lines.append("|---|---|---|")
        for name, coef in logit_result["coefficients"].items():
            lines.append(f"| {name} | {coef:.3f} | {logit_result['p_values'][name]:.3f} |")
    return "\n".join(lines)
