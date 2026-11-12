"""Shared statistical helpers for the Stage 1.5 research/diagnostic phase
(spec sections 25/38). No trading logic reads this module - it exists purely
to compare two cohorts of already-computed event/trade values (e.g.
"reversal" vs "continuation") and report effect sizes with honest small-N
caveats, not to feed any score/threshold. First real users of scipy in this
repo (previously an unused declared dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

MIN_N_FOR_BOOTSTRAP = 5
MIN_N_FOR_MANN_WHITNEY = 3


def clean_finite(values) -> np.ndarray:
    """Drop NaN/inf - the shared first step for every function below."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values, statistic=np.median, n_resamples: int = 2000, ci: float = 0.90, seed: int = 7) -> dict:
    """Percentile bootstrap CI for `statistic` over `values`. n < 5 finite
    values is too few to bootstrap meaningfully - returns NaN bounds with the
    real n so callers can render "insufficient data" rather than a fake CI."""
    arr = clean_finite(values)
    if len(arr) < MIN_N_FOR_BOOTSTRAP:
        return {"n": len(arr), "statistic": np.nan, "ci_low": np.nan, "ci_high": np.nan, "ci_level": ci}
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    boot_stats = np.array([statistic(arr[idx]) for idx in resample_idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_stats, [alpha, 1 - alpha])
    return {"n": len(arr), "statistic": float(statistic(arr)), "ci_low": float(lo), "ci_high": float(hi), "ci_level": ci}


def mann_whitney(group_a, group_b) -> dict:
    """Two-sided Mann-Whitney U test between two independent samples. Fewer
    than 3 finite values in either group is too few for the test to be
    meaningful - returns NaN p_value/u_statistic rather than a misleading
    number from a degenerate sample."""
    a, b = clean_finite(group_a), clean_finite(group_b)
    result = {
        "n_a": len(a), "n_b": len(b), "u_statistic": np.nan, "p_value": np.nan,
        "median_a": float(np.median(a)) if len(a) else np.nan,
        "median_b": float(np.median(b)) if len(b) else np.nan,
    }
    if len(a) < MIN_N_FOR_MANN_WHITNEY or len(b) < MIN_N_FOR_MANN_WHITNEY:
        return result
    u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    result["u_statistic"] = float(u)
    result["p_value"] = float(p)
    return result


def cliffs_delta(group_a, group_b) -> float:
    """Rank-based, non-parametric effect size in [-1, 1]: fraction of pairs
    (a_i, b_j) where a_i > b_j minus the fraction where a_i < b_j. Preferred
    over Cohen's d here since none of the R-multiple/return distributions in
    this project are assumed normal. O(n*m), fine at this project's
    event-count scale (low hundreds); NaN if either group is empty."""
    a, b = clean_finite(group_a), clean_finite(group_b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = sum(int((ai > b).sum()) for ai in a)
    less = sum(int((ai < b).sum()) for ai in a)
    return float((greater - less) / (len(a) * len(b)))


def conditional_probability(condition: pd.Series, outcome: pd.Series) -> dict:
    """P(outcome | condition) vs P(outcome | not condition) - the base
    building block for the counterfactual feature study (spec section 25).
    Rows where either side is NaN are dropped from both arms (a feature that
    wasn't computable for an event says nothing about that event's outcome
    either way - excluded, not treated as "condition absent")."""
    mask = condition.notna() & outcome.notna()
    cond, out = condition[mask].astype(bool), outcome[mask].astype(bool)
    with_cond, without_cond = out[cond], out[~cond]
    return {
        "n_with_condition": int(cond.sum()), "n_without_condition": int((~cond).sum()),
        "p_outcome_given_condition": float(with_cond.mean()) if len(with_cond) else np.nan,
        "p_outcome_given_not_condition": float(without_cond.mean()) if len(without_cond) else np.nan,
    }


def effect_size_report(group_a, group_b, label_a: str = "A", label_b: str = "B") -> dict:
    """One bundled two-group comparison: medians, bootstrap CIs, Mann-Whitney
    p-value, Cliff's delta - everything predictive_value.py needs for one
    report row, computed once instead of re-deriving the cleaned arrays
    per-statistic."""
    a, b = clean_finite(group_a), clean_finite(group_b)
    return {
        "label_a": label_a, "label_b": label_b, "n_a": len(a), "n_b": len(b),
        "median_a": float(np.median(a)) if len(a) else np.nan,
        "median_b": float(np.median(b)) if len(b) else np.nan,
        "bootstrap_a": bootstrap_ci(a), "bootstrap_b": bootstrap_ci(b),
        "mann_whitney": mann_whitney(a, b), "cliffs_delta": cliffs_delta(a, b),
    }
