"""Phase 3B points 1-3: the pre-registered, deliberately small feature set
for the logistic baseline - fixed BEFORE fitting anything, never expanded
after seeing a fit metric improve. Point 1's correlation-based dedup is a
DIAGNOSTIC over the full Phase 3A.1 KEEP/WEAK candidate list (used to decide
FEATURE_SET_V1 below, and to document why the three bars_since_last_new_high/
time_since_last_new_high_minutes/pullback_duration_bars aliases collapse to
one representative) - it is not itself part of the fitting pipeline.

Point 3: no old hand-made composite scores (pump_score, exhaustion_score,
failed_reclaim score, ...) anywhere in this module or in the baseline model -
the model must learn from the same raw causal features Phase 3A.1 audited,
not from our own prior assumptions about what matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Point 2: the canonical continuous feature set - Phase 3A.1's KEEP tier,
# already deduplicated by hand (bars_since_last_new_high stands in for its
# two aliases; see feature_correlation_and_dedup.md for the measured
# correlation that justifies this).
FEATURE_SET_V1_CONTINUOUS: tuple[str, ...] = (
    "bars_since_last_new_high",
    "pullback_depth_atr",
    "rejection_atr",
    "lower_high_strength",
    "realized_vol_15m",
    "current_return_from_start",
    "upper_wick_pct",
)
FEATURE_SET_V1_CATEGORICAL: tuple[str, ...] = ("oi_regime",)
FEATURE_SET_V1_OPTIONAL_WEAK: tuple[str, ...] = ("volume_per_return",)

# Point 1: the aliasing families already known from Phase 3A.1 (identical or
# near-identical Cliff's delta across the whole summary table is the
# giveaway) - listed here so the correlation matrix's dedup output can be
# checked against what was already suspected, not just accepted blindly.
KNOWN_ALIAS_FAMILIES: dict[str, tuple[str, ...]] = {
    "bars_since_last_new_high": ("time_since_last_new_high_minutes", "pullback_duration_bars"),
}

# Point 16: the four pre-specified ablation variants - marginal lift only,
# never accepted just because a train-set metric improved.
ABLATION_VARIANTS: dict[str, dict] = {
    "BASE": {"continuous": list(FEATURE_SET_V1_CONTINUOUS), "categorical": []},
    "BASE_PLUS_OI": {"continuous": list(FEATURE_SET_V1_CONTINUOUS), "categorical": list(FEATURE_SET_V1_CATEGORICAL)},
    "BASE_PLUS_VOLUME_EFFICIENCY": {"continuous": list(FEATURE_SET_V1_CONTINUOUS) + list(FEATURE_SET_V1_OPTIONAL_WEAK), "categorical": []},
    "BASE_PLUS_OI_PLUS_VOLUME_EFFICIENCY": {"continuous": list(FEATURE_SET_V1_CONTINUOUS) + list(FEATURE_SET_V1_OPTIONAL_WEAK), "categorical": list(FEATURE_SET_V1_CATEGORICAL)},
}


def compute_feature_correlation_matrix(snapshots_df: pd.DataFrame, features: list[str], method: str = "spearman") -> pd.DataFrame:
    """Point 1: pairwise correlation among candidate features. Spearman (not
    Pearson) by default - several Phase 3A.1 features are extremely
    heavy-tailed (e.g. volume_per_return spans ~7 orders of magnitude in its
    own decile table), where Pearson correlation is dominated by a handful
    of extreme rows and Spearman's rank-based measure is the more honest
    read of "do these two features order snapshots the same way"."""
    available = [f for f in features if f in snapshots_df.columns and pd.api.types.is_numeric_dtype(snapshots_df[f])]
    return snapshots_df[available].corr(method=method)


def deduplicate_correlated_features(corr_matrix: pd.DataFrame, ordered_features: list[str], threshold: float = 0.85) -> dict:
    """Point 1: greedy dedup in the CALLER-supplied priority order (e.g.
    Phase 3A.1's own top_candidate/summary ranking) - the first feature in
    a correlated cluster is kept, every later one whose |corr| with an
    ALREADY-KEPT feature exceeds `threshold` is dropped and recorded against
    the specific kept feature it duplicates. A feature is never dropped for
    correlating with something that was ITSELF already dropped - only
    against the surviving representative, so a family collapses to exactly
    one member regardless of chain length."""
    kept: list[str] = []
    dropped: dict[str, dict] = {}
    for f in ordered_features:
        if f not in corr_matrix.columns:
            continue
        redundant_with = None
        best_abs_corr = 0.0
        for k in kept:
            c = corr_matrix.loc[f, k]
            if pd.notna(c) and abs(c) > threshold and abs(c) > best_abs_corr:
                redundant_with, best_abs_corr = k, abs(c)
        if redundant_with is not None:
            dropped[f] = {"redundant_with": redundant_with, "abs_correlation": float(best_abs_corr)}
        else:
            kept.append(f)
    return {"kept": kept, "dropped": dropped, "threshold": threshold}
