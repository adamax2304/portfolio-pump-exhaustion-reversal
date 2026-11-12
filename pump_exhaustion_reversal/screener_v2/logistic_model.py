"""Phase 3B point 6: the baseline model, and ONLY the baseline model for
this phase - a regularized (L2) logistic regression, event-weighted,
standardized continuous features, one-hot categoricals. Deliberately a small
custom fit (scipy L-BFGS-B on the weighted-penalized log-likelihood) rather
than adding scikit-learn as a dependency: Phase 1.5 already made this same
call for the same reason (statsmodels.Logit only, no sklearn, to keep the
temptation toward tree ensembles on a still-modest N out of reach) - this
keeps that boundary intact while still delivering exactly the L2-logistic
behavior point 6 asks for (standardization, L2, coefficients, no automatic
feature selection). Point 24's gradient-boosting COMPARISON model, gated
behind this baseline actually showing OOS ranking lift, is out of scope for
this module and this run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from . import validation

_EPS = 1e-12


@dataclass
class FittedLogisticModel:
    feature_names: list[str]  # standardized-continuous + one-hot columns, in coefficient order
    coefficients: np.ndarray
    intercept: float
    continuous_features: list[str]
    categorical_features: list[str]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    categorical_reference_levels: dict[str, object]
    categorical_all_levels: dict[str, list]
    alpha: float
    n_train_rows: int
    n_train_events: int
    converged: bool

    def coefficient_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names, "coefficient": self.coefficients})


def _standardize_fit(df: pd.DataFrame, continuous_features: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    means, stds, cols = {}, {}, {}
    for f in continuous_features:
        m = float(df[f].mean())
        s = float(df[f].std())
        if not s or pd.isna(s):
            s = 1.0
        means[f], stds[f] = m, s
        cols[f] = (df[f] - m) / s
    return pd.DataFrame(cols, index=df.index), means, stds


def _standardize_apply(df: pd.DataFrame, continuous_features: list[str], means: dict, stds: dict) -> pd.DataFrame:
    return pd.DataFrame({f: (df[f] - means[f]) / stds[f] for f in continuous_features}, index=df.index)


def _one_hot_fit(df: pd.DataFrame, categorical_features: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    """Reference level (dropped, absorbed into the intercept) = the most
    frequent category IN THE FITTING DATA - decided once at fit time,
    reused unchanged at predict time (never re-derived from validation/test
    rows)."""
    reference_levels, all_levels, cols = {}, {}, {}
    for f in categorical_features:
        counts = df[f].dropna().value_counts()
        levels = sorted(counts.index.tolist(), key=str)
        reference = counts.idxmax() if len(counts) else None
        all_levels[f] = levels
        reference_levels[f] = reference
        for level in levels:
            if level == reference:
                continue
            cols[f"{f}={level}"] = (df[f] == level).astype(float)
    return pd.DataFrame(cols, index=df.index), reference_levels, all_levels


def _one_hot_apply(df: pd.DataFrame, categorical_features: list[str], reference_levels: dict, all_levels: dict) -> pd.DataFrame:
    cols = {}
    for f in categorical_features:
        for level in all_levels.get(f, []):
            if level == reference_levels.get(f):
                continue
            cols[f"{f}={level}"] = (df[f] == level).astype(float)
    return pd.DataFrame(cols, index=df.index)


def fit_l2_logistic(
    df: pd.DataFrame, continuous_features: list[str], categorical_features: list[str],
    outcome_col: str, positive_label: str, event_col: str = "event_id", alpha: float = 1.0,
) -> FittedLogisticModel:
    """Fits on ONLY the rows already present in `df` (the caller is
    responsible for restricting to resolved REVERSAL/CONTINUATION snapshots
    and to the desired training fold - this function does no outcome or
    fold filtering of its own, so it stays a pure fitting primitive).

    Sample weights = validation.event_weights (point 9: sum-per-event ~ 1),
    RENORMALIZED to sum to N (average weight 1) so alpha's scale means the
    same thing across folds of different sizes while preserving the
    "no single event dominates" property. Intercept is never penalized -
    the standard L2-logistic convention."""
    required = list(continuous_features) + list(categorical_features) + [outcome_col, event_col]
    resolved = df.dropna(subset=required).copy()
    y = (resolved[outcome_col] == positive_label).astype(float).to_numpy()

    standardized, means, stds = _standardize_fit(resolved, list(continuous_features))
    one_hot, reference_levels, all_levels = _one_hot_fit(resolved, list(categorical_features))
    X_features = pd.concat([standardized, one_hot], axis=1)
    feature_names = list(X_features.columns)
    X = np.column_stack([np.ones(len(resolved)), X_features.to_numpy(dtype=float)])

    raw_weights = validation.event_weights(resolved, event_col).to_numpy()
    total = raw_weights.sum()
    weights = raw_weights / total * len(raw_weights) if total else np.ones(len(raw_weights))

    def _objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ beta
        p = expit(z)
        ll = np.sum(weights * (y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS)))
        penalty = 0.5 * alpha * np.sum(beta[1:] ** 2)
        neg_obj = -ll + penalty
        grad_ll = X.T @ (weights * (p - y))
        grad_penalty = np.concatenate([[0.0], alpha * beta[1:]])
        return float(neg_obj), grad_ll + grad_penalty

    x0 = np.zeros(X.shape[1])
    result = minimize(_objective, x0, jac=True, method="L-BFGS-B")

    return FittedLogisticModel(
        feature_names=feature_names, coefficients=result.x[1:], intercept=float(result.x[0]),
        continuous_features=list(continuous_features), categorical_features=list(categorical_features),
        feature_means=means, feature_stds=stds, categorical_reference_levels=reference_levels,
        categorical_all_levels=all_levels, alpha=alpha, n_train_rows=len(resolved),
        n_train_events=int(resolved[event_col].nunique()), converged=bool(result.success),
    )


def coefficient_stability_report(fold_models: dict) -> pd.DataFrame:
    """Point 15: one row per feature (standardized-continuous or one-hot
    column, unioned across every fold's own feature_names - a feature
    absent from a given fold, e.g. an oi_regime level never observed in
    that fold's training data, is simply missing for that fold's column
    rather than treated as zero), with each fold's own coefficient plus a
    sign-consistency summary. A feature whose sign flips between folds is a
    bad feature even if the AGGREGATE fit looks clean."""
    all_features: set[str] = set()
    for model in fold_models.values():
        all_features.update(model.feature_names)

    rows = []
    for feature in sorted(all_features):
        per_fold = {}
        for fold_key, model in fold_models.items():
            if feature in model.feature_names:
                idx = model.feature_names.index(feature)
                per_fold[fold_key] = float(model.coefficients[idx])
        values = list(per_fold.values())
        row = {"feature": feature, **{f"fold_{k}": v for k, v in per_fold.items()}}
        if values:
            median_sign = np.sign(np.median(values))
            same_sign = (np.sign(values) == median_sign).mean() if median_sign != 0 else np.nan
            row.update(
                {
                    "mean_coefficient": float(np.mean(values)), "std_coefficient": float(np.std(values)),
                    "min_coefficient": float(np.min(values)), "max_coefficient": float(np.max(values)),
                    "n_folds_present": len(values), "fraction_same_sign": float(same_sign) if pd.notna(same_sign) else np.nan,
                    "sign_consistent": bool(same_sign == 1.0) if pd.notna(same_sign) else False,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def predict_proba(model: FittedLogisticModel, df: pd.DataFrame) -> pd.Series:
    """Applies the FROZEN standardization/one-hot mapping from fit time -
    never refits on df, so this is safe on validation/test/forward data
    without leakage. Rows missing a required feature score to NaN (never
    silently imputed) - the caller drops NaN scores before ranking."""
    standardized = _standardize_apply(df, model.continuous_features, model.feature_means, model.feature_stds)
    one_hot = _one_hot_apply(df, model.categorical_features, model.categorical_reference_levels, model.categorical_all_levels)
    X_features = pd.concat([standardized, one_hot], axis=1).reindex(columns=model.feature_names, fill_value=0.0)
    valid = X_features[model.continuous_features].notna().all(axis=1) if model.continuous_features else pd.Series(True, index=df.index)
    z = model.intercept + X_features.to_numpy(dtype=float) @ model.coefficients
    scores = pd.Series(expit(z), index=df.index)
    return scores.where(valid)
