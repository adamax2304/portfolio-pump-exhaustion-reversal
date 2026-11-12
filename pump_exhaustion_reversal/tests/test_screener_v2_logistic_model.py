from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import logistic_model


def _separable_df(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)  # should get a POSITIVE coefficient
    x2 = rng.normal(0, 1, n)  # should get a NEGATIVE coefficient
    z = 1.5 * x1 - 1.5 * x2
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(int)
    outcome = np.where(y == 1, "REVERSAL", "CONTINUATION")
    return pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "x1": x1, "x2": x2, "outcome": outcome})


def test_fit_l2_logistic_recovers_correct_coefficient_signs():
    df = _separable_df()
    model = logistic_model.fit_l2_logistic(df, ["x1", "x2"], [], "outcome", "REVERSAL", alpha=1.0)
    coefs = dict(zip(model.feature_names, model.coefficients))
    assert coefs["x1"] > 0
    assert coefs["x2"] < 0
    assert model.converged is True
    assert model.n_train_rows == len(df)


def test_fit_l2_logistic_higher_alpha_shrinks_coefficients():
    df = _separable_df()
    model_low = logistic_model.fit_l2_logistic(df, ["x1", "x2"], [], "outcome", "REVERSAL", alpha=0.01)
    model_high = logistic_model.fit_l2_logistic(df, ["x1", "x2"], [], "outcome", "REVERSAL", alpha=100.0)
    assert np.abs(model_high.coefficients).sum() < np.abs(model_low.coefficients).sum()


def test_fit_l2_logistic_event_weighting_prevents_one_huge_event_from_dominating():
    # one massive, mis-signed event (constant x=5, always CONTINUATION) vs
    # 100 small (1-row) events carrying the TRUE positive x->REVERSAL
    # relationship - event weighting (1/n_snapshots_in_event) must stop the
    # 1000:100 raw-row imbalance from flipping the learned sign.
    rng = np.random.default_rng(2)
    rows = [{"event_id": "big_event", "x": 5.0, "outcome": "CONTINUATION"} for _ in range(1000)]
    for i in range(100):
        x = rng.normal(0, 1)
        rows.append({"event_id": f"small_{i}", "x": x, "outcome": "REVERSAL" if x > 0 else "CONTINUATION"})
    df = pd.DataFrame(rows)
    model = logistic_model.fit_l2_logistic(df, ["x"], [], "outcome", "REVERSAL", alpha=0.1)
    coef = dict(zip(model.feature_names, model.coefficients))["x"]
    assert coef > 0


def test_predict_proba_separates_classes_on_training_data():
    df = _separable_df()
    model = logistic_model.fit_l2_logistic(df, ["x1", "x2"], [], "outcome", "REVERSAL", alpha=1.0)
    scores = logistic_model.predict_proba(model, df)
    mean_reversal = scores[df["outcome"] == "REVERSAL"].mean()
    mean_continuation = scores[df["outcome"] == "CONTINUATION"].mean()
    assert mean_reversal > mean_continuation


def test_fit_l2_logistic_categorical_feature_picks_up_association():
    n = 2000
    rng = np.random.default_rng(4)
    regime = rng.choice(["A", "B", "C"], size=n)
    p = np.where(regime == "A", 0.7, 0.3)
    y = (rng.uniform(size=n) < p).astype(int)
    outcome = np.where(y == 1, "REVERSAL", "CONTINUATION")
    df = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "regime": regime, "outcome": outcome})
    model = logistic_model.fit_l2_logistic(df, [], ["regime"], "outcome", "REVERSAL", alpha=0.1)
    scores = logistic_model.predict_proba(model, df)
    assert scores[df["regime"] == "A"].mean() > scores[df["regime"] != "A"].mean()


def test_predict_proba_returns_nan_for_missing_features_never_silently_imputed():
    n = 500
    rng = np.random.default_rng(5)
    x1 = rng.normal(0, 1, n)
    outcome = np.where(x1 > 0, "REVERSAL", "CONTINUATION")
    train_df = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "x1": x1, "outcome": outcome})
    model = logistic_model.fit_l2_logistic(train_df, ["x1"], [], "outcome", "REVERSAL", alpha=1.0)

    new_df = pd.DataFrame({"x1": [1.0, np.nan]})
    scores = logistic_model.predict_proba(model, new_df)
    assert pd.notna(scores.iloc[0])
    assert pd.isna(scores.iloc[1])


def test_coefficient_stability_report_flags_sign_flip():
    df1 = _separable_df(seed=10)
    df2 = _separable_df(seed=11)
    # a THIRD dataset where x1's relationship is deliberately FLIPPED
    rng = np.random.default_rng(12)
    x1 = rng.normal(0, 1, 2000)
    x2 = rng.normal(0, 1, 2000)
    z = -1.5 * x1 - 1.5 * x2  # x1 now NEGATIVELY associated
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=2000) < p).astype(int)
    df3 = pd.DataFrame({"event_id": [f"e{i}" for i in range(2000)], "x1": x1, "x2": x2, "outcome": np.where(y == 1, "REVERSAL", "CONTINUATION")})

    fold_models = {
        0: logistic_model.fit_l2_logistic(df1, ["x1", "x2"], [], "outcome", "REVERSAL", alpha=1.0),
        1: logistic_model.fit_l2_logistic(df2, ["x1", "x2"], [], "outcome", "REVERSAL", alpha=1.0),
        2: logistic_model.fit_l2_logistic(df3, ["x1", "x2"], [], "outcome", "REVERSAL", alpha=1.0),
    }
    result = logistic_model.coefficient_stability_report(fold_models)
    x1_row = result[result["feature"] == "x1"].iloc[0]
    x2_row = result[result["feature"] == "x2"].iloc[0]
    assert not x1_row["sign_consistent"]  # flips in fold 2
    assert x2_row["sign_consistent"]  # negative in all three folds
    assert x1_row["n_folds_present"] == 3


def test_predict_proba_reuses_frozen_standardization_not_refit_on_new_data():
    # a new dataframe with a WILDLY different mean/scale for x1 must NOT
    # shift the frozen fit-time standardization - predict_proba should keep
    # scoring relative to the ORIGINAL training distribution.
    n = 500
    rng = np.random.default_rng(6)
    x1 = rng.normal(0, 1, n)
    outcome = np.where(x1 > 0, "REVERSAL", "CONTINUATION")
    train_df = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "x1": x1, "outcome": outcome})
    model = logistic_model.fit_l2_logistic(train_df, ["x1"], [], "outcome", "REVERSAL", alpha=1.0)

    shifted_df = pd.DataFrame({"x1": [1000.0]})  # far outside the training range
    scores = logistic_model.predict_proba(model, shifted_df)
    assert scores.iloc[0] > 0.99  # correctly extrapolates as "very likely REVERSAL", not renormalized to ~0.5
