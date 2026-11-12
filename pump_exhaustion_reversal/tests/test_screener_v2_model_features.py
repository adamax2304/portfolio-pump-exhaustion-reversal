from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import model_features as mf


def test_compute_feature_correlation_matrix_detects_near_duplicate():
    n = 200
    rng = np.random.default_rng(3)
    base = rng.uniform(0, 100, n)
    df = pd.DataFrame(
        {
            "a": base,
            "b_alias_of_a": base + rng.normal(0, 0.5, n),  # near-identical to a
            "c_independent": rng.uniform(0, 100, n),
        }
    )
    corr = mf.compute_feature_correlation_matrix(df, ["a", "b_alias_of_a", "c_independent"])
    assert corr.loc["a", "b_alias_of_a"] > 0.9
    assert abs(corr.loc["a", "c_independent"]) < 0.3


def test_deduplicate_correlated_features_keeps_first_of_a_cluster():
    corr = pd.DataFrame(
        {
            "a": [1.0, 0.95, 0.93, 0.1],
            "b": [0.95, 1.0, 0.90, 0.05],
            "c": [0.93, 0.90, 1.0, 0.08],
            "d": [0.1, 0.05, 0.08, 1.0],
        },
        index=["a", "b", "c", "d"],
    )
    result = mf.deduplicate_correlated_features(corr, ["a", "b", "c", "d"], threshold=0.85)
    assert result["kept"] == ["a", "d"]
    assert set(result["dropped"]) == {"b", "c"}
    assert result["dropped"]["b"]["redundant_with"] == "a"
    assert result["dropped"]["c"]["redundant_with"] == "a"


def test_deduplicate_correlated_features_respects_priority_order():
    # same cluster, but here "b" is listed FIRST -> b survives, a gets dropped against it
    corr = pd.DataFrame({"a": [1.0, 0.95], "b": [0.95, 1.0]}, index=["a", "b"])
    result = mf.deduplicate_correlated_features(corr, ["b", "a"], threshold=0.85)
    assert result["kept"] == ["b"]
    assert result["dropped"]["a"]["redundant_with"] == "b"


def test_deduplicate_correlated_features_no_correlation_keeps_all():
    corr = pd.DataFrame({"a": [1.0, 0.1], "b": [0.1, 1.0]}, index=["a", "b"])
    result = mf.deduplicate_correlated_features(corr, ["a", "b"], threshold=0.85)
    assert result["kept"] == ["a", "b"]
    assert result["dropped"] == {}
