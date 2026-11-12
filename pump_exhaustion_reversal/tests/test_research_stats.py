from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import stats as research_stats


def test_clean_finite_drops_nan_and_inf():
    arr = research_stats.clean_finite([1.0, np.nan, 2.0, np.inf, -np.inf, 3.0])
    assert arr.tolist() == [1.0, 2.0, 3.0]


def test_bootstrap_ci_reasonable_on_known_distribution():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=10.0, scale=1.0, size=500)
    result = research_stats.bootstrap_ci(values, statistic=np.mean, n_resamples=1000, ci=0.90)
    assert result["n"] == 500
    assert result["ci_low"] < result["statistic"] < result["ci_high"]
    assert 9.5 < result["statistic"] < 10.5


def test_bootstrap_ci_too_few_points_returns_nan():
    result = research_stats.bootstrap_ci([1.0, 2.0], statistic=np.mean)
    assert result["n"] == 2
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])


def test_mann_whitney_detects_clear_separation():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [10.0, 11.0, 12.0, 13.0, 14.0]
    result = research_stats.mann_whitney(a, b)
    assert result["p_value"] < 0.05
    assert result["median_a"] == 3.0
    assert result["median_b"] == 12.0


def test_mann_whitney_too_few_points_returns_nan_pvalue():
    result = research_stats.mann_whitney([1.0, 2.0], [3.0, 4.0, 5.0, 6.0])
    assert np.isnan(result["p_value"])


def test_cliffs_delta_extremes():
    # every element of b exceeds every element of a -> delta == -1 (a is entirely "less than" b)
    assert research_stats.cliffs_delta([1, 2, 3], [10, 20, 30]) == pytest.approx(-1.0)
    assert research_stats.cliffs_delta([10, 20, 30], [1, 2, 3]) == pytest.approx(1.0)


def test_cliffs_delta_identical_groups_is_zero():
    assert research_stats.cliffs_delta([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)


def test_conditional_probability_basic():
    condition = pd.Series([True, True, False, False, True])
    outcome = pd.Series([True, True, False, True, False])
    result = research_stats.conditional_probability(condition, outcome)
    assert result["n_with_condition"] == 3
    assert result["n_without_condition"] == 2
    assert result["p_outcome_given_condition"] == pytest.approx(2 / 3)
    assert result["p_outcome_given_not_condition"] == pytest.approx(0.5)


def test_conditional_probability_drops_nan_rows_from_both_arms():
    condition = pd.Series([True, np.nan, False, True])
    outcome = pd.Series([True, True, False, np.nan])
    result = research_stats.conditional_probability(condition, outcome)
    # rows 1 (condition NaN) and 3 (outcome NaN) both excluded - only rows 0,2 remain
    assert result["n_with_condition"] == 1
    assert result["n_without_condition"] == 1


def test_effect_size_report_bundles_everything():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    report = research_stats.effect_size_report(a, b, label_a="reversal", label_b="continuation")
    assert report["label_a"] == "reversal"
    assert report["n_a"] == 6
    assert report["cliffs_delta"] < 0  # a is generally less than b
    assert "bootstrap_a" in report and "mann_whitney" in report
