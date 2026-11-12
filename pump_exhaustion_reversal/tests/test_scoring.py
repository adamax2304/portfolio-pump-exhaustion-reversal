from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.config import ExhaustionScoreWeights, PumpScoreWeights
from pump_exhaustion_reversal.strategy import scoring


def test_scaled_component_clips_to_0_100():
    assert scoring.scaled_component(-5, 6.0) == 0.0
    assert scoring.scaled_component(0, 6.0) == 0.0
    assert scoring.scaled_component(3, 6.0) == pytest.approx(50.0)
    assert scoring.scaled_component(60, 6.0) == 100.0  # way beyond divisor -> clipped, not >100


def test_scaled_abs_component_uses_absolute_value():
    assert scoring.scaled_abs_component(-4, 5.0) == pytest.approx(80.0)
    assert scoring.scaled_abs_component(4, 5.0) == pytest.approx(80.0)


def test_boolean_component_scalar_and_series():
    assert scoring.boolean_component(True) == 100.0
    assert scoring.boolean_component(False) == 0.0
    assert np.isnan(scoring.boolean_component(None))

    s = pd.Series([True, False, None])
    out = scoring.boolean_component(s)
    assert out.tolist()[:2] == [100.0, 0.0]
    assert np.isnan(out.iloc[2])


def test_liquidity_quality_score_scaling_and_degenerate_case():
    assert scoring.liquidity_quality_score(20_000_000, preferred_min=50_000_000, min_floor=0) == pytest.approx(40.0)
    assert scoring.liquidity_quality_score(100_000_000, preferred_min=50_000_000, min_floor=0) == 100.0
    assert np.isnan(scoring.liquidity_quality_score(10, preferred_min=10, min_floor=10))


def test_weighted_score_matches_manual_average_when_all_present():
    components = {"a": 100.0, "b": 0.0, "c": 50.0}
    weights = {"a": 50.0, "b": 30.0, "c": 20.0}
    result = scoring.weighted_score(components, weights)
    expected = (100.0 * 50 + 0.0 * 30 + 50.0 * 20) / 100
    assert result["score"] == pytest.approx(expected)
    assert result["missing_components"] == []
    assert result["available_weight_fraction"] == pytest.approx(1.0)


def test_weighted_score_renormalizes_over_available_components():
    # "b" missing -> score computed only over a+c, renormalized to sum 100
    components = {"a": 100.0, "b": np.nan, "c": 50.0}
    weights = {"a": 50.0, "b": 30.0, "c": 20.0}
    result = scoring.weighted_score(components, weights)
    expected = (100.0 * 50 + 50.0 * 20) / (50 + 20)
    assert result["score"] == pytest.approx(expected)
    assert result["missing_components"] == ["b"]
    assert result["available_weight_fraction"] == pytest.approx(0.70)


def test_weighted_score_all_missing_is_zero_not_error():
    components = {"a": np.nan, "b": np.nan}
    weights = {"a": 50.0, "b": 50.0}
    result = scoring.weighted_score(components, weights)
    assert result["score"] == 0.0
    assert result["available_weight_fraction"] == 0.0
    assert set(result["missing_components"]) == {"a", "b"}


def test_weighted_score_vectorized_matches_scalar_row_by_row():
    weights = {"a": 50.0, "b": 30.0, "c": 20.0}
    a = pd.Series([100.0, np.nan, 20.0])
    b = pd.Series([0.0, 40.0, np.nan])
    c = pd.Series([50.0, 60.0, 80.0])
    vec = scoring.weighted_score_vectorized({"a": a, "b": b, "c": c}, weights)

    for i in range(3):
        scalar = scoring.weighted_score({"a": a.iloc[i], "b": b.iloc[i], "c": c.iloc[i]}, weights)
        assert vec["score"].iloc[i] == pytest.approx(scalar["score"])
        assert vec["available_weight_fraction"].iloc[i] == pytest.approx(scalar["available_weight_fraction"])


def test_compute_pump_score_missing_liquidation_flagged_not_zeroed():
    weights = PumpScoreWeights()
    with_liq = scoring.compute_pump_score(
        max_return_zscore=6.0, volume_zscore=6.0, realized_vol_ratio=5.0,
        short_liq_zscore=8.0, oi_zscore=5.0, perp_spot_imbalance_score=np.nan,
        liquidity_score=100.0, weights=weights,
    )
    without_liq = scoring.compute_pump_score(
        max_return_zscore=6.0, volume_zscore=6.0, realized_vol_ratio=5.0,
        short_liq_zscore=np.nan, oi_zscore=5.0, perp_spot_imbalance_score=np.nan,
        liquidity_score=100.0, weights=weights,
    )
    assert without_liq["missing_liquidation_data"] is True
    assert with_liq["missing_liquidation_data"] is False
    # both should still be high scores (near 100) - missing data must not tank the score to 0
    assert without_liq["score"] > 90
    assert with_liq["score"] > 90


def test_compute_exhaustion_score_all_true_is_100():
    weights = ExhaustionScoreWeights()
    flags = {name: True for name in scoring.EXHAUSTION_SCORE_COMPONENT_NAMES}
    result = scoring.compute_exhaustion_score(flags, weights)
    assert result["score"] == pytest.approx(100.0)


def test_compute_exhaustion_score_all_false_is_0():
    weights = ExhaustionScoreWeights()
    flags = {name: False for name in scoring.EXHAUSTION_SCORE_COMPONENT_NAMES}
    result = scoring.compute_exhaustion_score(flags, weights)
    assert result["score"] == pytest.approx(0.0)
