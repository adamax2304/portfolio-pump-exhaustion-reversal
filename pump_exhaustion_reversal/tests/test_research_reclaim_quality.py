from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import reclaim_quality as rq


def _synthetic_ohlcv(n: int = 40) -> pd.DataFrame:
    # flat run-up to a climax at idx 20 (high=150), then a reclaim attempt at idx 25.
    close = [100.0] * n
    high = [100.0] * n
    low = [100.0] * n
    open_ = [100.0] * n
    volume = [10.0] * n
    high[20], close[20], open_[20], volume[20] = 150.0, 148.0, 145.0, 500.0  # climax bar, high volume
    # reclaim bar: pokes above climax high with a long upper wick, then closes back down
    high[25], low[25], open_[25], close[25], volume[25] = 152.0, 144.0, 145.0, 146.0, 50.0
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _synthetic_feature_frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_interest": np.linspace(1000.0, 1000.0, n),
            "cvd_rolling": np.linspace(0.0, -50.0, n),
            "cvd_slope": np.full(n, -2.0),
        }
    )


def test_compute_reclaim_quality_features_basic_values():
    ohlcv = _synthetic_ohlcv()
    feature_frame = _synthetic_feature_frame()
    result = rq.compute_reclaim_quality_features(ohlcv, feature_frame, climax_idx_abs=20, reclaim_idx_abs=25)
    assert result["reclaim_depth_pct"] == pytest.approx((152.0 - 150.0) / 150.0)
    assert result["time_climax_to_reclaim_bars"] == 5
    assert result["volume_contraction_from_climax"] == pytest.approx(50.0 / 500.0)
    # upper wick = 152 - max(146,145) = 6; body = |146-145| = 1
    assert result["wick_to_body_ratio"] == pytest.approx(6.0)
    assert result["cvd_at_reclaim"] == pytest.approx(feature_frame["cvd_rolling"].iloc[25])
    assert result["reclaim_above_avwap"] in (True, False)


def test_compute_reclaim_quality_features_misordered_returns_empty():
    ohlcv = _synthetic_ohlcv()
    feature_frame = _synthetic_feature_frame()
    assert rq.compute_reclaim_quality_features(ohlcv, feature_frame, climax_idx_abs=25, reclaim_idx_abs=20) == {}


def test_compute_reclaim_quality_features_out_of_range_returns_empty():
    ohlcv = _synthetic_ohlcv()
    feature_frame = _synthetic_feature_frame()
    assert rq.compute_reclaim_quality_features(ohlcv, feature_frame, climax_idx_abs=20, reclaim_idx_abs=999) == {}


def test_compare_quality_by_outcome_shape():
    quality_df = pd.DataFrame(
        {
            "reclaim_depth_pct": [0.01, 0.02, 0.03, 0.04],
            "time_climax_to_reclaim_bars": [3, 4, 5, 6],
        }
    )
    outcomes = pd.Series(["profit", "adverse", "profit", "timeout"])
    result = rq.compare_quality_by_outcome(quality_df, outcomes)
    assert set(result["feature"]) == {"reclaim_depth_pct", "time_climax_to_reclaim_bars"}
    row = result[result["feature"] == "reclaim_depth_pct"].iloc[0]
    assert row["n_a"] == 2  # two "profit" rows
    assert row["n_b"] == 2


def test_compare_quality_by_outcome_empty():
    result = rq.compare_quality_by_outcome(pd.DataFrame(), pd.Series(dtype=object))
    assert result.empty


def test_univariate_logit_check_skips_below_min_n():
    quality_df = pd.DataFrame({"reclaim_depth_pct": [0.01, 0.02, 0.03]})
    outcomes = pd.Series(["profit", "adverse", "profit"])
    result = rq.univariate_logit_check(quality_df, outcomes)
    row = result[result["feature"] == "reclaim_depth_pct"].iloc[0]
    assert row["n"] == 3
    assert np.isnan(row["coef"])
    assert "insufficient" in row["note"]


def test_univariate_logit_check_fits_on_sufficient_separable_data():
    rng = np.random.default_rng(3)
    n = 60
    feature = rng.normal(0, 1, n)
    # feature strongly predicts outcome -> should fit cleanly, not perfect-separate
    prob = 1 / (1 + np.exp(-1.5 * feature))
    outcome = rng.uniform(size=n) < prob
    quality_df = pd.DataFrame({"reclaim_depth_pct": feature})
    outcomes = pd.Series(np.where(outcome, "profit", "adverse"))
    result = rq.univariate_logit_check(quality_df, outcomes)
    row = result[result["feature"] == "reclaim_depth_pct"].iloc[0]
    assert row["n"] == 60
    assert row["coef"] > 0  # feature positively associated with "profit" by construction
    assert not np.isnan(row["p_value"])
