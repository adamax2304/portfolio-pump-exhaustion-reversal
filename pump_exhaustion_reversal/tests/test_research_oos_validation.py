from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import oos_validation as oos


def test_oi_quality_tier_high_medium_low_missing():
    high_frame = pd.DataFrame({"open_interest": [1.0] * 20})
    assert oos.oi_quality_tier(high_frame, 0, 19) == "HIGH"

    medium_frame = pd.DataFrame({"open_interest": [1.0] * 17 + [np.nan] * 3})
    assert oos.oi_quality_tier(medium_frame, 0, 19) == "MEDIUM"

    low_frame = pd.DataFrame({"open_interest": [1.0] * 5 + [np.nan] * 15})
    assert oos.oi_quality_tier(low_frame, 0, 19) == "LOW"

    all_nan_frame = pd.DataFrame({"open_interest": [np.nan] * 20})
    assert oos.oi_quality_tier(all_nan_frame, 0, 19) == "MISSING"

    no_column_frame = pd.DataFrame({"close": [1.0] * 20})
    assert oos.oi_quality_tier(no_column_frame, 0, 19) == "MISSING"


def test_symbol_clustering_summary_basic():
    candidates_df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "net_r": [1.0, 1.0, 1.0, -0.5, 0.2, 0.1, 0.1, 0.1],
        }
    )
    summary = oos.symbol_clustering_summary(candidates_df)
    assert summary["n_events"] == 8
    assert summary["n_unique_symbols"] == 6
    assert summary["max_events_per_symbol"] == 3
    # AAA (3.0) dominates total pnl (3.0-0.5+0.2+0.1+0.1+0.1=2.9); with only 6
    # symbols total, "top 5" excludes exactly one (the smallest single-event
    # contributor) - share should still be a large majority of total pnl.
    assert summary["top5_symbol_pnl_share"] > 0.9


def test_symbol_clustering_summary_empty():
    summary = oos.symbol_clustering_summary(pd.DataFrame())
    assert summary["n_events"] == 0


def test_cluster_bootstrap_ci_shape():
    rng = np.random.default_rng(3)
    symbols = [f"SYM{i}" for i in range(10)]
    rows = []
    for s in symbols:
        for _ in range(rng.integers(1, 4)):
            rows.append({"symbol": s, "net_r": float(rng.normal(0.1, 0.5))})
    candidates_df = pd.DataFrame(rows)
    result = oos.cluster_bootstrap_ci(candidates_df, n_resamples=200)
    assert result["n_symbols"] == 10
    assert result["ci_low"] <= result["median_expectancy"] <= result["ci_high"]


def test_cluster_bootstrap_ci_too_few_symbols():
    candidates_df = pd.DataFrame({"symbol": ["A", "A", "B"], "net_r": [1.0, 1.0, -1.0]})
    result = oos.cluster_bootstrap_ci(candidates_df)
    assert np.isnan(result["median_expectancy"])


def test_regime_breakdown_splits_by_dimension():
    candidates_df = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "net_r": [1.0, -1.0, 0.5]})
    events_with_regime = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"], "regime_trend": ["bull", "bull", "bear"],
            "regime_weekday": ["weekday", "weekday", "weekend"], "regime_session": ["asia", "eu", "us"],
        }
    )
    result = oos.regime_breakdown(candidates_df, events_with_regime)
    bull_row = result[(result["dimension"] == "regime_trend") & (result["value"] == "bull")].iloc[0]
    assert bull_row["n"] == 2
    assert bull_row["net_expectancy_r"] == pytest.approx(0.0)  # mean(1.0, -1.0)


def test_pump_magnitude_buckets_terciles():
    candidates_df = pd.DataFrame({"event_id": [f"e{i}" for i in range(9)], "net_r": [0.1] * 9})
    magnitude_by_event = {f"e{i}": float(i) for i in range(9)}  # 0..8, evenly spread -> clean terciles
    result = oos.pump_magnitude_buckets(candidates_df, magnitude_by_event)
    assert set(result["bucket"]) == {"small", "medium", "extreme"}
    assert result["n"].sum() == 9


def test_pump_magnitude_buckets_empty_magnitude_map():
    candidates_df = pd.DataFrame({"event_id": ["e1"], "net_r": [0.1]})
    result = oos.pump_magnitude_buckets(candidates_df, {})
    assert result.empty


def test_match_controls_finds_k_nearest():
    pool = pd.DataFrame(
        {
            "event_id": ["sig1", "c1", "c2", "c3", "c4"],
            "pump_score": [80.0, 81.0, 50.0, 79.0, 20.0],
            "volume_zscore": [5.0, 5.1, 2.0, 4.9, 1.0],
            "btc_return_concurrent": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = oos.match_controls({"sig1"}, pool, k=2)
    assert len(result) == 2
    assert set(result["control_event_id"]) == {"c1", "c3"}  # closest to sig1's (80, 5.0) values


def test_match_controls_empty_when_no_pool():
    pool = pd.DataFrame({"event_id": ["sig1"], "pump_score": [80.0], "volume_zscore": [5.0], "btc_return_concurrent": [0.0]})
    result = oos.match_controls({"sig1"}, pool, k=2)
    assert result.empty


def test_check_acceptance_criteria_insufficient_sample():
    criteria = {"n_signals_preferred_minimum": 50}
    result = oos.check_acceptance_criteria({"n": 10}, {}, {}, criteria)
    assert result["outcome"] == "INSUFFICIENT_SAMPLE"


def test_check_acceptance_criteria_confirmed():
    criteria = {
        "n_signals_preferred_minimum": 10, "net_expectancy_r_must_exceed": 0.0, "profit_factor_must_exceed": 1.15,
        "cluster_bootstrap_median_expectancy_must_exceed": 0.0, "max_single_symbol_pnl_share_pct": 25.0,
    }
    summary = {"n": 50, "net_expectancy_r": 0.2, "profit_factor": 1.5}
    cluster_bootstrap = {"median_expectancy": 0.15}
    symbol_summary = {"top5_symbol_pnl_share": 0.2}  # under the 25% max_single_symbol_pnl_share_pct threshold
    result = oos.check_acceptance_criteria(summary, cluster_bootstrap, symbol_summary, criteria, cost_stress_positive=True, regime_concentration_ok=True)
    assert result["outcome"] == "A_CONFIRMED_ENOUGH_FOR_PAPER_TRADING"


def test_check_acceptance_criteria_reject_when_net_negative():
    criteria = {
        "n_signals_preferred_minimum": 10, "net_expectancy_r_must_exceed": 0.0, "profit_factor_must_exceed": 1.15,
        "cluster_bootstrap_median_expectancy_must_exceed": 0.0, "max_single_symbol_pnl_share_pct": 25.0,
    }
    summary = {"n": 50, "net_expectancy_r": -0.1, "profit_factor": 0.8}
    result = oos.check_acceptance_criteria(summary, {"median_expectancy": -0.05}, {"top5_symbol_pnl_share": 0.4}, criteria)
    assert result["outcome"] == "C_REJECT_CLOSE_BRANCH"


def test_check_acceptance_criteria_weak_uncertain():
    # net expectancy positive but profit factor and cluster bootstrap fail -> weak/uncertain, not an outright reject
    criteria = {
        "n_signals_preferred_minimum": 10, "net_expectancy_r_must_exceed": 0.0, "profit_factor_must_exceed": 1.15,
        "cluster_bootstrap_median_expectancy_must_exceed": 0.0, "max_single_symbol_pnl_share_pct": 25.0,
    }
    summary = {"n": 50, "net_expectancy_r": 0.05, "profit_factor": 1.05}
    result = oos.check_acceptance_criteria(summary, {"median_expectancy": -0.01}, {"top5_symbol_pnl_share": 0.4}, criteria)
    assert result["outcome"] == "B_WEAK_UNCERTAIN_WAIT_FOR_MORE_DATA"
