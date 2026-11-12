from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import validation


def test_event_weights_sum_to_one_per_event():
    df = pd.DataFrame({"event_id": ["e1", "e1", "e1", "e2", "e2"]})
    weights = validation.event_weights(df)
    totals = weights.groupby(df["event_id"]).sum()
    assert totals["e1"] == pytest.approx(1.0)
    assert totals["e2"] == pytest.approx(1.0)
    assert weights.iloc[0] == pytest.approx(1 / 3)
    assert weights.iloc[3] == pytest.approx(1 / 2)


def test_cluster_bootstrap_proportion_shape_and_ci_contains_statistic():
    rng = np.random.default_rng(1)
    clusters = [f"e{i}" for i in range(20)]
    rows = []
    for c in clusters:
        for _ in range(rng.integers(1, 5)):
            rows.append({"cluster": c, "value": float(rng.uniform() < 0.4)})
    df = pd.DataFrame(rows)
    result = validation.cluster_bootstrap_proportion(df["value"], df["cluster"], n_resamples=300)
    assert result["n_clusters"] == 20
    assert result["ci_low"] <= result["statistic"] <= result["ci_high"]


def test_cluster_bootstrap_proportion_too_few_clusters_returns_nan():
    df = pd.DataFrame({"cluster": ["a", "a", "b"], "value": [1.0, 0.0, 1.0]})
    result = validation.cluster_bootstrap_proportion(df["value"], df["cluster"])
    assert np.isnan(result["statistic"])


def test_compute_drawdown_from_high_pct_zero_at_the_high():
    df = pd.DataFrame({"current_price": [100.0, 95.0, 90.0], "event_high_so_far": [100.0, 100.0, 100.0]})
    result = validation.compute_drawdown_from_high_pct(df)
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.05)
    assert result.iloc[2] == pytest.approx(0.10)


def test_filter_pre_reversal_censored_keeps_only_snapshots_near_the_high():
    df = pd.DataFrame({"current_price": [100.0, 97.0, 90.0], "event_high_so_far": [100.0, 100.0, 100.0]})
    result = validation.filter_pre_reversal_censored(df, max_drawdown_from_high_pct=0.05)
    assert len(result) == 2  # 0% and 3% drawdown pass a 5% ceiling; 10% does not


def test_compute_price_position_in_range_pct_bounds():
    df = pd.DataFrame({"current_price": [100.0, 50.0, 0.0], "event_high_so_far": [100.0, 100.0, 100.0], "event_low_so_far": [0.0, 0.0, 0.0]})
    result = validation.compute_price_position_in_range_pct(df)
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(0.5)
    assert result.iloc[2] == pytest.approx(0.0)


def test_filter_high_state_cohort_keeps_only_snapshots_near_the_top_of_range():
    df = pd.DataFrame({"current_price": [100.0, 91.0, 50.0], "event_high_so_far": [100.0, 100.0, 100.0], "event_low_so_far": [0.0, 0.0, 0.0]})
    result = validation.filter_high_state_cohort(df, min_position_in_range_pct=0.9)
    assert len(result) == 2  # positions 1.0 and 0.91 pass a 0.9 floor; 0.5 does not


def test_split_chronological_by_event_no_overlap_and_ordered():
    events_df = pd.DataFrame({"event_id": [f"e{i}" for i in range(10)], "pump_detected_ts": list(range(10))})
    splits = validation.split_chronological_by_event(events_df, fractions=(0.6, 0.2, 0.2))
    assert splits["train"] | splits["validation"] | splits["test"] == set(events_df["event_id"])
    assert not (splits["train"] & splits["validation"])
    assert not (splits["validation"] & splits["test"])
    assert len(splits["train"]) == 6
    # train must contain the EARLIEST events (ts 0..5), test the LATEST (ts 8..9)
    assert splits["train"] == {f"e{i}" for i in range(6)}
    assert splits["test"] == {f"e{i}" for i in range(8, 10)}
