from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import diagnostics


def _snapshots_df() -> pd.DataFrame:
    n = 12
    return pd.DataFrame(
        {
            "event_id": ["e1"] * 5 + ["e2"] * 7,
            "symbol": ["AAA"] * 5 + ["BBB"] * 7,
            "exchange": ["bybit"] * n,
            "timestamp": np.arange(n),
            "pump_start_timestamp": [0] * n,
            "pump_detected_timestamp": [0] * n,
            "pump_age_minutes": np.arange(n) * 5.0,
            "current_price": [100.0] * n,
            "event_high_so_far": [100.0] * n,
            "climax_price": [100.0] * n,
            "snapshot_abs_idx": np.arange(n),
            "my_feature": [1.0, 2.0, np.nan, 4.0, 5.0] + [1.0] * 7,
            "always_missing_feature": [np.nan] * n,
            "primary_outcome": ["REVERSAL", "REVERSAL", "CONTINUATION", "NEUTRAL", "REVERSAL"] + ["CONTINUATION"] * 7,
        }
    )


def test_compute_snapshot_data_quality_basic():
    df = _snapshots_df()
    quality = diagnostics.compute_snapshot_data_quality(df)
    assert quality["n_snapshots"] == 12
    assert quality["n_events"] == 2
    assert quality["n_symbols"] == 2
    assert quality["feature_nan_rates"]["my_feature"] == pytest.approx(1 / 12)
    assert "always_missing_feature" in quality["fully_missing_features"]
    assert "primary_outcome" not in quality["feature_nan_rates"]  # outcome columns excluded from the feature NaN-rate report


def test_compute_snapshot_data_quality_empty():
    result = diagnostics.compute_snapshot_data_quality(pd.DataFrame())
    assert result["n_snapshots"] == 0


def test_compute_label_distribution_event_weighted_differs_from_snapshot_weighted():
    df = _snapshots_df().rename(columns={"primary_outcome": "primary_outcome"})
    df["primary_outcome"] = df["primary_outcome"]
    result = diagnostics.compute_label_distribution(df, label_families=("primary",))
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_snapshots"] == 12
    assert row["n_events"] == 2
    # e1 (5 snapshots): 3 REVERSAL, 1 CONTINUATION, 1 NEUTRAL -> event-internal p_reversal=0.6
    # e2 (7 snapshots): all CONTINUATION -> event-internal p_reversal=0.0
    # event-weighted (each event contributes 0.5 total weight): (0.6+0.0)/2 = 0.3
    assert row["event_weighted_p_reversal"] == pytest.approx(0.3)
    # snapshot-weighted: 3 REVERSAL out of 12 total = 0.25 - DIFFERENT from the event-weighted number,
    # demonstrating exactly the long-event-domination gap this diagnostic exists to surface.
    assert row["snapshot_weighted_p_reversal"] == pytest.approx(3 / 12)
    assert row["event_weighted_p_reversal"] != pytest.approx(row["snapshot_weighted_p_reversal"])
