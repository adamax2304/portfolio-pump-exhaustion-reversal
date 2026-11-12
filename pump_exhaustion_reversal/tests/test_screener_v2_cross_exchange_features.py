from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import cross_exchange_features as cef


def _linked_df(lead_exchange: str = "binance") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "underlying_event_id": "u1", "symbol": "FOOUSDT", "binance_event_id": "b1", "bybit_event_id": "y1",
                "binance_present": True, "bybit_present": True, "lead_exchange": lead_exchange,
                "detection_time_delta_minutes": 5.0, "binance_return": 0.1, "bybit_return": 0.08,
                "binance_pump_class": "X", "bybit_pump_class": "X",
            }
        ]
    )


def _snapshots_df() -> pd.DataFrame:
    binance = pd.DataFrame(
        {
            "event_id": ["b1"] * 3, "timestamp": [0, 300_000, 600_000],
            "is_new_high": [True, False, True], "current_return_from_start": [0.05, 0.06, 0.10],
            "volume_current_over_climax": [1.0, 1.1, 1.3], "oi_change_60m": [0.01, 0.02, 0.03],
        }
    )
    bybit = pd.DataFrame(
        {
            "event_id": ["y1"] * 3, "timestamp": [0, 300_000, 600_000],
            "is_new_high": [True, False, False], "current_return_from_start": [0.04, 0.05, 0.06],
            "volume_current_over_climax": [0.9, 1.0, 1.1], "oi_change_60m": [0.005, 0.01, 0.015],
        }
    )
    solo = pd.DataFrame(
        {
            "event_id": ["solo1"] * 2, "timestamp": [0, 300_000],
            "is_new_high": [True, False], "current_return_from_start": [0.02, 0.03],
            "volume_current_over_climax": [1.0, 1.0], "oi_change_60m": [0.0, 0.0],
        }
    )
    return pd.concat([binance, bybit, solo], ignore_index=True)


def test_cross_exchange_features_detects_new_high_divergence_and_return_divergence():
    snapshots_df = _snapshots_df()
    linked_df = _linked_df(lead_exchange="binance")
    result = cef.compute_cross_exchange_snapshot_features(snapshots_df, linked_df)
    assert len(result) == len(snapshots_df)

    binance_idx = snapshots_df.index[snapshots_df["event_id"] == "b1"]
    bybit_idx = snapshots_df.index[snapshots_df["event_id"] == "y1"]

    assert result.loc[binance_idx, "other_exchange_present"].all()
    assert result.loc[bybit_idx, "other_exchange_present"].all()

    # at the LAST bar (timestamp=600000), binance made a new high but bybit did not -> divergence True on both sides
    last_binance = binance_idx[-1]
    last_bybit = bybit_idx[-1]
    assert bool(result.loc[last_binance, "one_exchange_new_high_only"]) is True
    assert bool(result.loc[last_bybit, "one_exchange_new_high_only"]) is True

    # binance's own return (0.10) minus bybit's as-of return (0.06) at the same timestamp
    assert result.loc[last_binance, "return_divergence"] == pytest.approx(0.10 - 0.06)
    # bybit's own return (0.06) minus binance's as-of return (0.10)
    assert result.loc[last_bybit, "return_divergence"] == pytest.approx(0.06 - 0.10)

    assert result.loc[binance_idx[0], "leader_lagger_state"] == "leader"
    assert result.loc[bybit_idx[0], "leader_lagger_state"] == "lagger"


def test_cross_exchange_features_solo_event_gets_default_no_data():
    snapshots_df = _snapshots_df()
    linked_df = _linked_df()
    result = cef.compute_cross_exchange_snapshot_features(snapshots_df, linked_df)
    solo_idx = snapshots_df.index[snapshots_df["event_id"] == "solo1"]
    assert not result.loc[solo_idx, "other_exchange_present"].any()
    assert result.loc[solo_idx, "return_divergence"].isna().all()
    assert result.loc[solo_idx, "leader_lagger_state"].isna().all()


def test_cross_exchange_features_empty_linked_df_returns_all_defaults():
    snapshots_df = _snapshots_df()
    result = cef.compute_cross_exchange_snapshot_features(snapshots_df, pd.DataFrame(columns=["binance_present", "bybit_present"]))
    assert len(result) == len(snapshots_df)
    assert not result["other_exchange_present"].any()
