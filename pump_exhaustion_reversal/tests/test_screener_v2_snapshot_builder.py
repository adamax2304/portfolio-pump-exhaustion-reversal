from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest.simulator import EventRecord, MarketBaseline, _run_single_event
from pump_exhaustion_reversal.config import AppConfig, PumpDetectorConfig
from pump_exhaustion_reversal.data.feature_store import build_feature_frame
from pump_exhaustion_reversal.screener_v2 import snapshot_builder as sb

from .test_retry_integration import _BASELINE_N, _build_symbol_data

_CFG = AppConfig(pump_detector=PumpDetectorConfig(beta_lookback_days=2))


def _real_event_and_data():
    ohlcv, btc_ohlcv, n_total = _build_symbol_data()
    feature_frame = build_feature_frame(ohlcv, btc_ohlcv, "5m", _CFG)
    detect_idx = _BASELINE_N + 25
    pump_frame = pd.DataFrame({"pump_detected": [False] * n_total, "pump_score": [0.0] * n_total, "data_confidence": ["HIGH"] * n_total})
    pump_frame.loc[detect_idx, "pump_detected"] = True
    pump_frame.loc[detect_idx, "pump_score"] = 90.0
    event_record, _ = _run_single_event(
        "test:TESTUSDT:1", "bybit", "TESTUSDT", detect_idx, ohlcv, feature_frame, pump_frame, "5m", _CFG,
        None, 80.0, 5000.0, MarketBaseline(), 200,
    )
    return event_record, ohlcv, feature_frame, btc_ohlcv


def test_build_snapshot_window_resolves_bounds():
    event, ohlcv, feature_frame, btc_ohlcv = _real_event_and_data()
    window = sb.build_snapshot_window(event, ohlcv, feature_frame, btc_ohlcv, "5m", _CFG, max_lifecycle_hours=2.0)
    assert window is not None
    assert window.detect_idx < window.end_idx
    # climax_idx_abs reflects the FINAL attempt's climax (see EventRecord.climax_local_idx's own
    # docstring) - on this retry fixture the final attempt's climax is a genuine new high AFTER
    # the original detection bar, so only pump_start_idx <= climax_idx_abs is guaranteed here,
    # not climax_idx_abs <= detect_idx (that only holds for single-attempt events).
    assert window.pump_start_idx <= window.climax_idx_abs
    # exactly 2h worth of bars at 5m = 24 bars beyond detection (or less if data runs out first)
    assert window.end_idx - window.detect_idx <= 24


def test_build_snapshot_window_none_when_detect_ts_out_of_range():
    event, ohlcv, feature_frame, btc_ohlcv = _real_event_and_data()
    bad_event = event.__class__(**{**event.__dict__, "pump_detected_ts": ohlcv["timestamp"].max() + 10_000_000})
    window = sb.build_snapshot_window(bad_event, ohlcv, feature_frame, btc_ohlcv, "5m", _CFG)
    assert window is None


def test_build_snapshot_window_clamps_climax_past_the_lifecycle_horizon():
    """On this retry fixture, climax_idx_abs (336, the FINAL attempt's
    climax) is past detect_idx (325) - with a short enough max_lifecycle_hours,
    it also falls past end_idx. Must clamp cleanly (no ValueError from
    _build_post_pump_inputs operating outside df_slice's bounds), not crash."""
    event, ohlcv, feature_frame, btc_ohlcv = _real_event_and_data()
    assert event.climax_local_idx is not None
    window = sb.build_snapshot_window(event, ohlcv, feature_frame, btc_ohlcv, "5m", _CFG, max_lifecycle_hours=0.5)
    assert window is not None
    assert window.climax_idx_abs == window.end_idx  # clamped, not the raw (out-of-window) EventRecord value
    assert window.climax_idx_abs <= window.end_idx
    assert len(window.post_pump.post_pump_df) <= len(window.df_slice)


def test_assemble_snapshots_identity_columns_and_row_count():
    event, ohlcv, feature_frame, btc_ohlcv = _real_event_and_data()
    window = sb.build_snapshot_window(event, ohlcv, feature_frame, btc_ohlcv, "5m", _CFG, max_lifecycle_hours=2.0)
    assert window is not None

    dummy_feature = pd.DataFrame({"dummy_col": np.arange(len(window.df_slice))}, index=window.df_slice.index)
    snapshots = sb.assemble_snapshots(window, [dummy_feature])

    expected_rows = window.end_idx - window.detect_idx + 1
    assert len(snapshots) == expected_rows
    assert list(snapshots["event_id"].unique()) == [event.event_id]
    assert (snapshots["symbol"] == "TESTUSDT").all()
    assert (snapshots["exchange"] == "bybit").all()
    # pump_age_minutes starts at (or very near) 0 at the first snapshot (the detection bar itself) and is strictly increasing
    assert snapshots["pump_age_minutes"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert (snapshots["pump_age_minutes"].diff().dropna() > 0).all()
    # event_high_so_far is non-decreasing (a running cummax)
    assert (snapshots["event_high_so_far"].diff().dropna() >= 0).all()
    assert "dummy_col" in snapshots.columns
