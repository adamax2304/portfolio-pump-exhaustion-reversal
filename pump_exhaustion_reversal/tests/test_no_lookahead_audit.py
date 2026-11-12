"""Phase 2D point 6: explicit no-lookahead audit for the two signals the
OI_ROLLOVER + VOLUME_DECAY frozen candidate depends on (see
config/oi_volume_reversal_candidate_v1.yaml). features/oi.py's own
causality is covered directly in test_features_advanced.py
(test_detect_oi_rollover_is_causal_no_lookahead) - this file covers the
PIPELINE-level guarantee that _build_event_inputs's volume-decay peak
(fixed once, at pump_detected_ts) is never influenced by bars that occur
after detection, which is where volume_decay_ratio's actual "no lookahead"
property lives (the feature function itself just divides by whatever
peak_volume scalar it's handed - see features/volume.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest.simulator import _build_event_inputs
from pump_exhaustion_reversal.config import AppConfig, PumpDetectorConfig
from pump_exhaustion_reversal.data.feature_store import build_feature_frame

_CFG = AppConfig(pump_detector=PumpDetectorConfig(beta_lookback_days=2))


def _symbol_with_pump_then_tail(future_spike: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_n = 300
    # flat -> pump run-up (bars 0-9) -> climax (bar 10, volume peaks at 500) -> decay (bars 11-30, volume back to ~50)
    pattern_close = [50.0] * 2 + list(np.linspace(50, 130, 9)) + [128.0] + list(np.linspace(126, 120, 19))
    pattern_volume = [100.0] * 2 + [200.0] * 9 + [500.0] + [50.0] * 19  # climax bar (idx 10 in the pattern) has the peak volume
    trail_n = 200
    n_total = baseline_n + len(pattern_close) + trail_n

    close = np.full(n_total, 50.0)
    volume = np.full(n_total, 100.0)
    for i, (c, v) in enumerate(zip(pattern_close, pattern_volume)):
        close[baseline_n + i] = c
        volume[baseline_n + i] = v
    tail_start = baseline_n + len(pattern_close)
    close[tail_start:] = pattern_close[-1]
    volume[tail_start:] = 50.0
    if future_spike:
        # a MASSIVE volume spike late in the tail - well after detection/horizon -
        # if peak_volume were ever recomputed with lookahead, this would raise it
        # and change volume_decay_ratio at every earlier bar.
        volume[tail_start + 50 : tail_start + 55] = 50_000.0

    high = close * 1.001
    low = close * 0.999
    open_ = close.copy()
    ohlcv = pd.DataFrame(
        {
            "timestamp": np.arange(n_total) * 300_000, "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": volume * close, "taker_buy_base_volume": volume * 0.5,
        }
    )
    rng = np.random.default_rng(1)
    btc_close = 60_000.0 * np.exp(np.cumsum(rng.normal(0, 0.0003, n_total)))
    btc_ohlcv = pd.DataFrame(
        {
            "timestamp": np.arange(n_total) * 300_000, "open": btc_close, "high": btc_close * 1.0005,
            "low": btc_close * 0.9995, "close": btc_close, "volume": [1000.0] * n_total,
            "quote_volume": btc_close * 1000.0, "taker_buy_base_volume": [500.0] * n_total,
        }
    )
    return ohlcv, btc_ohlcv


def test_volume_decay_ratio_peak_is_fixed_at_detection_not_lookahead():
    detect_idx = 300 + 10  # the climax bar itself (peak volume 500)

    ohlcv_a, btc_a = _symbol_with_pump_then_tail(future_spike=False)
    feature_frame_a = build_feature_frame(ohlcv_a, btc_a, "5m", _CFG)
    inputs_a = _build_event_inputs(ohlcv_a, feature_frame_a, detect_idx, "5m", _CFG, horizon_bars=200)

    ohlcv_b, btc_b = _symbol_with_pump_then_tail(future_spike=True)
    feature_frame_b = build_feature_frame(ohlcv_b, btc_b, "5m", _CFG)
    inputs_b = _build_event_inputs(ohlcv_b, feature_frame_b, detect_idx, "5m", _CFG, horizon_bars=200)

    # a bar shortly after detection - long before the future-only spike (which
    # only exists in ohlcv_b, at tail_start+50..55, far beyond this comparison point).
    check_local_idx = inputs_a.local_detect_idx + 15
    ratio_a = inputs_a.volume_decay_ratio.iloc[check_local_idx]
    ratio_b = inputs_b.volume_decay_ratio.iloc[check_local_idx]
    assert ratio_a == pytest.approx(ratio_b)
    assert bool(inputs_a.volume_decayed.iloc[check_local_idx]) == bool(inputs_b.volume_decayed.iloc[check_local_idx])


def test_volume_decay_ratio_peak_matches_manual_pre_detection_max():
    detect_idx = 300 + 10
    ohlcv, btc = _symbol_with_pump_then_tail(future_spike=False)
    feature_frame = build_feature_frame(ohlcv, btc, "5m", _CFG)
    inputs = _build_event_inputs(ohlcv, feature_frame, detect_idx, "5m", _CFG, horizon_bars=200)

    local_detect_idx = inputs.local_detect_idx
    expected_peak = float(inputs.df_slice["volume"].iloc[: local_detect_idx + 1].max())
    check_local_idx = local_detect_idx + 15
    actual_volume = float(inputs.df_slice["volume"].iloc[check_local_idx])
    assert inputs.volume_decay_ratio.iloc[check_local_idx] == pytest.approx(actual_volume / expected_peak)
