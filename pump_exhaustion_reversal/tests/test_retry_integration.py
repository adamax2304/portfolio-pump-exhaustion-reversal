"""Stage 2A: end-to-end proof that _run_single_event actually exercises a
reversal-attempt retry (not just the isolated pieces in
test_post_pump_reslicing.py / test_reversal_attempts.py) - a synthetic
symbol with an early irrelevant break, a real post-climax break, then a
genuine new pump high (which must re-arm the event as attempt 2 and re-anchor
the climax forward, not just terminate it)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pump_exhaustion_reversal.backtest.simulator import MarketBaseline, _run_single_event
from pump_exhaustion_reversal.config import AppConfig, PumpDetectorConfig
from pump_exhaustion_reversal.data.feature_store import build_feature_frame

_FAST_WARMUP_CFG = AppConfig(pump_detector=PumpDetectorConfig(beta_lookback_days=2))

# (idx, open, high, low, close), relative to where this pattern is embedded.
# Early break (0-11, irrelevant) -> climax1 (22) -> break1 (32) -> a genuine
# NEW HIGH at 37 (must trigger a retry: new climax, attempt_count -> 2).
_PATTERN_ROWS = [
    (0, 50, 50, 50, 50), (1, 50, 50, 50, 50),
    (2, 50, 55, 50, 54), (3, 54, 54, 52, 52.5), (4, 52.5, 53, 48, 48.5),
    (5, 48.5, 51, 48.5, 50.5), (6, 50.5, 52, 50, 51.5), (7, 51.5, 53, 51, 52.5),
    (8, 52.5, 53, 49, 49.5), (9, 49.5, 51, 49, 50.5), (10, 50.5, 51, 49.5, 50),
    (11, 50, 50, 46, 46.5), (12, 46.5, 47, 45, 45.5), (13, 45.5, 46, 45, 45.2), (14, 45.2, 46, 45, 45.5),
    (15, 46, 55, 45, 54), (16, 54, 65, 53, 64), (17, 64, 78, 63, 77),
    (18, 77, 95, 76, 94), (19, 94, 115, 93, 113), (20, 113, 125, 112, 123),
    (21, 123, 129, 122, 128), (22, 128, 131, 127, 129),  # climax1: high=131
    (23, 129, 129, 120, 121), (24, 121, 122, 116, 117),
    (25, 117, 118, 115, 115.5), (26, 115.5, 119, 115, 118.5), (27, 118.5, 121, 118, 120.5),
    (28, 120.5, 121, 119, 119.5), (29, 119.5, 120, 118, 118.5),
    (30, 118.5, 119.5, 118, 119), (31, 119, 119.5, 118.5, 119),
    (32, 119, 119, 114, 114.5), (33, 114.5, 115, 113, 113.5), (34, 113.5, 114, 113, 113.2),
    (35, 113.2, 118, 113, 117), (36, 117, 122, 116, 121),
    (37, 121, 133, 120, 132),  # new high: 133 > pre_break_swing_high(121) from break1
]
_BASELINE_N = 300
_TRAIL_N = 400
_CLIMAX1_LOCAL = 22


def _build_symbol_data():
    n_total = _BASELINE_N + len(_PATTERN_ROWS) + _TRAIL_N
    close = np.full(n_total, 50.0)
    high = np.full(n_total, 50.0)
    low = np.full(n_total, 50.0)
    open_ = np.full(n_total, 50.0)
    for (i, o, h, l, c) in _PATTERN_ROWS:
        idx = _BASELINE_N + i
        open_[idx], high[idx], low[idx], close[idx] = o, h, l, c
    tail_price = _PATTERN_ROWS[-1][4]
    close[_BASELINE_N + len(_PATTERN_ROWS):] = tail_price
    high[_BASELINE_N + len(_PATTERN_ROWS):] = tail_price
    low[_BASELINE_N + len(_PATTERN_ROWS):] = tail_price
    open_[_BASELINE_N + len(_PATTERN_ROWS):] = tail_price

    volume = np.full(n_total, 100.0)
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
    return ohlcv, btc_ohlcv, n_total


def test_run_single_event_retries_on_a_genuine_new_pump_high():
    ohlcv, btc_ohlcv, n_total = _build_symbol_data()
    cfg = _FAST_WARMUP_CFG
    feature_frame = build_feature_frame(ohlcv, btc_ohlcv, "5m", cfg)

    climax1_abs = _BASELINE_N + _CLIMAX1_LOCAL
    detect_idx = _BASELINE_N + 25  # detection lags the real top by a few bars, as is typical

    pump_frame = pd.DataFrame(
        {"pump_detected": [False] * n_total, "pump_score": [0.0] * n_total, "data_confidence": ["HIGH"] * n_total}
    )
    pump_frame.loc[detect_idx, "pump_detected"] = True
    pump_frame.loc[detect_idx, "pump_score"] = 90.0

    event_record, _ = _run_single_event(
        "test:TESTUSDT:1", "bybit", "TESTUSDT", detect_idx, ohlcv, feature_frame, pump_frame, "5m", cfg,
        None, 80.0, 5000.0, MarketBaseline(), 200,
    )

    # The retry actually happened - not just eligible in isolation, but
    # observed end-to-end through the real per-bar loop.
    assert event_record.attempt_count == 2
    # pump_peak_ts (EventRecord's own field) stays tied to attempt 1's climax
    # even after a retry re-anchors the LIVE structure to a new one.
    assert event_record.pump_peak_ts == int(ohlcv["timestamp"].iloc[climax1_abs])
    # climax_local_idx (wide/df_slice-local) moved forward past attempt 1's
    # climax - proof the re-anchor on retry actually happened, not just the
    # counter incrementing without the structure changing underneath it.
    pump_start_idx = detect_idx - 288  # lookback_bars for the default pump_detector.windows
    climax1_wide_local = climax1_abs - pump_start_idx
    assert event_record.climax_local_idx is not None
    assert event_record.climax_local_idx > climax1_wide_local

    # Phase 2B: per-attempt history - one entry per attempt, attempt 1's
    # climax preserved (not overwritten by the retry's re-anchor).
    assert len(event_record.attempt_climax_local_idxs) == 2
    assert event_record.attempt_climax_local_idxs[0] == climax1_wide_local
    assert event_record.attempt_climax_local_idxs[1] == event_record.climax_local_idx
    assert len(event_record.attempt_structure_break_local_idxs) == 2
    assert len(event_record.attempt_exhaustion_score_peaks) == 2
    assert len(event_record.attempt_invalidation_reason_codes) == 2
    # attempt 1 concluded via the new-pump-high retry - recorded with that reason code.
    assert event_record.attempt_invalidation_reason_codes[0] == "new_pump_high"


def test_reversal_attempts_disabled_terminates_on_first_new_pump_high_like_before_stage_2a():
    """Regression guard: cfg.reversal_attempts.enabled=False must reproduce
    the pre-Stage-2A always-terminal behavior - same scenario as the retry
    test above, but the very first new_pump_high ends the event outright."""
    ohlcv, btc_ohlcv, n_total = _build_symbol_data()
    cfg = _FAST_WARMUP_CFG.model_copy(
        update={"reversal_attempts": _FAST_WARMUP_CFG.reversal_attempts.model_copy(update={"enabled": False})}
    )
    feature_frame = build_feature_frame(ohlcv, btc_ohlcv, "5m", cfg)
    detect_idx = _BASELINE_N + 25

    pump_frame = pd.DataFrame(
        {"pump_detected": [False] * n_total, "pump_score": [0.0] * n_total, "data_confidence": ["HIGH"] * n_total}
    )
    pump_frame.loc[detect_idx, "pump_detected"] = True
    pump_frame.loc[detect_idx, "pump_score"] = 90.0

    event_record, _ = _run_single_event(
        "test:TESTUSDT:1", "bybit", "TESTUSDT", detect_idx, ohlcv, feature_frame, pump_frame, "5m", cfg,
        None, 80.0, 5000.0, MarketBaseline(), 200,
    )
    assert event_record.attempt_count == 1
    assert event_record.final_state == "INVALIDATED"
    assert "new pump high" in (event_record.invalidation_reason or "")
