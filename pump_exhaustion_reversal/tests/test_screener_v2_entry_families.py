from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import entry_families as ef


def _ohlcv(rows: list[tuple[float, float]]) -> pd.DataFrame:
    # rows: (high, low)
    return pd.DataFrame([{"open": h, "high": h, "low": l, "close": h} for h, l in rows])


def test_family_a_immediate_signals_on_the_alert_bar_itself():
    result = ef.detect_family_a_immediate(alert_abs_idx=5, event_high_so_far=100.0)
    assert result["valid"] is True
    assert result["signal_bar_idx"] == 5
    assert result["stop_reference_price"] == pytest.approx(100.0)


def test_family_a_immediate_invalid_when_event_high_missing():
    result = ef.detect_family_a_immediate(alert_abs_idx=5, event_high_so_far=np.nan)
    assert result["valid"] is False


def test_family_b_retest_confirms_rebound_then_stall():
    # bar0 = the alert bar itself (unused by the walk, start = alert_abs_idx+1)
    # bar1: low=100 (running_low=100), high=100 -> high-running_low=0, no rebound yet
    # bar2: low=98 (new running_low=98), high=98 -> 0, no rebound yet
    # bar3: high=98.6 -> 98.6-98=0.6 >= 0.5*ATR(1.0) -> REBOUND CONFIRMED, peak=98.6
    # bar4: high=99.0 > 98.6 -> new peak=99.0, stall counter resets
    # bar5: high=98.8 (not a new high) -> stall=1
    # bar6: high=98.7 (not a new high) -> stall=2 -> STALL CONFIRMED here
    ohlcv = _ohlcv([(100, 100), (100, 100), (98, 98), (98.6, 98), (99.0, 98.5), (98.8, 98.3), (98.7, 98.2)])
    atr_series = pd.Series([1.0] * len(ohlcv))
    result = ef.detect_family_b_retest(ohlcv, alert_abs_idx=0, atr_series=atr_series, max_wait_bars=10)
    assert result["valid"] is True
    assert result["signal_bar_idx"] == 6
    assert result["stop_reference_price"] == pytest.approx(99.0)
    assert result["rebound_confirmed_bar_idx"] == 3


def test_family_b_retest_invalid_when_no_rebound_within_horizon():
    # price only ever falls - never rebounds by 0.5 ATR off its own running low
    ohlcv = _ohlcv([(100, 100), (99, 98), (97, 96), (95, 94), (93, 92)])
    atr_series = pd.Series([1.0] * len(ohlcv))
    result = ef.detect_family_b_retest(ohlcv, alert_abs_idx=0, atr_series=atr_series, max_wait_bars=10)
    assert result["valid"] is False
    assert result["reason"] == "no_rebound_within_horizon"


def test_family_c_failed_continuation_confirms_attempt_then_rejection():
    # event_high_so_far=100, approach_tolerance_atr=1.0 -> approach_price=99, invalidation_price=105
    # bar1: high=98 -> below approach_price(99), no attempt yet
    # bar2: high=99.5 -> >=99 -> ATTEMPT CONFIRMED, peak=99.5
    # bar3: high=99.2 (not a new high) -> stall=1
    # bar4: high=99.0 (not a new high) -> stall=2 -> REJECTION CONFIRMED here
    ohlcv = _ohlcv([(100, 100), (98, 95), (99.5, 97), (99.2, 97), (99.0, 96)])
    atr_series = pd.Series([1.0] * len(ohlcv))
    result = ef.detect_family_c_failed_continuation(ohlcv, alert_abs_idx=0, event_high_so_far=100.0, atr_series=atr_series, max_wait_bars=10)
    assert result["valid"] is True
    assert result["signal_bar_idx"] == 4
    assert result["stop_reference_price"] == pytest.approx(99.5)
    assert result["attempt_confirmed_bar_idx"] == 2


def test_family_c_failed_continuation_invalidated_by_real_breakout():
    # bar2's high=106 exceeds the invalidation price (105) BEFORE any attempt/rejection logic runs
    ohlcv = _ohlcv([(100, 100), (98, 95), (106, 97), (107, 100), (108, 101)])
    atr_series = pd.Series([1.0] * len(ohlcv))
    result = ef.detect_family_c_failed_continuation(ohlcv, alert_abs_idx=0, event_high_so_far=100.0, atr_series=atr_series, max_wait_bars=10)
    assert result["valid"] is False
    assert result["reason"] == "continuation_invalidated_thesis"


def test_family_c_failed_continuation_invalid_when_event_high_missing():
    ohlcv = _ohlcv([(100, 100), (98, 95)])
    atr_series = pd.Series([1.0] * len(ohlcv))
    result = ef.detect_family_c_failed_continuation(ohlcv, alert_abs_idx=0, event_high_so_far=np.nan, atr_series=atr_series, max_wait_bars=10)
    assert result["valid"] is False
