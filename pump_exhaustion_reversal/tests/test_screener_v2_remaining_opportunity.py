from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import remaining_opportunity as ro


def _ohlcv(rows: list[tuple[float, float]]) -> pd.DataFrame:
    # rows: (high, low)
    return pd.DataFrame([{"high": h, "low": l} for h, l in rows])


def test_compute_remaining_opportunity_tracks_upside_then_full_horizon_downside():
    # bar0 (anchor): high=100 low=98 (== event_high_so_far, no extension yet)
    # bar1: high=105 low=103 (new peak=105)
    # bar2: high=106 low=104 (new peak=106)
    # bar3: high=105 low=98  -> drawdown from peak(106) = 7.55%, not yet reversal
    # bar4: high=104 low=94  -> drawdown from peak(106) = 11.3% -> REVERSAL confirmed here (bars_to_reversal=4)
    # bar5: high=95  low=80  -> drawdown from peak(106) = 24.5% -> becomes the new (larger) max, walk does NOT stop early
    ohlcv = _ohlcv([(100, 98), (105, 103), (106, 104), (105, 98), (104, 94), (95, 80)])
    result = ro.compute_remaining_opportunity(ohlcv, anchor_idx=0, event_high_so_far=100.0, horizon_bars=5, reversal_drawdown_pct=0.10)
    assert result["valid"] is True
    assert result["additional_upside_before_reversal_pct"] == pytest.approx((106 - 100) / 100, abs=1e-6)
    assert result["remaining_downside_pct"] == pytest.approx((106 - 80) / 106, abs=1e-6)
    assert result["bars_to_reversal"] == 4
    assert result["reversal_confirmed"] is True
    assert result["right_censored"] is False


def test_compute_remaining_opportunity_no_upside_when_price_never_exceeds_anchor_high():
    # price only ever falls from the anchor's own high - additional_upside should be exactly 0
    ohlcv = _ohlcv([(100, 98), (98, 95), (95, 90), (90, 85), (85, 75), (75, 65)])
    result = ro.compute_remaining_opportunity(ohlcv, anchor_idx=0, event_high_so_far=100.0, horizon_bars=5, reversal_drawdown_pct=0.10)
    assert result["additional_upside_before_reversal_pct"] == pytest.approx(0.0)
    assert result["bars_to_reversal"] is not None  # (100-90)/100=10% reached by bar3


def test_compute_remaining_opportunity_right_censored_when_data_runs_out_before_reversal():
    # only 3 bars of data exist; requested horizon_bars=10 - the walk runs out of data long before either the
    # nominal horizon or a reversal, and price never gets close to a 10% drawdown
    ohlcv = _ohlcv([(100, 99), (101, 99), (102, 100)])
    result = ro.compute_remaining_opportunity(ohlcv, anchor_idx=0, event_high_so_far=100.0, horizon_bars=10, reversal_drawdown_pct=0.10)
    assert result["bars_to_reversal"] is None
    assert result["reversal_confirmed"] is False
    assert result["right_censored"] is True


def test_compute_remaining_opportunity_invalid_when_event_high_missing():
    ohlcv = _ohlcv([(100, 98), (101, 99)])
    result = ro.compute_remaining_opportunity(ohlcv, anchor_idx=0, event_high_so_far=np.nan, horizon_bars=5)
    assert result == {"valid": False}


def test_compute_remaining_opportunity_for_first_alerts_looks_up_by_exchange_symbol_and_skips_missing():
    first_alerts = pd.DataFrame(
        [
            {"event_id": "e1", "exchange": "binance", "symbol": "FOOUSDT", "snapshot_abs_idx": 0, "event_high_so_far": 100.0},
            {"event_id": "e2", "exchange": "bybit", "symbol": "MISSINGUSDT", "snapshot_abs_idx": 0, "event_high_so_far": 100.0},
        ]
    )
    ohlcv_by_key = {("binance", "FOOUSDT"): _ohlcv([(100, 98), (105, 103), (106, 104), (105, 98), (104, 94), (95, 80)])}
    result = ro.compute_remaining_opportunity_for_first_alerts(first_alerts, ohlcv_by_key, horizon_bars=5)
    assert list(result["event_id"]) == ["e1"]  # e2 skipped - its (exchange, symbol) key isn't in ohlcv_by_key
    assert result["bars_to_reversal"].iloc[0] == 4
