from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import data_quality
from pump_exhaustion_reversal.tests.test_metrics import _event


def test_classify_event_data_confidence_high_when_all_four_present():
    ev = _event(oi_zscore=1.0, short_liq_zscore=2.0, funding_percentile=50.0, cvd_divergence_magnitude_at_peak=0.1)
    assert data_quality.classify_event_data_confidence(ev) == "HIGH"


def test_classify_event_data_confidence_low_when_all_four_missing():
    ev = _event(oi_zscore=np.nan, short_liq_zscore=np.nan, funding_percentile=np.nan, cvd_divergence_magnitude_at_peak=np.nan)
    assert data_quality.classify_event_data_confidence(ev) == "LOW"


def test_classify_event_data_confidence_medium_when_some_present():
    ev = _event(oi_zscore=1.0, short_liq_zscore=np.nan, funding_percentile=np.nan, cvd_divergence_magnitude_at_peak=np.nan)
    assert data_quality.classify_event_data_confidence(ev) == "MEDIUM"


def test_classify_events_data_confidence_series():
    events = [
        _event(oi_zscore=1.0, short_liq_zscore=2.0, funding_percentile=50.0, cvd_divergence_magnitude_at_peak=0.1),
        _event(oi_zscore=np.nan, short_liq_zscore=np.nan, funding_percentile=np.nan, cvd_divergence_magnitude_at_peak=np.nan),
    ]
    result = data_quality.classify_events_data_confidence(events)
    assert result.tolist() == ["HIGH", "LOW"]


def test_summarize_availability_computes_fractions():
    df = pd.DataFrame(
        {
            "has_oi_file": [True, True, False, False],
            "has_funding_file": [True, False, False, False],
            "has_taker_buy_volume_cvd": [True, True, True, True],
            "has_oi_notional": [False, False, False, False],
        }
    )
    summary = data_quality.summarize_availability(df)
    assert summary["n_symbols"] == 4
    assert summary["pct_with_oi_file"] == pytest.approx(0.5)
    assert summary["pct_with_funding_file"] == pytest.approx(0.25)
    assert summary["pct_with_taker_buy_volume_cvd"] == pytest.approx(1.0)
    assert summary["pct_with_oi_notional"] == pytest.approx(0.0)


def test_summarize_availability_empty():
    summary = data_quality.summarize_availability(pd.DataFrame())
    assert summary["n_symbols"] == 0
    assert np.isnan(summary["pct_with_oi_file"])


def test_compute_symbol_level_availability_reads_real_files(tmp_path, monkeypatch):
    import pump_exhaustion_reversal.data.storage as storage_mod

    monkeypatch.setattr(storage_mod, "RAW_ROOT", tmp_path)

    ohlcv_with_cvd = pd.DataFrame({"timestamp": [0, 1], "taker_buy_base_volume": [1.0, 2.0]})
    ohlcv_without_cvd = pd.DataFrame({"timestamp": [0, 1], "taker_buy_base_volume": [pd.NA, pd.NA]})
    storage_mod.merge_write(storage_mod.raw_path("binance", "ohlcv", "AAAUSDT", timeframe="5m"), ohlcv_with_cvd, dedup_subset=["timestamp"])
    storage_mod.merge_write(storage_mod.raw_path("binance", "ohlcv", "BBBUSDT", timeframe="5m"), ohlcv_without_cvd, dedup_subset=["timestamp"])
    oi_with_notional = pd.DataFrame({"timestamp": [0, 1], "open_interest": [10.0, 11.0], "open_interest_value": [100.0, 110.0]})
    storage_mod.merge_write(storage_mod.raw_path("binance", "oi", "AAAUSDT", timeframe="5m"), oi_with_notional, dedup_subset=["timestamp"])
    # BBBUSDT: no OI file at all, no funding file at all - the "cheap/file-existence" case

    availability = data_quality.compute_symbol_level_availability("binance", timeframe="5m")
    assert set(availability["symbol"]) == {"AAAUSDT", "BBBUSDT"}
    aaa = availability.set_index("symbol").loc["AAAUSDT"]
    bbb = availability.set_index("symbol").loc["BBBUSDT"]
    assert aaa["has_oi_file"] and aaa["has_oi_notional"] and aaa["has_taker_buy_volume_cvd"]
    assert not bbb["has_oi_file"] and not bbb["has_taker_buy_volume_cvd"] and not bbb["has_funding_file"]


def test_render_data_quality_report_includes_spot_unavailable_notice():
    summaries = {
        "binance": {"n_symbols": 10, "pct_with_oi_file": 1.0, "pct_with_funding_file": 1.0, "pct_with_taker_buy_volume_cvd": 1.0, "pct_with_oi_notional": 1.0},
        "bybit": {"n_symbols": 5, "pct_with_oi_file": 1.0, "pct_with_funding_file": 1.0, "pct_with_taker_buy_volume_cvd": 0.0, "pct_with_oi_notional": 0.0},
    }
    confidences = {"binance": pd.Series(["HIGH", "LOW"]), "bybit": pd.Series(["LOW"])}
    report = data_quality.render_data_quality_report(summaries, confidences)
    assert "SPOT_DATA_UNAVAILABLE" in report
    assert "binance" in report and "bybit" in report
    assert "100%" in report or "0%" in report
