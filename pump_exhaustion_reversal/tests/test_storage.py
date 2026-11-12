from __future__ import annotations

import pandas as pd
import pytest

from pump_exhaustion_reversal.data.normalizer import (
    normalize_symbol,
    ohlcv_rows_to_df,
    parse_timeframe_ms,
)
from pump_exhaustion_reversal.data.storage import merge_write, raw_path, read_range


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTC/USDT:USDT", "BTCUSDT"),
        ("BTC/USDT", "BTCUSDT"),
        ("btcusdt", "BTCUSDT"),
        ("1000PEPE/USDT:USDT", "1000PEPEUSDT"),
    ],
)
def test_normalize_symbol(raw, expected):
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize(
    "tf,expected_ms",
    [("1m", 60_000), ("5m", 300_000), ("15m", 900_000), ("1h", 3_600_000), ("4h", 14_400_000), ("1d", 86_400_000)],
)
def test_parse_timeframe_ms(tf, expected_ms):
    assert parse_timeframe_ms(tf) == expected_ms


def test_ohlcv_rows_to_df_optional_columns_are_nan_not_zero():
    rows = [[1_000, 1.0, 1.1, 0.9, 1.05, 100.0]]
    df = ohlcv_rows_to_df(rows)
    assert df.loc[0, "volume"] == 100.0
    assert pd.isna(df.loc[0, "quote_volume"])
    assert pd.isna(df.loc[0, "taker_buy_base_volume"])

    df2 = ohlcv_rows_to_df(rows, quote_volume=[105.0], taker_buy_base_volume=[60.0])
    assert df2.loc[0, "quote_volume"] == 105.0
    assert df2.loc[0, "taker_buy_base_volume"] == 60.0


def test_merge_write_is_idempotent_and_dedups(tmp_path, monkeypatch):
    import pump_exhaustion_reversal.data.storage as storage_mod

    monkeypatch.setattr(storage_mod, "RAW_ROOT", tmp_path)
    path = raw_path("bybit", "ohlcv", "BTCUSDT", timeframe="5m")

    batch1 = ohlcv_rows_to_df([[1000, 1, 1, 1, 1, 10], [1300, 2, 2, 2, 2, 20]])
    merge_write(path, batch1, dedup_subset=["timestamp"])

    # overlapping batch: ts=1300 is a corrected re-fetch (should win, keep="last"),
    # ts=1600 is genuinely new
    batch2 = ohlcv_rows_to_df([[1300, 2, 2, 2, 2.5, 25], [1600, 3, 3, 3, 3, 30]])
    result = merge_write(path, batch2, dedup_subset=["timestamp"])

    assert list(result["timestamp"]) == [1000, 1300, 1600]
    assert result.loc[result["timestamp"] == 1300, "close"].iloc[0] == 2.5  # corrected value won
    assert result.loc[result["timestamp"] == 1300, "volume"].iloc[0] == 25

    # re-running the exact same write again must not change anything (true idempotency)
    result2 = merge_write(path, batch2, dedup_subset=["timestamp"])
    pd.testing.assert_frame_equal(result, result2)


def test_read_range_filters_by_timestamp(tmp_path, monkeypatch):
    import pump_exhaustion_reversal.data.storage as storage_mod

    monkeypatch.setattr(storage_mod, "RAW_ROOT", tmp_path)
    path = raw_path("binance", "ohlcv", "ETHUSDT", timeframe="1h")
    rows = ohlcv_rows_to_df([[t, 1, 1, 1, 1, 1] for t in range(0, 10_000, 1000)])
    merge_write(path, rows, dedup_subset=["timestamp"])

    windowed = read_range("binance", "ohlcv", "ETHUSDT", timeframe="1h", start_ms=2000, end_ms=5000)
    assert list(windowed["timestamp"]) == [2000, 3000, 4000, 5000]
