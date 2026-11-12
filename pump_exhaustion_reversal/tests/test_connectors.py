from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from pump_exhaustion_reversal.connectors.base import http_get_json, paginate_by_time, paginate_by_time_backward


def _http_error_response(status_code: int) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = {"ok": True}
    error = requests.HTTPError(f"{status_code} error")
    error.response = resp
    resp.raise_for_status.side_effect = error
    return resp


def test_http_get_json_does_not_retry_non_429_4xx():
    resp = _http_error_response(400)
    with patch("pump_exhaustion_reversal.connectors.base.requests.get", return_value=resp) as mock_get:
        with pytest.raises(RuntimeError, match="non-retryable HTTP 400"):
            http_get_json("http://x", {}, max_retries=4)
    assert mock_get.call_count == 1  # no retries burned on a guaranteed-repeat validation error


def test_http_get_json_retries_429_and_5xx():
    resp = _http_error_response(500)
    with patch("pump_exhaustion_reversal.connectors.base.requests.get", return_value=resp) as mock_get, \
         patch("pump_exhaustion_reversal.connectors.base.time.sleep"):
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            http_get_json("http://x", {}, max_retries=3)
    assert mock_get.call_count == 3


def test_http_get_json_succeeds_after_transient_failure():
    good = Mock()
    good.raise_for_status.return_value = None
    good.json.return_value = {"ok": True}
    bad = _http_error_response(503)
    with patch("pump_exhaustion_reversal.connectors.base.requests.get", side_effect=[bad, good]), \
         patch("pump_exhaustion_reversal.connectors.base.time.sleep"):
        result = http_get_json("http://x", {}, max_retries=3)
    assert result == {"ok": True}


def test_paginate_by_time_forward_basic():
    pages = {0: [{"timestamp": 0}, {"timestamp": 100}], 101: [{"timestamp": 200}], 201: []}

    def fetch_page(cursor):
        return pages.get(cursor, [])

    rows = paginate_by_time(
        fetch_page, ts_of=lambda r: r["timestamp"], since_ms=0, until_ms=None,
        step_ms_on_stall=50, max_pages=10, sleep_between_pages=0,
    )
    assert [r["timestamp"] for r in rows] == [0, 100, 200]


def test_paginate_by_time_stall_forces_advance_and_terminates():
    calls = {"n": 0}

    def fetch_page(cursor):
        calls["n"] += 1
        return [{"timestamp": 500}]  # same stale row forever regardless of cursor

    rows = paginate_by_time(
        fetch_page, ts_of=lambda r: r["timestamp"], since_ms=0, until_ms=5000,
        step_ms_on_stall=1000, max_pages=100, sleep_between_pages=0,
    )
    assert rows  # got at least the first page
    assert calls["n"] < 100  # terminated via until_ms + forced advance, not by exhausting max_pages


def test_paginate_by_time_backward_basic():
    def fetch_page(cursor_end):
        if cursor_end is None:
            return [{"timestamp": 300}, {"timestamp": 200}]
        if cursor_end == 199:
            return [{"timestamp": 100}, {"timestamp": 0}]
        return []

    rows = paginate_by_time_backward(
        fetch_page, ts_of=lambda r: r["timestamp"], since_ms=0, until_ms=None,
        step_ms_on_stall=50, max_pages=10, sleep_between_pages=0,
    )
    assert sorted(r["timestamp"] for r in rows) == [0, 100, 200, 300]


def test_paginate_by_time_backward_stalls_and_terminates():
    calls = {"n": 0}

    def fetch_page(cursor_end):
        calls["n"] += 1
        return [{"timestamp": 5000}]  # same stale row forever

    rows = paginate_by_time_backward(
        fetch_page, ts_of=lambda r: r["timestamp"], since_ms=0, until_ms=None,
        step_ms_on_stall=1000, max_pages=100, sleep_between_pages=0,
    )
    assert rows
    assert calls["n"] < 100


# ---- Binance connector -------------------------------------------------


def _binance_kline_row(ts: int, close: float = 1.5) -> list:
    return [ts, "1.0", "2.0", "0.5", str(close), "10.0", ts + 299_999, "15.0", 100, "6.0", "9.0", "0"]


def test_binance_fetch_ohlcv_paged(monkeypatch):
    import pump_exhaustion_reversal.connectors.binance as binance_mod

    all_rows = [_binance_kline_row(i * 300_000, close=1.0 + i) for i in range(7)]

    def fake_http_get_json(url, params, **kwargs):
        assert url == binance_mod.KLINES_URL
        start = params["startTime"]
        return [r for r in all_rows if r[0] >= start][: params["limit"]]

    monkeypatch.setattr(binance_mod, "http_get_json", fake_http_get_json)
    df = binance_mod.fetch_ohlcv_paged("BTCUSDT", "5m", since_ms=0, limit=3, max_requests=10)

    assert list(df["timestamp"]) == [i * 300_000 for i in range(7)]
    assert df["close"].tolist() == [1.0 + i for i in range(7)]
    assert df["taker_buy_base_volume"].tolist() == [6.0] * 7
    assert df["quote_volume"].tolist() == [15.0] * 7


def test_binance_fetch_ohlcv_paged_empty(monkeypatch):
    import pump_exhaustion_reversal.connectors.binance as binance_mod

    monkeypatch.setattr(binance_mod, "http_get_json", lambda url, params, **kw: [])
    df = binance_mod.fetch_ohlcv_paged("NOPEUSDT", "5m", since_ms=0)
    assert df.empty
    assert list(df.columns) == binance_mod.OHLCV_COLUMNS


def test_binance_fetch_funding_history(monkeypatch):
    import pump_exhaustion_reversal.connectors.binance as binance_mod

    step = 8 * 3_600_000
    all_rows = [{"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": i * step} for i in range(5)]

    def fake_http_get_json(url, params, **kwargs):
        assert url == binance_mod.FUNDING_URL
        start = params["startTime"]
        return [r for r in all_rows if r["fundingTime"] >= start][: params["limit"]]

    monkeypatch.setattr(binance_mod, "http_get_json", fake_http_get_json)
    df = binance_mod.fetch_funding_history("BTCUSDT", since_ms=0, limit=2, max_requests=10)
    assert list(df["timestamp"]) == [i * step for i in range(5)]
    assert (df["funding_rate"] == 0.0001).all()


def test_binance_fetch_oi_history_respects_interval_validation():
    import pump_exhaustion_reversal.connectors.binance as binance_mod

    with pytest.raises(ValueError):
        binance_mod.fetch_oi_history("BTCUSDT", interval="7m")


def test_binance_fetch_oi_history_clamps_old_since_ms_to_avoid_1130_rejection(monkeypatch):
    # Binance rejects (HTTP 400, code -1130) a startTime far enough in the
    # past instead of silently truncating it (confirmed against the live API) -
    # since_ms must be clamped to a recent floor before the request is made,
    # not passed straight through.
    import pump_exhaustion_reversal.connectors.binance as binance_mod

    seen_start_times = []

    def fake_http_get_json(url, params, **kw):
        seen_start_times.append(params.get("startTime"))
        return []

    monkeypatch.setattr(binance_mod, "http_get_json", fake_http_get_json)
    six_months_ago = binance_mod.utc_now_ms() - 180 * 86_400_000
    binance_mod.fetch_oi_history("BTCUSDT", interval="5m", since_ms=six_months_ago)

    assert seen_start_times  # at least one request was made
    safe_floor = binance_mod.utc_now_ms() - 3 * 86_400_000
    assert seen_start_times[0] >= safe_floor - 1000  # clamped, not the raw 6-month-old value


# ---- Bybit connector ----------------------------------------------------


def test_bybit_fetch_ohlcv_paged(monkeypatch):
    import pump_exhaustion_reversal.connectors.bybit as bybit_mod

    # bybit returns newest-first; simulate two backward pages
    def fake_http_get_json(url, params, **kwargs):
        assert url == bybit_mod.KLINE_URL
        assert params["category"] == "linear"
        end = params.get("end")
        if end is None:
            rows = [[600_000, "1", "2", "0.5", "3.0", "10", "20"], [300_000, "1", "2", "0.5", "2.0", "10", "20"]]
        elif end == 299_999:
            rows = [[0, "1", "2", "0.5", "1.0", "10", "20"]]
        else:
            rows = []
        return {"retCode": 0, "retMsg": "OK", "result": {"list": rows}}

    monkeypatch.setattr(bybit_mod, "http_get_json", fake_http_get_json)
    df = bybit_mod.fetch_ohlcv_paged("BTCUSDT", "5m", since_ms=0, limit=2, max_requests=10)

    assert list(df["timestamp"]) == [0, 300_000, 600_000]
    assert df["close"].tolist() == [1.0, 2.0, 3.0]
    assert df["taker_buy_base_volume"].isna().all()  # Bybit klines never carry this field


def test_bybit_fetch_funding_history(monkeypatch):
    import pump_exhaustion_reversal.connectors.bybit as bybit_mod

    def fake_http_get_json(url, params, **kwargs):
        assert url == bybit_mod.FUNDING_URL
        rows = [{"symbol": "BTCUSDT", "fundingRate": "0.0002", "fundingRateTimestamp": "100"}]
        return {"retCode": 0, "retMsg": "OK", "result": {"list": rows if params.get("endTime") is None else []}}

    monkeypatch.setattr(bybit_mod, "http_get_json", fake_http_get_json)
    df = bybit_mod.fetch_funding_history("BTCUSDT", since_ms=0)
    assert list(df["timestamp"]) == [100]
    assert df["funding_rate"].iloc[0] == 0.0002


def test_bybit_raises_on_error_retcode(monkeypatch):
    import pump_exhaustion_reversal.connectors.bybit as bybit_mod

    monkeypatch.setattr(
        bybit_mod, "http_get_json",
        lambda url, params, **kw: {"retCode": 10001, "retMsg": "bad symbol", "result": {}},
    )
    with pytest.raises(RuntimeError, match="bad symbol"):
        bybit_mod.fetch_oi_history("NOPEUSDT", since_ms=0)


# ---- Universe filter ------------------------------------------------------


def test_is_stablecoin_and_leveraged_token_predicates():
    from pump_exhaustion_reversal.connectors.ccxt_fallback import is_leveraged_token, is_stablecoin_base

    assert is_stablecoin_base("USDC")
    assert not is_stablecoin_base("BTC")
    assert is_leveraged_token("BTCUP")
    assert is_leveraged_token("ETHBULL")
    assert is_leveraged_token("BTC3L")
    assert not is_leveraged_token("BTC")


def test_apply_universe_filter_marks_low_confidence_instead_of_dropping():
    from pump_exhaustion_reversal.config import UniverseConfig
    from pump_exhaustion_reversal.data.universe import apply_universe_filter

    cfg = UniverseConfig(min_24h_quote_volume_usd=20_000_000, preferred_min_24h_quote_volume_usd=50_000_000)
    snapshot = pd.DataFrame(
        [
            {"symbol": "BTCUSDT", "market_symbol": "BTC/USDT:USDT", "base": "BTC",
             "quote_volume_24h": 500_000_000, "last_price": 60000, "spread_bps": 1, "listing_ts": None},
            {"symbol": "OBSCUREUSDT", "market_symbol": "OBSCURE/USDT:USDT", "base": "OBSCURE",
             "quote_volume_24h": 25_000_000, "last_price": 1.0, "spread_bps": 5, "listing_ts": None},
            {"symbol": "TOOSMALLUSDT", "market_symbol": "TOOSMALL/USDT:USDT", "base": "TOOSMALL",
             "quote_volume_24h": 1_000_000, "last_price": 1.0, "spread_bps": 5, "listing_ts": None},
            {"symbol": "USDCUSDT", "market_symbol": "USDC/USDT:USDT", "base": "USDC",
             "quote_volume_24h": 999_000_000, "last_price": 1.0, "spread_bps": 1, "listing_ts": None},
        ]
    )
    out = apply_universe_filter(snapshot, cfg)

    btc = out.set_index("symbol").loc["BTCUSDT"]
    assert btc["passed_filter"] and btc["liquidity_confidence"] == "HIGH"

    obscure = out.set_index("symbol").loc["OBSCUREUSDT"]
    assert obscure["passed_filter"] and obscure["liquidity_confidence"] == "LOW"

    too_small = out.set_index("symbol").loc["TOOSMALLUSDT"]
    assert not too_small["passed_filter"]  # below hard min-volume cutoff

    usdc = out.set_index("symbol").loc["USDCUSDT"]
    assert usdc["is_stablecoin"]
    assert not usdc["passed_filter"]  # exclude_stablecoins defaults to True


def test_apply_universe_filter_excludes_stablecoins_and_leveraged(monkeypatch):
    from pump_exhaustion_reversal.config import UniverseConfig
    from pump_exhaustion_reversal.data.universe import apply_universe_filter

    cfg = UniverseConfig(exclude_stablecoins=True, exclude_leveraged_tokens=True, min_24h_quote_volume_usd=0)
    snapshot = pd.DataFrame(
        [
            {"symbol": "USDCUSDT", "base": "USDC", "quote_volume_24h": 999_000_000, "spread_bps": 1, "listing_ts": None},
            {"symbol": "BTCUPUSDT", "base": "BTCUP", "quote_volume_24h": 999_000_000, "spread_bps": 1, "listing_ts": None},
            {"symbol": "BTCUSDT", "base": "BTC", "quote_volume_24h": 999_000_000, "spread_bps": 1, "listing_ts": None},
        ]
    )
    out = apply_universe_filter(snapshot, cfg)
    passed = set(out.loc[out["passed_filter"], "symbol"])
    assert passed == {"BTCUSDT"}
