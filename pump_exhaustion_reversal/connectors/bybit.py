"""Bybit USDT linear-perpetual connector.

Bybit's v5 market-data endpoints (kline / funding history / open interest)
return at most `limit` rows within [start, end], newest-first, and are
paginated by walking `end` backward toward `start` - the opposite direction
from Binance's forward start-cursor pagination. See
connectors/base.py:paginate_by_time_backward.

No taker-buy-volume field exists on Bybit klines (unlike Binance), so the
free full-history CVD proxy is unavailable here - Bybit CVD confidence is
"low" outside event-anchored fetch_trades_window() calls (project plan).
"""

from __future__ import annotations

import ccxt
import pandas as pd

from ..data.normalizer import parse_timeframe_ms
from .base import DEFAULT_MAX_PAGES, http_get_json, paginate_by_time_backward

EXCHANGE_NAME = "bybit"

BASE_URL = "https://api.bybit.com"
KLINE_URL = f"{BASE_URL}/v5/market/kline"
FUNDING_URL = f"{BASE_URL}/v5/market/funding/history"
OI_URL = f"{BASE_URL}/v5/market/open-interest"

OHLCV_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base_volume",
]

_TF_TO_KLINE_INTERVAL = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720", "1d": "D",
}
_TF_TO_OI_INTERVAL = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1d"}


def make_exchange() -> ccxt.bybit:
    """ccxt instance for calls that ARE well unified (trades, tickers)."""
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap", "defaultSubType": "linear"}})
    ex.load_markets()
    return ex


def _unwrap(raw: dict, url: str) -> list[dict]:
    if raw.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error for {url}: retCode={raw.get('retCode')} retMsg={raw.get('retMsg')}")
    return raw.get("result", {}).get("list", [])


def fetch_ohlcv_paged(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1000,
    max_requests: int = DEFAULT_MAX_PAGES,
) -> pd.DataFrame:
    interval = _TF_TO_KLINE_INTERVAL.get(timeframe)
    if interval is None:
        raise ValueError(f"Unsupported timeframe for Bybit kline: {timeframe!r}")
    step_ms = parse_timeframe_ms(timeframe)

    def fetch_page(cursor_end: int | None) -> list[dict]:
        params = {"category": "linear", "symbol": symbol, "interval": interval, "start": since_ms, "limit": limit}
        if cursor_end is not None:
            params["end"] = cursor_end
        raw = http_get_json(KLINE_URL, params)
        rows = _unwrap(raw, KLINE_URL)
        # each row: [startTime, open, high, low, close, volume, turnover] (strings), newest-first
        return [
            {
                "timestamp": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "quote_volume": float(r[6]),
            }
            for r in rows
        ]

    rows = paginate_by_time_backward(
        fetch_page,
        ts_of=lambda r: r["timestamp"],
        since_ms=since_ms,
        until_ms=until_ms,
        step_ms_on_stall=step_ms,
        max_pages=max_requests,
    )
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = pd.DataFrame(rows)
    df["taker_buy_base_volume"] = pd.NA
    return (
        df.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)[OHLCV_COLUMNS]
    )


def fetch_funding_history(
    symbol: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 200,
    max_requests: int = 500,
) -> pd.DataFrame:
    def fetch_page(cursor_end: int | None) -> list[dict]:
        params = {"category": "linear", "symbol": symbol, "startTime": since_ms, "limit": limit}
        if cursor_end is not None:
            params["endTime"] = cursor_end
        raw = http_get_json(FUNDING_URL, params)
        rows = _unwrap(raw, FUNDING_URL)
        return [
            {"timestamp": int(r["fundingRateTimestamp"]), "funding_rate": float(r["fundingRate"])}
            for r in rows
        ]

    rows = paginate_by_time_backward(
        fetch_page,
        ts_of=lambda r: r["timestamp"],
        since_ms=since_ms,
        until_ms=until_ms,
        step_ms_on_stall=8 * 3_600_000,
        max_pages=max_requests,
    )
    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def fetch_oi_history(
    symbol: str,
    interval: str = "5m",
    since_ms: int | None = None,
    until_ms: int | None = None,
    limit: int = 200,
    max_requests: int = 2000,
) -> pd.DataFrame:
    """Best-effort OI history. Verified against the live API: Bybit's
    retention here is notably deeper than Binance's - a full, gapless 60-day
    5m history came back on the first real check - but still well short of a
    12-24 month backtest window. A real exchange-side limit either way
    (project plan / README limitations)."""
    interval_time = _TF_TO_OI_INTERVAL.get(interval)
    if interval_time is None:
        raise ValueError(f"interval must be one of {sorted(_TF_TO_OI_INTERVAL)}, got {interval!r}")
    since_ms = since_ms or 0

    def fetch_page(cursor_end: int | None) -> list[dict]:
        params = {"category": "linear", "symbol": symbol, "intervalTime": interval_time, "startTime": since_ms, "limit": limit}
        if cursor_end is not None:
            params["endTime"] = cursor_end
        raw = http_get_json(OI_URL, params)
        rows = _unwrap(raw, OI_URL)
        return [{"timestamp": int(r["timestamp"]), "open_interest": float(r["openInterest"])} for r in rows]

    rows = paginate_by_time_backward(
        fetch_page,
        ts_of=lambda r: r["timestamp"],
        since_ms=since_ms,
        until_ms=until_ms,
        step_ms_on_stall=parse_timeframe_ms(interval),
        max_pages=max_requests,
    )
    df = pd.DataFrame(rows, columns=["timestamp", "open_interest"])
    df["open_interest_value"] = pd.NA
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
