"""Binance USDT-M perpetual futures connector.

OHLCV uses Binance's raw REST klines endpoint (not ccxt's unified
fetch_ohlcv) so we capture quote_volume + taker_buy_base_volume on the same
request - a free, full-history CVD proxy (features/cvd.py). Funding and OI
use native REST with time-based pagination; OI depth is capped by Binance's
own ~30-day retention on that endpoint regardless of pagination (see
project plan - this is an exchange limitation, not a code one).
"""

from __future__ import annotations

import ccxt
import pandas as pd

from ..data.normalizer import parse_timeframe_ms, utc_now_ms
from .base import DEFAULT_MAX_PAGES, http_get_json, paginate_by_time

EXCHANGE_NAME = "binance"

KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"

OHLCV_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base_volume",
]

# Binance's openInterestHist accepts these period codes (not raw "5m"/"1h" strings).
_OI_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}


def make_exchange() -> ccxt.binance:
    """ccxt instance for the calls that ARE well unified (trades, tickers)."""
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    ex.load_markets()
    return ex


def fetch_ohlcv_paged(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1500,
    max_requests: int = DEFAULT_MAX_PAGES,
) -> pd.DataFrame:
    step_ms = parse_timeframe_ms(timeframe)

    def fetch_page(cursor: int) -> list[dict]:
        params = {"symbol": symbol, "interval": timeframe, "startTime": cursor, "limit": limit}
        if until_ms is not None:
            params["endTime"] = until_ms
        raw = http_get_json(KLINES_URL, params)
        return [
            {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "quote_volume": float(row[7]),
                "taker_buy_base_volume": float(row[9]),
            }
            for row in raw
        ]

    rows = paginate_by_time(
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
    return (
        df.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)[OHLCV_COLUMNS]
    )


def fetch_funding_history(
    symbol: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1000,
    max_requests: int = 500,
) -> pd.DataFrame:
    def fetch_page(cursor: int) -> list[dict]:
        params = {"symbol": symbol, "startTime": cursor, "limit": limit}
        if until_ms is not None:
            params["endTime"] = until_ms
        raw = http_get_json(FUNDING_URL, params)
        return [{"timestamp": int(r["fundingTime"]), "funding_rate": float(r["fundingRate"])} for r in raw]

    rows = paginate_by_time(
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
    limit: int = 500,
    max_requests: int = 50,
) -> pd.DataFrame:
    """Best-effort OI history. Empirically (verified against the live API, not
    just the docs) Binance's openInterestHist endpoint only ever returns
    roughly the last ~1.25 days of data regardless of how far back you ask -
    much shallower than the commonly-cited 30-day retention. Below some
    threshold (confirmed OK at 10 days back, confirmed rejected at 6 months
    back) it stops silently truncating and instead returns HTTP 400
    (code -1130 "parameter 'startTime' is invalid"). Since real data never
    goes back further than ~1.25 days anyway, since_ms is defensively
    clamped to a recent floor before the request is even made - a caller
    asking for 6 months of OI gets back ~1.25 days, not a wasted failed
    request (see project plan / README limitations). Bybit's equivalent
    (bybit.py) goes back much further (60+ days verified) - not comparable."""
    if interval not in _OI_PERIODS:
        raise ValueError(f"interval must be one of {_OI_PERIODS}, got {interval!r}")

    safe_floor_ms = utc_now_ms() - 3 * 86_400_000
    if since_ms is not None and since_ms < safe_floor_ms:
        since_ms = safe_floor_ms

    def fetch_page(cursor: int) -> list[dict]:
        params = {"symbol": symbol, "period": interval, "limit": limit}
        if cursor:
            params["startTime"] = cursor
        if until_ms is not None:
            params["endTime"] = until_ms
        raw = http_get_json(OI_HIST_URL, params)
        return [
            {
                "timestamp": int(r["timestamp"]),
                "open_interest": float(r["sumOpenInterest"]),
                "open_interest_value": float(r["sumOpenInterestValue"]),
            }
            for r in raw
        ]

    rows = paginate_by_time(
        fetch_page,
        ts_of=lambda r: r["timestamp"],
        since_ms=since_ms or 0,
        until_ms=until_ms,
        step_ms_on_stall=parse_timeframe_ms(interval),
        max_pages=max_requests,
    )
    df = pd.DataFrame(rows, columns=["timestamp", "open_interest", "open_interest_value"])
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
