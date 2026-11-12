"""Generic ccxt-based helpers shared by both exchange connectors.

Used for OHLCV (Bybit, which has no free taker-buy-volume field so gains
nothing from a native klines call), event-anchored public trades (CVD), and
universe/ticker metadata - the calls ccxt already unifies well across
exchanges (spec section 3: "CCXT можно использовать для унифицированных
OHLCV/ticker calls").
"""

from __future__ import annotations

import re

import ccxt
import pandas as pd

from ..data.normalizer import (
    CANONICAL_OHLCV_COLUMNS,
    normalize_symbol,
    ohlcv_rows_to_df,
    parse_timeframe_ms,
)

DEFAULT_OHLCV_LIMIT = 1000
DEFAULT_MAX_REQUESTS = 1000
DEFAULT_TRADES_LIMIT = 1000
DEFAULT_TRADES_MAX_REQUESTS = 500

STABLECOIN_BASES = {
    "USDC", "DAI", "TUSD", "FDUSD", "USDP", "BUSD", "USDD", "PYUSD",
    "EUR", "EURI", "USTC", "GUSD", "USDE",
}
_LEVERAGED_TOKEN_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def fetch_ohlcv_paged(
    exchange: ccxt.Exchange,
    market_symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = DEFAULT_OHLCV_LIMIT,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> pd.DataFrame:
    """ccxt since-loop OHLCV pager (proven pattern: stop on empty page / stalled
    cursor / until_ms / max_requests - never on a partial page, see connectors/base.py).
    """
    step_ms = parse_timeframe_ms(timeframe)
    all_rows: list[list] = []
    cursor = since_ms
    last_ts: int | None = None
    for _ in range(max_requests):
        batch = exchange.fetch_ohlcv(market_symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        newest_ts = batch[-1][0]
        if last_ts is not None and newest_ts <= last_ts:
            break
        last_ts = newest_ts
        cursor = newest_ts + step_ms
        if until_ms is not None and cursor > until_ms:
            break
    df = ohlcv_rows_to_df(all_rows)
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_trades_window(
    exchange: ccxt.Exchange,
    market_symbol: str,
    since_ms: int,
    until_ms: int,
    limit: int = DEFAULT_TRADES_LIMIT,
    max_requests: int = DEFAULT_TRADES_MAX_REQUESTS,
) -> pd.DataFrame:
    """Public trades in [since_ms, until_ms] for CVD - intended for narrow,
    event-anchored windows (hours around a detected pump), not full-history
    backfill (see project plan: full-universe tick history is infeasible).
    `side` is 'buy'/'sell' (taker side) when the exchange/ccxt provides it,
    else None - callers should treat None as cvd_confidence=low (spec 4.2).
    """
    all_trades: list[dict] = []
    cursor = since_ms
    last_ts: int | None = None
    for _ in range(max_requests):
        batch = exchange.fetch_trades(market_symbol, since=cursor, limit=limit)
        if not batch:
            break
        all_trades.extend(batch)
        newest_ts = batch[-1]["timestamp"]
        if last_ts is not None and newest_ts <= last_ts:
            break
        last_ts = newest_ts
        cursor = newest_ts + 1
        if cursor > until_ms:
            break
    rows = [
        {
            "timestamp": t["timestamp"],
            "price": t["price"],
            "amount": t["amount"],
            "side": t.get("side"),
        }
        for t in all_trades
        if since_ms <= t["timestamp"] <= until_ms
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "price", "amount", "side"])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp", "price", "amount", "side"]).sort_values("timestamp")
    df["cvd_confidence"] = df["side"].apply(lambda s: "high" if s in ("buy", "sell") else "low")
    return df.reset_index(drop=True)


def is_stablecoin_base(base_asset: str) -> bool:
    return base_asset.upper() in STABLECOIN_BASES


_LEVERAGED_TOKEN_NUMERIC_SUFFIX = re.compile(r"\d[LS]$")


def is_leveraged_token(base_asset: str) -> bool:
    """True for tokenized-leverage products like BTCUP, ETHBULL, BTC3L, BTC5S."""
    base = base_asset.upper()
    return any(base.endswith(suf) for suf in _LEVERAGED_TOKEN_SUFFIXES) or bool(
        _LEVERAGED_TOKEN_NUMERIC_SUFFIX.search(base)
    )


def fetch_universe_snapshot(exchange: ccxt.Exchange, quote_asset: str = "USDT") -> pd.DataFrame:
    """One row per active linear-perp market: canonical symbol, 24h quote
    volume, last price, a spread proxy in bps from the ticker's best bid/ask,
    and listing timestamp when ccxt's market metadata provides one (not all
    exchanges do - missing means "unknown", not "brand new").
    """
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    rows = []
    for market_symbol, market in markets.items():
        if not (market.get("swap") or market.get("contract")):
            continue
        if market.get("linear") is False:
            continue
        if market.get("quote") != quote_asset:
            continue
        if not market.get("active", True):
            continue
        ticker = tickers.get(market_symbol)
        if not ticker:
            continue
        bid, ask = ticker.get("bid"), ticker.get("ask")
        spread_bps = (ask - bid) / bid * 10_000 if bid and ask and bid > 0 else None
        rows.append(
            {
                "symbol": normalize_symbol(market_symbol),
                "market_symbol": market_symbol,
                "base": market.get("base"),
                "quote_volume_24h": ticker.get("quoteVolume"),
                "last_price": ticker.get("last"),
                "spread_bps": spread_bps,
                "listing_ts": market.get("created"),
            }
        )
    return pd.DataFrame(rows)
