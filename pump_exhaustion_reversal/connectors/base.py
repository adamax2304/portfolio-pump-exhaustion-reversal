"""Shared connector primitives: HTTP pagination and the liquidation-data interface.

Native REST (not ccxt) is used for funding/OI because ccxt does not expose
paginated historical funding/OI history in a way we can control on Binance
or Bybit - see project plan. OHLCV goes through ccxt (or, for Binance, raw
klines to pick up the free taker-buy-volume field - see connectors/binance.py).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
import requests

DEFAULT_HTTP_TIMEOUT = 15.0
DEFAULT_MAX_PAGES = 2000
DEFAULT_SLEEP_BETWEEN_PAGES = 0.2


def http_get_json(
    url: str,
    params: dict,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    max_retries: int = 4,
):
    """GET with exponential backoff on transient failures (429/5xx/network).

    A 4xx client error other than 429 (bad symbol, invalid parameter - e.g.
    Binance's -1130 "startTime is invalid" for an out-of-range OI history
    request) is NOT retried: retrying an unchanged request against the same
    validation error just burns ~15s of backoff per call for a guaranteed
    repeat failure - confirmed against the live API during the Stage 1
    pilot run. Raises RuntimeError either way - callers should let a single
    symbol's failure be caught and logged by the ingest script rather than
    aborting the whole backfill (matches the "skip and record failure"
    behaviour of the proven historical_ohlcv_backfill.py pattern).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(f"GET {url} params={params} failed (non-retryable HTTP {status}): {exc}") from exc
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"GET {url} params={params} failed after {max_retries} attempts: {last_exc}")


def paginate_by_time(
    fetch_page: Callable[[int], list[dict]],
    ts_of: Callable[[dict], int],
    since_ms: int,
    until_ms: int | None,
    step_ms_on_stall: int,
    max_pages: int = DEFAULT_MAX_PAGES,
    sleep_between_pages: float = DEFAULT_SLEEP_BETWEEN_PAGES,
) -> list[dict]:
    """Generic since-loop pager for REST endpoints paginated by a rising timestamp.

    `fetch_page(cursor_ms)` returns the next page of row-dicts (or []).
    Stops on: empty page, a stalled cursor (page's max ts didn't advance -
    force-advance by step_ms_on_stall and keep going, matching how a real
    exchange endpoint can return the same last row forever near the front of
    its retention window), until_ms reached, or max_pages hit. Deliberately
    does NOT stop on a partial (< limit) page - a short/gappy page in the
    middle of history (illiquid symbol, exchange outage) does not mean
    we've reached the end (proven pattern, see project plan).
    """
    out: list[dict] = []
    cursor = since_ms
    seen_max_ts = -1
    for _ in range(max_pages):
        page = fetch_page(cursor)
        if not page:
            break
        out.extend(page)
        page_max_ts = max(ts_of(row) for row in page)
        if page_max_ts <= seen_max_ts:
            cursor = cursor + step_ms_on_stall
        else:
            cursor = page_max_ts + 1
            seen_max_ts = page_max_ts
        if until_ms is not None and cursor > until_ms:
            break
        if sleep_between_pages:
            time.sleep(sleep_between_pages)
    return out


def paginate_by_time_backward(
    fetch_page: Callable[[int | None], list[dict]],
    ts_of: Callable[[dict], int],
    since_ms: int,
    until_ms: int | None,
    step_ms_on_stall: int,
    max_pages: int = DEFAULT_MAX_PAGES,
    sleep_between_pages: float = DEFAULT_SLEEP_BETWEEN_PAGES,
) -> list[dict]:
    """Backward since-loop pager for REST endpoints (e.g. Bybit v5 market data)
    that return the most recent `limit` rows within [start, end] and expect
    callers to page by decreasing `end` toward `start`, rather than advancing
    a `start` cursor forward. `fetch_page(cursor_end_ms)` is called with the
    current `end` cursor (None on the first call means "now"/unbounded).
    Stops on: empty page, a stalled cursor (force-retreat by
    step_ms_on_stall), since_ms reached, or max_pages hit.
    """
    out: list[dict] = []
    cursor: int | None = until_ms
    seen_min_ts: int | None = None
    for _ in range(max_pages):
        page = fetch_page(cursor)
        if not page:
            break
        out.extend(page)
        page_min_ts = min(ts_of(row) for row in page)
        if seen_min_ts is not None and page_min_ts >= seen_min_ts:
            cursor = (cursor if cursor is not None else page_min_ts) - step_ms_on_stall
        else:
            cursor = page_min_ts - 1
            seen_min_ts = page_min_ts
        if cursor < since_ms:
            break
        if sleep_between_pages:
            time.sleep(sleep_between_pages)
    return out


class LiquidationDataProvider(ABC):
    """Interface for historical liquidation data (spec section 4.5).

    mode is one of:
      - "real_liquidation_data": a genuine historical liquidation feed.
        Not available for free on Binance/Bybit as of this build (only
        real-time WS liquidation streams exist; no historical REST endpoint).
      - "paid_provider_placeholder": stub for a future paid API (e.g.
        Coinglass/CryptoQuant).
      - "proxy_mode": OI-drop + price-spike + volume-spike composite proxy -
        see features/liquidations.py:ProxyLiquidationProvider, the only mode
        actually implemented and used in Stage 1.
    """

    mode: str

    @abstractmethod
    def get_liquidations(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Columns: timestamp, side ('long'|'short'), notional_usd_estimate, confidence."""


LIQUIDATION_COLUMNS = ["timestamp", "side", "notional_usd_estimate", "confidence"]


class UnavailableLiquidationProvider(LiquidationDataProvider):
    """Placeholder for real_liquidation_data / paid_provider_placeholder - not wired to any feed."""

    def __init__(self, mode: str = "paid_provider_placeholder"):
        if mode not in ("real_liquidation_data", "paid_provider_placeholder"):
            raise ValueError(f"invalid mode for UnavailableLiquidationProvider: {mode!r}")
        self.mode = mode

    def get_liquidations(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        return pd.DataFrame(columns=LIQUIDATION_COLUMNS)
