"""CLI: historical data ingestion (spec section 25).

Examples:
    python scripts/ingest.py --exchange bybit --symbols all --start 2024-09-01 --end 2026-08-25 --types ohlcv,oi,funding
    python scripts/ingest.py --exchange binance --symbols BTCUSDT,ETHUSDT --start 2024-09-01 --types ohlcv

Full-depth 1m OHLCV for the whole universe is infeasible (see README
limitations) - 1m is backfilled only for the last --m1-lookback-days days by
default; 5m/15m/1h/4h/1d go back to --start. "trades" is intentionally not
backfilled here at all: full-history tick data is infeasible too, so CVD's
high-confidence source is event-anchored trades fetched later, at backtest
time, around each detected pump (see features/cvd.py, backtest/simulator.py).
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Windows consoles often default to a non-UTF8 codepage (e.g. cp1252), which
# raises UnicodeEncodeError on any symbol/error text containing a character
# outside that codepage - replace rather than crash mid-run and lose progress.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer

from pump_exhaustion_reversal.config import DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.connectors import binance, bybit, ccxt_fallback
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data import universe as universe_mod
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, utc_now_ms

app = typer.Typer(add_completion=False)

CONNECTORS = {"binance": binance, "bybit": bybit}
DEFAULT_OHLCV_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_M1_LOOKBACK_DAYS = 90


_PRINT_LOCK = threading.Lock()


def _echo(msg: str) -> None:
    with _PRINT_LOCK:
        typer.echo(msg)


def _ingest_one(connector, exchange: str, symbol: str, data_type: str, since_ms: int, until_ms: int, timeframes: list[str], m1_lookback_days: int) -> str:
    if data_type == "ohlcv":
        counts = []
        for tf in timeframes:
            tf_since = since_ms
            if tf == "1m":
                tf_since = max(since_ms, utc_now_ms() - m1_lookback_days * 86_400_000)
            df = connector.fetch_ohlcv_paged(symbol, tf, tf_since, until_ms)
            path = storage.raw_path(exchange, "ohlcv", symbol, timeframe=tf)
            merged = storage.merge_write(path, df, dedup_subset=["timestamp"])
            counts.append(f"{tf}:{len(df)}new/{len(merged)}total")
        return ", ".join(counts)

    if data_type == "funding":
        df = connector.fetch_funding_history(symbol, since_ms, until_ms)
        path = storage.raw_path(exchange, "funding", symbol)
        merged = storage.merge_write(path, df, dedup_subset=["timestamp"])
        return f"{len(df)}new/{len(merged)}total"

    if data_type == "oi":
        df = connector.fetch_oi_history(symbol, interval="5m", since_ms=since_ms, until_ms=until_ms)
        path = storage.raw_path(exchange, "oi", symbol, timeframe="5m")
        merged = storage.merge_write(path, df, dedup_subset=["timestamp"])
        return f"{len(df)}new/{len(merged)}total (best-effort - exchange retention is shallow, see README)"

    if data_type == "trades":
        return "skipped - trades are event-anchored only, fetched at backtest time (see README limitations)"

    raise ValueError(f"unknown data type {data_type!r}")


def _process_symbol(
    connector, exchange: str, symbol: str, requested_types: list[str], since_ms: int, until_ms: int,
    timeframes: list[str], m1_lookback_days: int, label: str,
) -> list[tuple[str, str, str]]:
    """Runs every requested data type for one symbol. Each symbol writes to
    its own distinct parquet files (storage.raw_path is per-symbol), so
    running many symbols concurrently (different threads, disjoint files) is
    safe without extra locking - only console output needs a lock (_echo)."""
    local_failures: list[tuple[str, str, str]] = []
    for data_type in requested_types:
        try:
            result = _ingest_one(connector, exchange, symbol, data_type, since_ms, until_ms, timeframes, m1_lookback_days)
            _echo(f"  [{label}] {symbol} {data_type}: {result}")
        except Exception as exc:  # noqa: BLE001 - one symbol's failure must not abort the whole backfill
            local_failures.append((symbol, data_type, str(exc)))
            _echo(f"  [FAIL] {symbol} {data_type}: {exc}")
    return local_failures


@app.command()
def main(
    exchange: str = typer.Option(..., help="binance | bybit"),
    symbols: str = typer.Option("all", help="'all' (universe-filtered) or comma-separated e.g. BTCUSDT,ETHUSDT"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(None, help="YYYY-MM-DD, defaults to now"),
    types: str = typer.Option("ohlcv,oi,funding", help="comma-separated: ohlcv,oi,funding,trades"),
    timeframes: str = typer.Option(",".join(DEFAULT_OHLCV_TIMEFRAMES), help="comma-separated OHLCV timeframes"),
    m1_lookback_days: int = typer.Option(DEFAULT_M1_LOOKBACK_DAYS, help="1m OHLCV backfill window"),
    config: str = typer.Option(str(DEFAULT_CONFIG_PATH)),
    max_symbols: int = typer.Option(0, help="0 = no cap; otherwise stop after this many symbols (useful for a quick pilot run)"),
    symbols_file: str = typer.Option(None, help="path to a comma-separated symbol list (e.g. from scripts/screen_candidates.py) - overrides --symbols"),
    concurrency: int = typer.Option(1, help="fetch this many symbols in parallel (1 = sequential, the safe default). Each symbol's own pagination still self-throttles (sleep_between_pages) - concurrency multiplies across symbols, not within one. Keep modest (e.g. 4-8) to avoid tripping an exchange's per-IP rate limit."),
) -> None:
    cfg = load_config(config)
    if exchange not in CONNECTORS:
        raise typer.BadParameter(f"exchange must be one of {list(CONNECTORS)}")
    connector = CONNECTORS[exchange]
    requested_types = [t.strip() for t in types.split(",") if t.strip()]
    requested_tfs = [t.strip() for t in timeframes.split(",") if t.strip()]

    since_ms = datetime_to_ms(start)
    until_ms = datetime_to_ms(end) if end else utc_now_ms()

    ex = connector.make_exchange()

    if symbols_file:
        text = Path(symbols_file).read_text(encoding="utf-8").strip()
        symbol_list = [s.strip().upper() for s in text.split(",") if s.strip()]
        if not symbol_list:
            typer.echo(f"{symbols_file} contains no symbols (empty screen result) - nothing to do.")
            raise typer.Exit(0)
    elif symbols == "all":
        snapshot = ccxt_fallback.fetch_universe_snapshot(ex, quote_asset=cfg.universe.quote_asset)
        filtered = universe_mod.apply_universe_filter(snapshot, cfg.universe)
        passed = filtered[filtered["passed_filter"]].sort_values("quote_volume_24h", ascending=False)
        # sorted by volume descending BEFORE any --max-symbols cap, so a capped
        # pilot run gets the most liquid symbols, not an arbitrary alphabetical prefix
        symbol_list = passed["symbol"].drop_duplicates().tolist()
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if max_symbols:
        symbol_list = symbol_list[:max_symbols]

    typer.echo(f"[{exchange}] {len(symbol_list)} symbols, types={requested_types}, range={start}..{end or 'now'}, concurrency={concurrency}")

    failures: list[tuple[str, str, str]] = []
    n = len(symbol_list)

    if concurrency <= 1:
        for i, symbol in enumerate(symbol_list, 1):
            failures.extend(
                _process_symbol(connector, exchange, symbol, requested_types, since_ms, until_ms, requested_tfs, m1_lookback_days, f"{i}/{n}")
            )
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_symbol = {
                pool.submit(
                    _process_symbol, connector, exchange, symbol, requested_types, since_ms, until_ms,
                    requested_tfs, m1_lookback_days, symbol,
                ): symbol
                for symbol in symbol_list
            }
            for future in as_completed(future_to_symbol):
                done += 1
                try:
                    failures.extend(future.result())
                except Exception as exc:  # noqa: BLE001 - a worker-thread crash must not abort the whole backfill
                    failures.append((future_to_symbol[future], "unknown", str(exc)))
                    _echo(f"  [FAIL] {future_to_symbol[future]}: worker crashed: {exc}")
                _echo(f"  progress: {done}/{n} symbols done")

    total_pairs = n * len(requested_types)
    typer.echo(f"Done. {len(failures)}/{total_pairs} (symbol, type) pairs failed.")
    for s, t, e in failures[:20]:
        typer.echo(f"  - {s} {t}: {e}")


if __name__ == "__main__":
    app()
