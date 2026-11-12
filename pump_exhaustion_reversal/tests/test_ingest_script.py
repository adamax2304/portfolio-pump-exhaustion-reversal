"""Tests for scripts/ingest.py's concurrent-fetch path. The script lives
outside the pump_exhaustion_reversal package (it's a CLI entrypoint), so it's
loaded here via importlib rather than a normal package import.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"


@pytest.fixture(scope="module")
def ingest_module():
    spec = importlib.util.spec_from_file_location("ingest_script_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeConnector:
    """Simulates network latency so concurrent workers genuinely overlap -
    if _process_symbol had a race condition, running this under a thread
    pool would surface it (e.g. via corrupted shared state or lost results)."""

    def __init__(self, delay_s: float = 0.02, fail_symbols: set[str] | None = None):
        self.delay_s = delay_s
        self.fail_symbols = fail_symbols or set()
        self.calls: list[str] = []

    def fetch_ohlcv_paged(self, symbol, timeframe, since_ms, until_ms=None):
        time.sleep(self.delay_s)
        self.calls.append(symbol)
        if symbol in self.fail_symbols:
            raise RuntimeError(f"simulated failure for {symbol}")
        return pd.DataFrame({"timestamp": [0], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0], "quote_volume": [1.0], "taker_buy_base_volume": [1.0]})

    def fetch_funding_history(self, symbol, since_ms, until_ms=None):
        time.sleep(self.delay_s)
        return pd.DataFrame({"timestamp": [0], "funding_rate": [0.0001]})

    def fetch_oi_history(self, symbol, interval="5m", since_ms=None, until_ms=None):
        time.sleep(self.delay_s)
        return pd.DataFrame({"timestamp": [0], "open_interest": [1.0], "open_interest_value": [1.0]})


def test_process_symbol_reports_failures_without_raising(ingest_module, tmp_path, monkeypatch):
    import pump_exhaustion_reversal.data.storage as storage_mod

    monkeypatch.setattr(storage_mod, "RAW_ROOT", tmp_path)
    connector = _FakeConnector(fail_symbols={"BADUSDT"})

    ok_failures = ingest_module._process_symbol(
        connector, "bybit", "GOODUSDT", ["ohlcv", "funding"], 0, 1000, ["5m"], 90, "1/2"
    )
    assert ok_failures == []

    bad_failures = ingest_module._process_symbol(
        connector, "bybit", "BADUSDT", ["ohlcv", "funding"], 0, 1000, ["5m"], 90, "2/2"
    )
    assert len(bad_failures) == 1
    assert bad_failures[0][0] == "BADUSDT"
    assert bad_failures[0][1] == "ohlcv"


def test_concurrent_symbols_all_complete_and_write_distinct_files(ingest_module, tmp_path, monkeypatch):
    import pump_exhaustion_reversal.data.storage as storage_mod

    monkeypatch.setattr(storage_mod, "RAW_ROOT", tmp_path)
    symbols = [f"SYM{i}USDT" for i in range(12)]
    connector = _FakeConnector(delay_s=0.05)  # larger delay -> thread/scheduling overhead is proportionally smaller noise

    start = time.time()
    all_failures = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(ingest_module._process_symbol, connector, "bybit", sym, ["ohlcv"], 0, 1000, ["5m"], 90, sym): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            all_failures.extend(future.result())
    elapsed = time.time() - start

    assert all_failures == []
    # 12 symbols at 6 workers should take roughly 2 batches worth of delay,
    # not 12x - a real speedup, not just "didn't crash". Threshold is
    # deliberately loose (well above the ~2-batch theoretical minimum) since
    # this runs on a shared, sometimes-busy dev machine - it only needs to
    # rule out "actually sequential", not assert a precise speedup factor.
    sequential_time = connector.delay_s * len(symbols)
    assert elapsed < sequential_time * 0.75

    for sym in symbols:
        path = storage_mod.raw_path("bybit", "ohlcv", sym, timeframe="5m")
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 1


def test_concurrent_symbols_isolate_per_symbol_failures(ingest_module, tmp_path, monkeypatch):
    import pump_exhaustion_reversal.data.storage as storage_mod

    monkeypatch.setattr(storage_mod, "RAW_ROOT", tmp_path)
    symbols = [f"SYM{i}USDT" for i in range(8)]
    fail_set = {"SYM2USDT", "SYM5USDT"}
    connector = _FakeConnector(delay_s=0.01, fail_symbols=fail_set)

    all_failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(ingest_module._process_symbol, connector, "bybit", sym, ["ohlcv"], 0, 1000, ["5m"], 90, sym)
            for sym in symbols
        ]
        for future in futures:
            all_failures.extend(future.result())

    failed_symbols = {f[0] for f in all_failures}
    assert failed_symbols == fail_set
    # the other 6 symbols must have succeeded despite 2 failing in other threads
    for sym in symbols:
        if sym not in fail_set:
            assert storage_mod.raw_path("bybit", "ohlcv", sym, timeframe="5m").exists()
