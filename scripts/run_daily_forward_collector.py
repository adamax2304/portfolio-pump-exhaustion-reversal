"""Phase 2D forward-OOS collector (two-stage, matches
run_stage_a_pump_discovery.py / run_stage_b_event_scoped_backfill.py) - runs
daily via the "PumpExhaustionReversal_ForwardCollector" Windows Scheduled
Task (see scripts/forward_collector.ps1). Pulls cheap OHLCV for the trailing
window, scans ONLY that window with the frozen pump detector, and fetches
event-scoped OI/funding ONLY for symbols/windows with a NEWLY qualifying
pump - never a full-universe OI/funding pull, matching the same discipline
now used for the historical backfill.

New candidates are appended (never overwritten) to
data/diagnostics/forward_oos_v1/stage_a_pump_candidates_forward.parquet -
kept in a SEPARATE file from the historical discovery-sample/OOS-sample
Stage A outputs, so genuine forward data is never accidentally mixed into a
backward-looking run. This stream starts accumulating from the freeze date
in config/oi_volume_reversal_candidate_v1.yaml and must NEVER be used to
retune that candidate - only to eventually evaluate it once enough forward
signals exist (see that file's oos_acceptance_criteria).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import typer

from ingest import CONNECTORS, _ingest_one  # sibling module in scripts/
from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.normalizer import parse_timeframe_ms, utc_now_ms
from run_stage_a_pump_discovery import MARKET_CONTEXT_SYMBOLS, _scan_symbol
from run_stage_b_event_scoped_backfill import DEFAULT_WINDOW_AFTER_HOURS, DEFAULT_WINDOW_BEFORE_HOURS, _merge_windows, _process_symbol

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
FORWARD_RUN_ID = "forward_oos_v1"
LOOKBACK_DAYS = 7  # self-healing window - safe to re-scan already-seen bars, dedup handles it


@app.command()
def main(
    exchange: str = typer.Option("bybit"),
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    window_before_hours: float = typer.Option(DEFAULT_WINDOW_BEFORE_HOURS),
    window_after_hours: float = typer.Option(DEFAULT_WINDOW_AFTER_HOURS),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    now_ms = utc_now_ms()
    since_ms = now_ms - LOOKBACK_DAYS * 86_400_000

    out_dir = DIAGNOSTICS_ROOT / FORWARD_RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    forward_path = out_dir / "stage_a_pump_candidates_forward.parquet"
    already_seen_ts: set[tuple[str, int]] = set()
    if forward_path.exists():
        prior = pd.read_parquet(forward_path, columns=["symbol", "pump_detected_ts"])
        already_seen_ts = set(zip(prior["symbol"], prior["pump_detected_ts"]))

    connector = CONNECTORS[exchange]
    symbols = sorted(storage.list_symbols(exchange, "ohlcv", timeframe))
    typer.echo(f"[{exchange}] forward collector: {len(symbols)} symbols, trailing {LOOKBACK_DAYS}d OHLCV pull + scan")

    # Step 1: cheap OHLCV pull, trailing window, all symbols (including MARKET_CONTEXT ones - needed for btc_ohlcv).
    for symbol in symbols:
        try:
            result = _ingest_one(connector, exchange, symbol, "ohlcv", since_ms, now_ms, [timeframe], 0)
            typer.echo(f"  [ohlcv] {symbol}: {result}")
        except Exception as exc:
            typer.echo(f"  [FAIL ohlcv] {symbol}: {exc}")

    # Step 2: scan only the trailing window for newly qualifying pumps (Stage A logic, reused unchanged).
    until_ms = now_ms
    btc_ohlcv = storage.read_range(exchange, "ohlcv", "BTCUSDT", timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
    if btc_ohlcv.empty:
        typer.echo("ERROR: no BTCUSDT OHLCV for the trailing window - cannot scan.")
        raise typer.Exit(1)
    cooldown_bars = max(1, int(24 * 3_600_000 / bar_interval_ms))

    new_rows = []
    for symbol in symbols:
        ohlcv = storage.read_range(exchange, "ohlcv", symbol, timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        if ohlcv.empty or len(ohlcv) < 50:
            continue
        try:
            candidates = _scan_symbol(symbol, ohlcv, btc_ohlcv, timeframe, cfg, cooldown_bars)
        except Exception as exc:
            typer.echo(f"  [SKIP scan] {symbol}: {exc}")
            continue
        for _, row in candidates.iterrows():
            key = (row["symbol"], int(row["pump_detected_ts"]))
            if key not in already_seen_ts:
                new_rows.append(row.to_dict())

    if not new_rows:
        typer.echo("No new qualifying pump candidates this run.")
        return

    new_df = pd.DataFrame(new_rows)
    new_df["is_market_context_symbol"] = new_df["symbol"].isin(MARKET_CONTEXT_SYMBOLS)
    combined = pd.concat([pd.read_parquet(forward_path), new_df], ignore_index=True) if forward_path.exists() else new_df
    combined.to_parquet(forward_path, index=False)
    typer.echo(f"{len(new_df)} NEW pump candidates this run ({len(new_df[~new_df['is_market_context_symbol']])} candidate-universe, rest MARKET_CONTEXT). Total accumulated: {len(combined)}.")

    # Step 3: event-scoped OI/funding backfill for the NEW candidate-universe events only (Stage B logic, reused unchanged).
    new_candidate_universe = new_df[~new_df["is_market_context_symbol"]]
    if new_candidate_universe.empty:
        typer.echo("No new candidate-universe events - skipping OI/funding fetch.")
        return
    windows_by_symbol: dict[str, list[tuple[int, int]]] = {}
    for symbol, group in new_candidate_universe.groupby("symbol"):
        raw_windows = [
            (int(ts - window_before_hours * 3_600_000), int(ts + window_after_hours * 3_600_000)) for ts in group["pump_detected_ts"]
        ]
        windows_by_symbol[symbol] = _merge_windows(raw_windows)

    for symbol, windows in windows_by_symbol.items():
        for line in _process_symbol(connector, exchange, symbol, windows):
            typer.echo(f"  {line}")

    typer.echo("Forward collector run complete.")


if __name__ == "__main__":
    app()
