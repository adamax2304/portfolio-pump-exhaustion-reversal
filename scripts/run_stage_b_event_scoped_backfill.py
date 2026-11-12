"""Phase 2D universe redesign - Stage B: event-scoped microstructure (OI +
funding) backfill. Reads Stage A's cheap OHLCV-only pump-candidate list and
fetches OI/funding ONLY for the window around each qualifying event (default
T-48h -> T+24h, T = pump_detected_ts), merged across overlapping events on
the same symbol - not each symbol's full multi-year history. MARKET_CONTEXT
symbols (BTC/ETH/SOL) are excluded here by construction (Stage A already
tags and this script filters them out) - they never get event-scoped OI/
funding because they're never PUMP_REVERSAL_CANDIDATE_UNIVERSE members.

Reuses scripts/ingest.py's own `_ingest_one` (same connector calls, same
storage.merge_write dedup) - just called once per event window instead of
once per symbol's full history, so results land in the exact same parquet
files run_stage_a/run_phase_2d scripts already read.

python scripts/run_stage_b_event_scoped_backfill.py --exchange bybit --stage-a-run-id stage_a_v1
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import typer

from ingest import CONNECTORS, _ingest_one  # sibling module in scripts/ - same directory is on sys.path[0] when run directly
from pump_exhaustion_reversal.data import storage

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
DEFAULT_WINDOW_BEFORE_HOURS = 48.0
DEFAULT_WINDOW_AFTER_HOURS = 24.0


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Standard sorted interval-merge - overlapping or touching (start2 <=
    end1) windows collapse into one, so a symbol with several nearby pump
    candidates gets ONE wider API request instead of several redundant
    overlapping ones."""
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _process_symbol(connector, exchange: str, symbol: str, windows: list[tuple[int, int]]) -> list[str]:
    lines = []
    for start_ms, end_ms in windows:
        for data_type in ("oi", "funding"):
            try:
                result = _ingest_one(connector, exchange, symbol, data_type, start_ms, end_ms, [], 0)
                lines.append(f"[{symbol}] {data_type} [{start_ms}-{end_ms}]: {result}")
            except Exception as exc:  # one window's failure must not abort the whole run
                lines.append(f"[FAIL] [{symbol}] {data_type} [{start_ms}-{end_ms}]: {exc}")
    return lines


@app.command()
def main(
    exchange: str = typer.Option("bybit"),
    stage_a_run_id: str = typer.Option("stage_a_v1"),
    window_before_hours: float = typer.Option(DEFAULT_WINDOW_BEFORE_HOURS),
    window_after_hours: float = typer.Option(DEFAULT_WINDOW_AFTER_HOURS),
    concurrency: int = typer.Option(4),
) -> None:
    stage_a_path = DIAGNOSTICS_ROOT / stage_a_run_id / "stage_a_pump_candidates.parquet"
    if not stage_a_path.exists():
        typer.echo(f"ERROR: {stage_a_path} not found - run scripts/run_stage_a_pump_discovery.py first.")
        raise typer.Exit(1)
    candidates = pd.read_parquet(stage_a_path)
    candidate_universe = candidates[~candidates["is_market_context_symbol"]]
    typer.echo(f"Stage A candidates: {len(candidates)} total, {len(candidate_universe)} in PUMP_REVERSAL_CANDIDATE_UNIVERSE (MARKET_CONTEXT symbols excluded)")

    windows_by_symbol: dict[str, list[tuple[int, int]]] = {}
    for symbol, group in candidate_universe.groupby("symbol"):
        raw_windows = [
            (int(ts - window_before_hours * 3_600_000), int(ts + window_after_hours * 3_600_000)) for ts in group["pump_detected_ts"]
        ]
        windows_by_symbol[symbol] = _merge_windows(raw_windows)

    total_windows = sum(len(w) for w in windows_by_symbol.values())
    typer.echo(f"{len(windows_by_symbol)} symbols, {total_windows} merged event windows (window={window_before_hours}h before / {window_after_hours}h after each pump_detected_ts)")

    connector = CONNECTORS[exchange]
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_symbol, connector, exchange, symbol, windows): symbol for symbol, windows in windows_by_symbol.items()}
        for fut in as_completed(futures):
            for line in fut.result():
                typer.echo(f"  {line}")
            done += 1
            if done % 20 == 0:
                typer.echo(f"...{done}/{len(windows_by_symbol)} symbols done")

    typer.echo(f"Stage B done. {done}/{len(windows_by_symbol)} symbols processed, {total_windows} event windows fetched (OI+funding).")


if __name__ == "__main__":
    app()
