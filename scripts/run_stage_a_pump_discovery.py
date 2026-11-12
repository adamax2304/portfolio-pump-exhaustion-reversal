"""Phase 2D universe redesign - Stage A: cheap OHLCV-only pump discovery
across the full backfilled perp universe, using the UNCHANGED frozen pump
detector (strategy/pump_detector.compute_pump_score_frame). No OI/funding
needed - the hard gate (abnormal return + return z-score + volume z-score)
is pure price/volume, and the score's oi_anomaly/liquidation_spike
components gracefully renormalize away when absent (feature_store.py's
documented missing-data behavior, unchanged here - not a new hack).

This exists specifically so the expensive OI/funding backfill (Stage B) only
ever targets symbols/windows that actually had a qualifying pump - BTC/ETH/
SOL are scanned too (harmless, cheap) but excluded from the candidate output
since spec says they are MARKET_CONTEXT only, never PUMP_REVERSAL_CANDIDATE_
UNIVERSE members (see scripts/run_stage_b_event_scoped_backfill.py).

Output: data/diagnostics/<out-run-id>/stage_a_pump_candidates.parquet - one
row per qualifying pump-detected bar (cooldown-deduped so one multi-bar pump
doesn't produce near-duplicate rows): symbol, pump_detected_idx,
pump_detected_ts, pump_score, data_confidence.

python scripts/run_stage_a_pump_discovery.py --exchange bybit --out-run-id stage_a_v1
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

from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.feature_store import build_feature_frame
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.features.returns import bars_for_window
from pump_exhaustion_reversal.strategy import pump_detector

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
DEFAULT_LIQUIDITY_SCORE = 60.0  # matches scripts/run_backtest.py's own default - no live orderbook feed (Stage 1 limitation, unchanged)

# Excluded from PUMP_REVERSAL_CANDIDATE_UNIVERSE per the user's explicit
# instruction - MARKET_CONTEXT only (BTC regime / BTC-adjusted return /
# volatility / negative control), never a candidate symbol themselves.
MARKET_CONTEXT_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})


def _scan_symbol(symbol: str, ohlcv: pd.DataFrame, btc_ohlcv: pd.DataFrame, timeframe: str, cfg, cooldown_bars: int) -> pd.DataFrame:
    feature_frame = build_feature_frame(ohlcv, btc_ohlcv, timeframe, cfg)  # no oi_df/funding_df - Stage A is OHLCV-only by design
    pump_frame = pump_detector.compute_pump_score_frame(feature_frame, DEFAULT_LIQUIDITY_SCORE, cfg.pump_detector, cfg.pump_score_weights)

    rows = []
    n = len(ohlcv)
    i = 0
    detected = pump_frame["pump_detected"].to_numpy()
    while i < n:
        if not bool(detected[i]):
            i += 1
            continue
        rows.append(
            {
                "symbol": symbol, "pump_detected_idx": i, "pump_detected_ts": int(ohlcv["timestamp"].iloc[i]),
                "pump_score": float(pump_frame["pump_score"].iloc[i]), "data_confidence": pump_frame["data_confidence"].iloc[i],
            }
        )
        i += cooldown_bars  # skip past the cooldown window before scanning for the next distinct pump - Stage B re-derives real event boundaries per candidate with full data
    return pd.DataFrame(rows)


@app.command()
def main(
    exchange: str = typer.Option("bybit"),
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("stage_a_v1"),
    start: str = typer.Option(None, help="YYYY-MM-DD, defaults to config.backtest.start_date"),
    end: str = typer.Option(None, help="YYYY-MM-DD, defaults to config.backtest.end_date"),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    since_ms = datetime_to_ms(start) if start else datetime_to_ms(cfg.backtest.start_date)
    until_ms = datetime_to_ms(end) if end else datetime_to_ms(cfg.backtest.end_date)
    cooldown_bars = max(bars_for_window(24 * 3_600_000, bar_interval_ms), 1)

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(storage.list_symbols(exchange, "ohlcv", timeframe))
    typer.echo(f"[{exchange}] {len(symbols)} symbols with stored {timeframe} OHLCV, scanning {start or cfg.backtest.start_date}..{end or cfg.backtest.end_date}")

    btc_ohlcv = storage.read_range(exchange, "ohlcv", "BTCUSDT", timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
    if btc_ohlcv.empty:
        typer.echo("ERROR: no BTCUSDT OHLCV in range - every symbol's BTC-adjusted return needs this. Backfill BTCUSDT first.")
        raise typer.Exit(1)

    all_rows = []
    for i, symbol in enumerate(symbols):
        ohlcv = storage.read_range(exchange, "ohlcv", symbol, timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        if ohlcv.empty or len(ohlcv) < 300:
            continue
        try:
            candidates = _scan_symbol(symbol, ohlcv, btc_ohlcv, timeframe, cfg, cooldown_bars)
        except Exception as exc:  # a single bad symbol (e.g. a data gap) must not abort the whole scan
            typer.echo(f"  [{symbol}] SKIPPED ({exc.__class__.__name__}: {exc})")
            continue
        if not candidates.empty:
            all_rows.append(candidates)
        if (i + 1) % 25 == 0:
            typer.echo(f"  ...{i + 1}/{len(symbols)} symbols scanned, {sum(len(c) for c in all_rows)} pump candidates so far")

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["symbol", "pump_detected_idx", "pump_detected_ts", "pump_score", "data_confidence"]
    )
    result["is_market_context_symbol"] = result["symbol"].isin(MARKET_CONTEXT_SYMBOLS)
    result.to_parquet(out_dir / "stage_a_pump_candidates.parquet", index=False)

    candidate_universe = result[~result["is_market_context_symbol"]]
    typer.echo(
        f"Done. {len(result)} total pump candidates ({len(candidate_universe)} in PUMP_REVERSAL_CANDIDATE_UNIVERSE, "
        f"{len(result) - len(candidate_universe)} on MARKET_CONTEXT symbols, excluded from candidacy) across "
        f"{result['symbol'].nunique()} symbols. Saved: {out_dir / 'stage_a_pump_candidates.parquet'}"
    )


if __name__ == "__main__":
    app()
