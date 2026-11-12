"""CLI: run the full backtest - event study + trade simulation + walk-forward
split + acceptance evaluation (spec sections 22-23, 25).

python scripts/run_backtest.py --config config/backtest.yaml --start 2024-09-01 --end 2026-08-25

Reads whatever OHLCV/OI/funding scripts/ingest.py has already cached under
data/raw/ - it does not fetch anything itself.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Windows consoles often default to a non-UTF8 codepage (e.g. cp1252), which
# raises UnicodeEncodeError on any symbol/error text containing a character
# outside that codepage - replace rather than crash mid-run and lose progress.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer

from pump_exhaustion_reversal.backtest import event_study, metrics, reports, walk_forward
from pump_exhaustion_reversal.backtest.simulator import DEFAULT_MAX_EVENT_HORIZON_BARS, EventRecord, TradeRecord
from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms
from pump_exhaustion_reversal.features.news import load_catalyst_csv

app = typer.Typer(add_completion=False)


def _events_for_metrics(events_df) -> list[EventRecord]:
    if events_df.empty:
        return []
    return [EventRecord(**{**row, "trade": None}) for row in events_df.to_dict("records")]


def _trades_for_metrics(trades_df) -> list[TradeRecord]:
    if trades_df.empty:
        return []
    return [TradeRecord(**row) for row in trades_df.to_dict("records")]


def _load_universe(exchange_list: list[str], symbols: str, timeframe: str, since_ms: int, until_ms: int, liquidity_score: float) -> list[dict]:
    universe_entries = []
    for exchange in exchange_list:
        btc_ohlcv = storage.read_range(exchange, "ohlcv", "BTCUSDT", timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        if btc_ohlcv.empty:
            typer.echo(f"[{exchange}] no BTC {timeframe} OHLCV cached - skipping this exchange (run ingest.py first)")
            continue
        symbol_list = (
            storage.list_symbols(exchange, "ohlcv", timeframe=timeframe)
            if symbols == "all"
            else [s.strip().upper() for s in symbols.split(",")]
        )
        for symbol in symbol_list:
            if symbol == "BTCUSDT":
                continue
            ohlcv = storage.read_range(exchange, "ohlcv", symbol, timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
            if len(ohlcv) < 100:
                continue
            oi_df = storage.read_range(exchange, "oi", symbol, timeframe="5m", start_ms=since_ms, end_ms=until_ms)
            funding_df = storage.read_range(exchange, "funding", symbol, start_ms=since_ms, end_ms=until_ms)
            universe_entries.append(
                {
                    "exchange": exchange, "symbol": symbol, "ohlcv": ohlcv, "btc_ohlcv": btc_ohlcv,
                    "oi_df": oi_df if not oi_df.empty else None,
                    "funding_df": funding_df if not funding_df.empty else None,
                    "liquidity_score": liquidity_score,
                }
            )
    return universe_entries


@app.command()
def main(
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    start: str = typer.Option(None, help="override backtest.start_date"),
    end: str = typer.Option(None, help="override backtest.end_date"),
    exchanges: str = typer.Option("bybit,binance"),
    symbols: str = typer.Option("all", help="'all' = every symbol with cached OHLCV, or comma-separated"),
    catalyst_csv: str = typer.Option(None, help="path to a manual catalyst CSV (spec section 4.7 format)"),
    equity: float = typer.Option(5000.0),
    horizon_bars: int = typer.Option(DEFAULT_MAX_EVENT_HORIZON_BARS),
    liquidity_score: float = typer.Option(60.0, help="fixed liquidity quality score applied to the whole universe (no live spread/depth feed - see limitations)"),
    skip_slippage_stress: bool = typer.Option(False, help="skip the doubled-slippage robustness re-run (faster, but the GO/NO-GO verdict will be incomplete)"),
    run_id: str = typer.Option(None),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    if start:
        cfg = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"start_date": start})})
    if end:
        cfg = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"end_date": end})})

    since_ms = datetime_to_ms(cfg.backtest.start_date)
    until_ms = datetime_to_ms(cfg.backtest.end_date)
    timeframe = cfg.backtest.timeframe_base
    exchange_list = [e.strip() for e in exchanges.split(",") if e.strip()]
    catalyst_df = load_catalyst_csv(catalyst_csv) if catalyst_csv else None

    universe_entries = _load_universe(exchange_list, symbols, timeframe, since_ms, until_ms, liquidity_score)
    typer.echo(f"Universe: {len(universe_entries)} symbols across {exchange_list}, timeframe={timeframe}")
    if not universe_entries:
        typer.echo("No cached data found - run scripts/ingest.py first.")
        raise typer.Exit(1)

    events_df, trades_df = event_study.run_event_study(universe_entries, timeframe, cfg, catalyst_df, equity, horizon_bars)
    typer.echo(f"Event study: {len(events_df)} pump events, {len(trades_df)} triggered trades")

    run_id = run_id or uuid.uuid4().hex[:10]
    paths = event_study.save_event_study(events_df, trades_df, run_id)
    typer.echo(f"Saved: {paths}")

    if events_df.empty:
        typer.echo("No pump events detected in this window/universe - nothing to report.")
        raise typer.Exit(0)

    event_metrics = metrics.compute_event_metrics(_events_for_metrics(events_df))
    all_trades = _trades_for_metrics(trades_df)
    overall_metrics = metrics.compute_trade_metrics(all_trades)

    splits = walk_forward.split_chronological(events_df, cfg.backtest.train_pct, cfg.backtest.validation_pct, cfg.backtest.oos_pct)
    split_trade_metrics = walk_forward.evaluate_split_metrics(events_df, trades_df, splits)
    oos_metrics = split_trade_metrics["oos"]

    slippage_stress_metrics = None
    if not skip_slippage_stress and oos_metrics["n_trades"] > 0:
        typer.echo("Running slippage-stress re-run (doubled default/stop slippage bps)...")
        stressed_cfg = cfg.model_copy(
            update={
                "execution_model": cfg.execution_model.model_copy(
                    update={
                        "default_slippage_bps": cfg.execution_model.default_slippage_bps * 2,
                        "stop_slippage_bps": cfg.execution_model.stop_slippage_bps * 2,
                    }
                )
            }
        )
        oos_symbols = set(splits["oos"]["symbol"])
        oos_universe = [u for u in universe_entries if u["symbol"] in oos_symbols]
        _stress_events_df, stress_trades_df = event_study.run_event_study(
            oos_universe, timeframe, stressed_cfg, catalyst_df, equity, horizon_bars
        )
        slippage_stress_metrics = metrics.compute_trade_metrics(_trades_for_metrics(stress_trades_df))

    acceptance = reports.evaluate_acceptance(oos_metrics, event_metrics, slippage_stress_metrics, cfg.acceptance)

    markdown = reports.render_markdown_report(
        run_id=run_id, event_metrics=event_metrics, overall_trade_metrics=overall_metrics,
        oos_trade_metrics=oos_metrics, by_pump_class=metrics.breakdown_by_pump_class(all_trades),
        by_entry_mode=metrics.breakdown_by_entry_mode(all_trades), acceptance=acceptance,
        limitations=reports.STANDARD_LIMITATIONS,
    )
    report_path = reports.save_report(markdown, run_id)
    typer.echo(f"Report saved: {report_path}")
    typer.echo(f"Verdict: {'GO' if acceptance['go'] else 'NO-GO (research-only)'}")


if __name__ == "__main__":
    app()
