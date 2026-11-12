"""CLI: regenerate/print a saved backtest run's report (spec section 25).

python scripts/make_report.py --backtest-id latest
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Windows consoles often default to a non-UTF8 codepage (e.g. cp1252), which
# raises UnicodeEncodeError on any symbol/error text containing a character
# outside that codepage - replace rather than crash mid-run and lose progress.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import typer

from pump_exhaustion_reversal.backtest import metrics, reports, walk_forward
from pump_exhaustion_reversal.backtest.simulator import EventRecord, TradeRecord
from pump_exhaustion_reversal.config import DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage

app = typer.Typer(add_completion=False)


def _resolve_run_id(backtest_id: str) -> str:
    if backtest_id != "latest":
        return backtest_id
    candidates = sorted(storage.EVENTS_ROOT.glob("events_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise typer.BadParameter("no saved event studies found under data/events/ - run scripts/run_backtest.py first")
    return candidates[-1].stem.removeprefix("events_")


@app.command()
def main(backtest_id: str = typer.Option("latest"), config: str = typer.Option(str(DEFAULT_CONFIG_PATH))) -> None:
    run_id = _resolve_run_id(backtest_id)
    events_path = storage.EVENTS_ROOT / f"events_{run_id}.parquet"
    trades_path = storage.BACKTESTS_ROOT / f"trades_{run_id}.parquet"
    if not events_path.exists():
        raise typer.BadParameter(f"no such run: {events_path}")

    events_df = pd.read_parquet(events_path)
    trades_df = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    cfg = load_config(config)

    events_for_metrics = (
        [EventRecord(**{**row, "trade": None}) for row in events_df.to_dict("records")] if not events_df.empty else []
    )
    trades_for_metrics = [TradeRecord(**row) for row in trades_df.to_dict("records")] if not trades_df.empty else []

    event_metrics = metrics.compute_event_metrics(events_for_metrics)
    overall_metrics = metrics.compute_trade_metrics(trades_for_metrics)

    if not events_df.empty:
        splits = walk_forward.split_chronological(
            events_df, cfg.backtest.train_pct, cfg.backtest.validation_pct, cfg.backtest.oos_pct
        )
        oos_metrics = walk_forward.evaluate_split_metrics(events_df, trades_df, splits)["oos"]
    else:
        oos_metrics = overall_metrics

    acceptance = reports.evaluate_acceptance(oos_metrics, event_metrics, None, cfg.acceptance)

    markdown = reports.render_markdown_report(
        run_id=run_id, event_metrics=event_metrics, overall_trade_metrics=overall_metrics,
        oos_trade_metrics=oos_metrics, by_pump_class=metrics.breakdown_by_pump_class(trades_for_metrics),
        by_entry_mode=metrics.breakdown_by_entry_mode(trades_for_metrics), acceptance=acceptance,
        limitations=reports.STANDARD_LIMITATIONS,
    )
    path = reports.save_report(markdown, run_id)
    typer.echo(markdown)
    typer.echo(f"\nSaved to {path}")


if __name__ == "__main__":
    app()
