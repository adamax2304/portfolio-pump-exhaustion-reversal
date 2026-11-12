"""CLI: Stage 2A's lean 6-table diagnostic set (see project plan "Этап 2A"
Part 4) - runs against event studies already regenerated with the temporal/
event architecture fix (climax-anchored structure, attempt/event split).
Reads whatever scripts/run_backtest.py saved under data/events/ and
data/backtests/ for the given run_id(s); does not re-run the backtest.

python scripts/run_temporal_fix_diagnostics.py --binance-run-id binance_canonical_v2 --bybit-run-id bybit_canonical_v2

Writes under data/diagnostics/<out-run-id>/ - separate from Stage 1.5's
run_research_diagnostics.py output directories.
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

from pump_exhaustion_reversal.backtest.simulator import EventRecord, TradeRecord
from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.features.returns import bars_for_window
from pump_exhaustion_reversal.research import cross_exchange, exhaustion_audit
from pump_exhaustion_reversal.research import first_passage as fp
from pump_exhaustion_reversal.research import lifecycle_diagnostics as ld
from pump_exhaustion_reversal.research import predictive_value, underlying_classification as uc

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"


def _load_events_with_trades(run_id: str) -> list[EventRecord]:
    events_path = storage.EVENTS_ROOT / f"events_{run_id}.parquet"
    if not events_path.exists():
        return []
    events_df = pd.read_parquet(events_path)
    trades_path = storage.BACKTESTS_ROOT / f"trades_{run_id}.parquet"
    trades_by_event_id: dict[str, TradeRecord] = {}
    if trades_path.exists():
        trades_df = pd.read_parquet(trades_path)
        for row in trades_df.to_dict("records"):
            trades_by_event_id[row["event_id"]] = TradeRecord(**row)
    events = []
    for row in events_df.to_dict("records"):
        row = dict(row)
        row["trade"] = trades_by_event_id.get(row["event_id"])
        events.append(EventRecord(**row))
    return events


def _load_ohlcv_by_symbol(
    exchange: str, events: list[EventRecord], timeframe: str, since_ms: int, until_ms: int
) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in sorted({e.symbol for e in events}):
        df = storage.read_range(exchange, "ohlcv", symbol, timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        if not df.empty:
            out[(exchange, symbol)] = df
    return out


@app.command()
def main(
    binance_run_id: str = typer.Option(None, help="run_id previously saved by scripts/run_backtest.py --exchanges binance"),
    bybit_run_id: str = typer.Option(None, help="run_id previously saved by scripts/run_backtest.py --exchanges bybit"),
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("temporal_fix_diagnostics_v1"),
    cross_exchange_window_minutes: int = typer.Option(60),
    label_family: str = typer.Option(ld.DEFAULT_LABEL_FAMILY, help=f"one of {list(fp.LABEL_FAMILIES)}"),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    since_ms = datetime_to_ms(cfg.backtest.start_date)
    until_ms = datetime_to_ms(cfg.backtest.end_date)
    lookback_bars = max(
        [bars_for_window(parse_timeframe_ms(w), bar_interval_ms) for w in cfg.pump_detector.windows], default=20
    )

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    exchange_events: dict[str, list[EventRecord]] = {}
    if binance_run_id:
        exchange_events["binance"] = _load_events_with_trades(binance_run_id)
        typer.echo(f"Loaded {len(exchange_events['binance'])} binance events from run_id={binance_run_id}")
    if bybit_run_id:
        exchange_events["bybit"] = _load_events_with_trades(bybit_run_id)
        typer.echo(f"Loaded {len(exchange_events['bybit'])} bybit events from run_id={bybit_run_id}")
    if not exchange_events or all(len(v) == 0 for v in exchange_events.values()):
        typer.echo("No events loaded for any exchange - nothing to analyze. Check the run_id(s) given.")
        raise typer.Exit(1)

    all_events = [e for events in exchange_events.values() for e in events]
    typer.echo(f"Total events across all exchanges: {len(all_events)}")

    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    for exchange, events in exchange_events.items():
        ohlcv_by_symbol.update(_load_ohlcv_by_symbol(exchange, events, timeframe, since_ms, until_ms))
    typer.echo(f"Loaded OHLCV for {len(ohlcv_by_symbol)} (exchange, symbol) pairs")

    # ---- Table 1: lifecycle funnel ----
    typer.echo("Table 1: lifecycle funnel...")
    funnel = ld.lifecycle_funnel(all_events, armed_threshold=cfg.exhaustion_thresholds.armed)
    (out_dir / "1_lifecycle_funnel.md").write_text(ld.render_lifecycle_funnel(funnel), encoding="utf-8")

    # ---- Table 2: hit rates (reuses Stage 1.5's exhaustion_audit unchanged) ----
    typer.echo("Table 2: exhaustion feature hit rates...")
    hit_rates = exhaustion_audit.hit_rate_by_component(all_events)
    hit_rate_md = ["# Table 2: Exhaustion Feature Hit Rates (post temporal fix)", "", "| Component | N fired | Hit rate |", "|---|---|---|"]
    for _, row in hit_rates.iterrows():
        hit_rate_md.append(f"| {row['component']} | {row['n_fired']} | {row['hit_rate']:.1%} |")
    (out_dir / "2_exhaustion_hit_rates.md").write_text("\n".join(hit_rate_md), encoding="utf-8")

    # ---- Tables 3/4/5: first-passage by anchor / by feature / by combination ----
    typer.echo(f"Tables 3-5: first-passage diagnostics (label family={label_family})...")
    first_passage_df = ld.compute_first_passage_by_anchor(all_events, ohlcv_by_symbol, bar_interval_ms, lookback_bars, label_family)
    by_anchor = ld.summarize_by_anchor(first_passage_df)
    single_feature = ld.first_passage_single_feature_counterfactual(first_passage_df)
    combinations = ld.first_passage_combination_counterfactual(first_passage_df, predictive_value.FEATURE_COMBINATIONS)
    (out_dir / "3_4_5_first_passage.md").write_text(
        ld.render_first_passage_report(by_anchor, single_feature, combinations, label_family), encoding="utf-8"
    )
    first_passage_df.to_parquet(out_dir / "first_passage_by_anchor.parquet", index=False)

    # ---- Table 6: underlying cross-exchange classification ----
    if "binance" in exchange_events and "bybit" in exchange_events and exchange_events["binance"] and exchange_events["bybit"]:
        typer.echo("Table 6: underlying cross-exchange classification...")
        linked_df = cross_exchange.link_cross_exchange_events(
            exchange_events["binance"], exchange_events["bybit"], window_minutes=cross_exchange_window_minutes
        )
        classified_df = uc.classify_underlying_events(linked_df, exchange_events["binance"], exchange_events["bybit"])
        distribution = uc.underlying_class_distribution(classified_df)
        (out_dir / "6_underlying_classification.md").write_text(
            uc.render_underlying_classification_report(classified_df, distribution), encoding="utf-8"
        )
        classified_df.to_parquet(out_dir / "underlying_classified_events.parquet", index=False)
    else:
        typer.echo("Table 6: skipped (need events on both binance and bybit)")

    typer.echo(f"Done. All outputs under: {out_dir}")


if __name__ == "__main__":
    app()
