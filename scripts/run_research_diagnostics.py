"""CLI: runs the full Stage 1.5 research/diagnostic pipeline (see project
plan's "Этап 1.5 - Research / Diagnostic Phase") against already-saved
canonical event studies. This does NOT re-run the backtest itself - it reads
whatever scripts/run_backtest.py already saved under data/events/ and
data/backtests/, per the exact run_id(s) given.

python scripts/run_research_diagnostics.py --binance-run-id binance_canonical_v1 --bybit-run-id bybit_canonical_v1

Writes every deliverable under data/diagnostics/<out-run-id>/ - deliberately
separate from data/reports/ (Stage 1's GO/NO-GO backtest reports), since this
is a diagnostic pass, not a strategy verdict.
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

from pump_exhaustion_reversal.backtest import simulator as sim
from pump_exhaustion_reversal.backtest.simulator import DEFAULT_MAX_EVENT_HORIZON_BARS, EventRecord, TradeRecord
from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.feature_store import build_feature_frame
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.features import returns as returns_feat
from pump_exhaustion_reversal.features import structure as struct_feat
from pump_exhaustion_reversal.research import (
    charts,
    classifier_audit,
    cross_exchange,
    data_quality,
    dataset_builder,
    exhaustion_audit,
    forward_study,
    negative_analysis,
    predictive_value,
    research_summary,
    structure_audit,
)

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"


def _load_events_with_trades(run_id: str) -> list[EventRecord]:
    """Reconstructs EventRecord objects (with .trade properly re-attached by
    joining the separately-saved trades parquet on event_id) from a run_id
    scripts/run_backtest.py already saved. Returns [] if that run_id has no
    saved events file (not an error - the caller may only have one exchange)."""
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


def _compute_broken_hl_distribution(
    events: list[EventRecord], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame],
    btc_ohlcv_by_exchange: dict[str, pd.DataFrame], cfg, timeframe: str, horizon_bars: int,
) -> pd.DataFrame | None:
    """Best-effort re-derivation of features/structure.py's compute_swing_magnitude
    at each event's own structure_break_bar_local_idx.

    Stage 2A: structure_break_bar_local_idx reflects the FINAL reversal
    attempt (not necessarily attempt 1, if a retry happened - see
    EventRecord.attempt_count/climax_local_idx), so this must re-derive the
    post-pump structure from event.climax_local_idx via
    backtest/simulator.py's _build_post_pump_inputs directly - calling
    _build_event_inputs here would only ever reconstruct attempt 1's
    post-pump frame, silently misaligning structure_break_bar_local_idx for
    any event that actually retried. df_slice/cvd_slice are rebuilt with the
    exact same pump_start_idx/slice_end_idx windowing _build_event_inputs
    itself uses, so the wide coordinate system lines up.

    One bad symbol (e.g. a feature-building edge case) is skipped, not
    fatal - this is a supplementary analysis, not the diagnostic's critical
    path."""
    rows = []
    feature_frame_cache: dict[tuple[str, str], pd.DataFrame] = {}
    bar_interval_ms = parse_timeframe_ms(timeframe)
    for event in events:
        if event.structure_break_bar_local_idx is None or event.climax_local_idx is None:
            continue
        key = (event.exchange, event.symbol)
        ohlcv = ohlcv_by_symbol.get(key)
        btc_ohlcv = btc_ohlcv_by_exchange.get(event.exchange)
        if ohlcv is None or btc_ohlcv is None:
            continue
        detect_positions = ohlcv.index[ohlcv["timestamp"] >= event.pump_detected_ts]
        if len(detect_positions) == 0:
            continue
        detect_idx = int(detect_positions[0])
        try:
            if key not in feature_frame_cache:
                feature_frame_cache[key] = build_feature_frame(ohlcv, btc_ohlcv, timeframe, cfg)
            windows_bars = [returns_feat.bars_for_window(parse_timeframe_ms(w), bar_interval_ms) for w in cfg.pump_detector.windows]
            lookback_bars = max(windows_bars) if windows_bars else 20
            pump_start_idx = max(0, detect_idx - lookback_bars)
            slice_end_idx = min(len(ohlcv) - 1, detect_idx + horizon_bars)
            df_slice = ohlcv.iloc[pump_start_idx : slice_end_idx + 1].reset_index(drop=True)
            cvd_slice = feature_frame_cache[key].iloc[pump_start_idx : slice_end_idx + 1].reset_index(drop=True)

            if event.climax_local_idx >= len(df_slice) or event.structure_break_bar_local_idx >= len(df_slice):
                continue
            post_pump = sim._build_post_pump_inputs(df_slice, cvd_slice, event.climax_local_idx, bar_interval_ms, cfg)
            magnitude = struct_feat.compute_swing_magnitude(
                post_pump.structure, post_pump.post_pump_df, post_pump.post_pump_atr, cfg.structure
            )
            local_break_idx = event.structure_break_bar_local_idx - post_pump.structure_start_idx
            if local_break_idx < 0 or local_break_idx >= len(magnitude):
                continue
            val = magnitude["swing_distance_atr"].iloc[local_break_idx]
            rows.append({"event_id": event.event_id, "swing_distance_atr": val})
        except Exception:
            continue
    if not rows:
        return None
    return pd.DataFrame(rows)


@app.command()
def main(
    binance_run_id: str = typer.Option(None, help="run_id previously saved by scripts/run_backtest.py --exchanges binance"),
    bybit_run_id: str = typer.Option(None, help="run_id previously saved by scripts/run_backtest.py --exchanges bybit"),
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("diagnostics_v1"),
    top_n_charts: int = typer.Option(10),
    cross_exchange_window_minutes: int = typer.Option(60),
    horizon_bars: int = typer.Option(DEFAULT_MAX_EVENT_HORIZON_BARS),
    skip_broken_hl_join: bool = typer.Option(False, help="skip the best-effort structure-magnitude re-derivation (faster, but structure_break_audit.md's broken-HL table will be empty)"),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    since_ms = datetime_to_ms(cfg.backtest.start_date)
    until_ms = datetime_to_ms(cfg.backtest.end_date)

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"

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

    # ---- Phase 1: Data Quality ----
    typer.echo("Phase 1: data quality audit...")
    exchange_summaries = {}
    exchange_event_confidence = {}
    for exchange, events in exchange_events.items():
        availability = data_quality.compute_symbol_level_availability(exchange, timeframe=timeframe)
        exchange_summaries[exchange] = data_quality.summarize_availability(availability)
        exchange_event_confidence[exchange] = data_quality.classify_events_data_confidence(events)
    (out_dir / "data_quality_report.md").write_text(
        data_quality.render_data_quality_report(exchange_summaries, exchange_event_confidence), encoding="utf-8"
    )

    # ---- Phase 2: Classifier Audit ----
    typer.echo("Phase 2: pump classifier audit...")
    class_distributions = {exchange: classifier_audit.class_distribution(events) for exchange, events in exchange_events.items()}
    oi_hypothesis = {exchange: classifier_audit.fallback_bias_check(events) for exchange, events in exchange_events.items()}
    (out_dir / "pump_classifier_audit.md").write_text(classifier_audit.render_classifier_audit_report(exchange_events), encoding="utf-8")

    # ---- Phase 3: Cross-Exchange Audit ----
    linked_df = pd.DataFrame()
    if "binance" in exchange_events and "bybit" in exchange_events and exchange_events["binance"] and exchange_events["bybit"]:
        typer.echo("Phase 3: cross-exchange audit...")
        linked_df = cross_exchange.link_cross_exchange_events(
            exchange_events["binance"], exchange_events["bybit"], window_minutes=cross_exchange_window_minutes
        )
        confusion_df = cross_exchange.confusion_matrix(linked_df)
        lead_lag = cross_exchange.lead_lag_summary(linked_df)
        (out_dir / "cross_exchange_audit.md").write_text(
            cross_exchange.render_cross_exchange_report(linked_df, confusion_df, lead_lag), encoding="utf-8"
        )
    else:
        typer.echo("Phase 3: skipped (need events on both binance and bybit)")

    # ---- Load raw OHLCV per symbol (forward_study, charts, structure re-derivation) ----
    typer.echo("Loading raw OHLCV for forward-path analysis...")
    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    btc_ohlcv_by_exchange: dict[str, pd.DataFrame] = {}
    for exchange, events in exchange_events.items():
        ohlcv_by_symbol.update(_load_ohlcv_by_symbol(exchange, events, timeframe, since_ms, until_ms))
        btc = storage.read_range(exchange, "ohlcv", "BTCUSDT", timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        if not btc.empty:
            btc_ohlcv_by_exchange[exchange] = btc
    typer.echo(f"Loaded OHLCV for {len(ohlcv_by_symbol)} (exchange, symbol) pairs")

    # ---- Phase 4: Forward Event Study ----
    typer.echo("Phase 4: forward event study...")
    forward_paths_df = forward_study.compute_forward_paths_for_events(all_events, ohlcv_by_symbol, bar_interval_ms, max_forward_bars=horizon_bars)
    by_class_df = forward_study.breakdown_by_pump_class(forward_paths_df)
    (out_dir / "forward_event_study.md").write_text(forward_study.render_forward_event_study_report(forward_paths_df, by_class_df), encoding="utf-8")
    typer.echo(f"Forward paths computed for {len(forward_paths_df)} events")

    # ---- Phase 5: Structure Break Audit ----
    typer.echo("Phase 5: structure break audit...")
    new_high_timing = structure_audit.new_pump_high_timing_distribution(all_events)
    broken_hl_distribution = None
    if not skip_broken_hl_join:
        broken_hl_magnitudes = _compute_broken_hl_distribution(all_events, ohlcv_by_symbol, btc_ohlcv_by_exchange, cfg, timeframe, horizon_bars)
        if broken_hl_magnitudes is not None:
            broken_hl_distribution = structure_audit.broken_hl_significance_distribution(broken_hl_magnitudes)
    (out_dir / "structure_break_audit.md").write_text(
        structure_audit.render_structure_audit_report(new_high_timing, broken_hl_distribution), encoding="utf-8"
    )

    # ---- Phase 6: Exhaustion Feature Audit ----
    typer.echo("Phase 6: exhaustion feature audit...")
    hit_rates = exhaustion_audit.hit_rate_by_component(all_events)
    (out_dir / "exhaustion_feature_audit.md").write_text(
        exhaustion_audit.render_exhaustion_audit_report(
            hit_rates,
            exhaustion_audit.timing_by_component(all_events),
            exhaustion_audit.co_occurrence_matrix(all_events),
            exhaustion_audit.signal_count_distribution(all_events),
            exhaustion_audit.score_decomposition(all_events, cfg.exhaustion_score_weights),
        ),
        encoding="utf-8",
    )

    # ---- Phase 7: False/True Negative Analysis ----
    typer.echo("Phase 7: false/true negative analysis...")
    fn_df = negative_analysis.find_false_negatives(all_events, forward_paths_df) if not forward_paths_df.empty else pd.DataFrame()
    tn_df = negative_analysis.find_true_negatives(all_events, forward_paths_df) if not forward_paths_df.empty else pd.DataFrame()
    fn_summary = negative_analysis.summarize_false_negatives(fn_df)
    tn_summary = negative_analysis.summarize_true_negatives(tn_df)
    fn_report, tn_report = negative_analysis.render_negative_analysis_reports(fn_df, fn_summary, tn_df, tn_summary)
    (out_dir / "false_negative_analysis.md").write_text(fn_report, encoding="utf-8")
    (out_dir / "true_negative_analysis.md").write_text(tn_report, encoding="utf-8")

    # ---- Phase 8: Feature Predictive Value ----
    typer.echo("Phase 8: feature predictive value...")
    logit_result = {"sufficient_n": False, "n": 0, "note": "no events with forward-path data"}
    rvc = {}
    single_tables: dict[str, pd.DataFrame] = {}
    combo_tables: dict[str, pd.DataFrame] = {}
    if not forward_paths_df.empty:
        feature_frame = predictive_value.build_feature_outcome_frame(all_events, forward_paths_df)
        if not feature_frame.empty:
            labeled_frame = predictive_value.label_reversal_continuation(feature_frame)
            for label, threshold in (("24h_10pct", 0.10), ("24h_20pct", 0.20)):
                single_tables[label] = predictive_value.single_feature_counterfactual(feature_frame, "drawdown_from_eventual_peak_24h", threshold)
                combo_tables[label] = predictive_value.combination_counterfactual(feature_frame, "drawdown_from_eventual_peak_24h", threshold)
            rvc = predictive_value.compare_reversal_vs_continuation(labeled_frame, "exhaustion_score_peak")
            chronological = labeled_frame.merge(
                pd.DataFrame([{"event_id": e.event_id, "pump_detected_ts": e.pump_detected_ts} for e in all_events]),
                on="event_id", how="left",
            ).sort_values("pump_detected_ts").reset_index(drop=True)
            logit_result = predictive_value.logistic_regression_check(chronological)
    (out_dir / "feature_predictive_value.md").write_text(
        predictive_value.render_predictive_value_report(single_tables, combo_tables, rvc, logit_result), encoding="utf-8"
    )

    # ---- Phase 9: dataset + charts + summary ----
    typer.echo("Phase 9: assembling research dataset, charts, and summary...")
    event_frame = dataset_builder.build_event_level_frame(all_events)
    if not forward_paths_df.empty:
        event_frame = dataset_builder.join_forward_paths(event_frame, forward_paths_df)
    if not linked_df.empty:
        event_frame = dataset_builder.join_underlying_event_linking(event_frame, linked_df)
    dataset_path = dataset_builder.save_research_dataset(event_frame, out_dir / "pump_event_research_dataset.parquet")
    typer.echo(f"Saved research dataset: {dataset_path}")

    if not forward_paths_df.empty:
        cohorts = charts.select_cohorts(all_events, forward_paths_df, top_n=top_n_charts)
        chart_paths = charts.render_cohort_charts(cohorts, all_events, ohlcv_by_symbol, charts_dir)
        typer.echo(f"Rendered charts for cohorts: {[(k, len(v)) for k, v in chart_paths.items()]}")

    summary_md = research_summary.render_research_summary(
        exchange_event_counts={exchange: len(events) for exchange, events in exchange_events.items()},
        class_distributions=class_distributions,
        oi_hypothesis=oi_hypothesis,
        new_high_timing=new_high_timing,
        exhaustion_hit_rates=hit_rates,
        false_negative_summary=fn_summary,
        true_negative_summary=tn_summary,
        logit_result=logit_result,
    )
    (out_dir / "research_summary.md").write_text(summary_md, encoding="utf-8")

    typer.echo(f"Done. All outputs under: {out_dir}")


if __name__ == "__main__":
    app()
