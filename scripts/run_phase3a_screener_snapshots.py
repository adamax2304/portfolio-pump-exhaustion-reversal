"""CLI: Phase 3A - Pump Exhaustion Screener v2 event-snapshot dataset +
diagnostic/decile/effect-size/stability study. Builds the 7 Phase 3A
deliverables (see project plan "Phase 3"). Reuses ALL existing historical
pump events (discovery-sample Binance+Bybit canonical + the Phase 2D
historical OOS Bybit universe) and data - no new backfill.

Deliberately NOT connected to v1's trading logic anywhere: no PnL, no entry/
risk/exit, no live scanner. Pure descriptive/diagnostic statistics about
whether current pump STATE carries P(REVERSAL) vs P(CONTINUATION) ranking
information.

python scripts/run_phase3a_screener_snapshots.py --out-run-id phase3a_v1
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import typer

from pump_exhaustion_reversal.backtest import walk_forward
from pump_exhaustion_reversal.backtest.simulator import EventRecord, TradeRecord
from pump_exhaustion_reversal.config import AppConfig, BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.feature_store import build_feature_frame
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.features.structure import atr as compute_atr
from pump_exhaustion_reversal.screener_v2 import diagnostics, features, labels, pump_dynamics, ranking, reports
from pump_exhaustion_reversal.screener_v2 import snapshot_builder as sb

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
MAX_LIFECYCLE_HOURS = sb.DEFAULT_MAX_LIFECYCLE_HOURS
STABILITY_SPLIT_DIMS = ("period", "exchange", "regime_trend")
STABILITY_PREFILTER_ABS_CLIFFS_DELTA = 0.10

SOURCES = [
    ("binance", "binance_canonical_v2", None, None),
    ("bybit", "bybit_canonical_v2", None, None),
    ("bybit", "bybit_historical_oos_v1", "2021-01-01", "2026-02-24"),
]


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


def _load_ohlcv_by_symbol(exchange: str, symbols: set[str], timeframe: str, since_ms: int, until_ms: int) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in sorted(symbols):
        df = storage.read_range(exchange, "ohlcv", symbol, timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        if not df.empty:
            out[(exchange, symbol)] = df
    return out


def _load_feature_frame_by_symbol(
    exchange: str, symbols: set[str], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame], timeframe: str,
    since_ms: int, until_ms: int, cfg: AppConfig, btc_ohlcv: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in sorted(symbols):
        ohlcv = ohlcv_by_symbol.get((exchange, symbol))
        if ohlcv is None or ohlcv.empty:
            continue
        oi_df = storage.read_range(exchange, "oi", symbol, timeframe="5m", start_ms=since_ms, end_ms=until_ms)
        funding_df = storage.read_range(exchange, "funding", symbol, start_ms=since_ms, end_ms=until_ms)
        out[(exchange, symbol)] = build_feature_frame(
            ohlcv, btc_ohlcv, timeframe, cfg, oi_df=oi_df if not oi_df.empty else None,
            funding_df=funding_df if not funding_df.empty else None,
        )
    return out


def _build_one_event_snapshots(event: EventRecord, ohlcv: pd.DataFrame, feature_frame: pd.DataFrame, btc_ohlcv: pd.DataFrame, timeframe: str, cfg: AppConfig) -> pd.DataFrame | None:
    window = sb.build_snapshot_window(event, ohlcv, feature_frame, btc_ohlcv, timeframe, cfg, max_lifecycle_hours=MAX_LIFECYCLE_HOURS)
    if window is None:
        return None
    atr_series = compute_atr(window.df_slice, cfg.stops.atr_length).bfill()
    climax_idx_wide = window.wide(window.climax_idx_abs)
    detect_idx_wide = window.wide(window.detect_idx)

    feature_dfs = [
        pump_dynamics.compute_velocity_features(window.df_slice, window.bar_interval_ms),
        pump_dynamics.compute_high_extension_features(window.df_slice, window.bar_interval_ms),
        pump_dynamics.compute_pullback_features(window.df_slice, atr_series),
        pump_dynamics.compute_efficiency_features(window.df_slice, window.cvd_slice, window.bar_interval_ms),
        pump_dynamics.compute_failed_continuation_features(window.df_slice, window.cvd_slice, window.bar_interval_ms),
        features.compute_volume_features(window.df_slice, window.cvd_slice, climax_idx_wide, window.bar_interval_ms),
        features.compute_oi_features(window.cvd_slice, window.bar_interval_ms),
        features.compute_price_oi_regime_features(window.df_slice, window.cvd_slice, window.bar_interval_ms),
        features.compute_funding_features(window.cvd_slice, window.bar_interval_ms),
        features.compute_structure_deterioration_features(window.df_slice, window.post_pump, atr_series),
        features.compute_wick_rejection_features(window.df_slice, atr_series),
        features.compute_volatility_features(window.df_slice, atr_series, climax_idx_wide, window.bar_interval_ms),
        features.compute_pump_age_stage_features(window.df_slice, detect_idx_wide, climax_idx_wide, window.bar_interval_ms),
        features.compute_market_context_features(window.df_slice, btc_ohlcv, window.bar_interval_ms),
    ]
    snapshots = sb.assemble_snapshots(window, feature_dfs)
    if snapshots.empty:
        return None

    label_rows = []
    for abs_idx, high_so_far in zip(snapshots["snapshot_abs_idx"], snapshots["event_high_so_far"]):
        family_labels = labels.label_snapshot_across_families(ohlcv, int(abs_idx), window.bar_interval_ms)
        peak_hazard = labels.compute_peak_hazard_label(ohlcv, int(abs_idx), float(high_so_far), window.bar_interval_ms)
        row = {}
        for family, result in family_labels.items():
            row[f"{family}_outcome"] = result["outcome"]
            row[f"{family}_bars"] = result["bars"]
            row[f"{family}_right_censored"] = result["right_censored"]
        row["peak_hazard"] = peak_hazard["peak_hazard"]
        row["peak_hazard_right_censored"] = peak_hazard["right_censored"]
        label_rows.append(row)
    labels_df = pd.DataFrame(label_rows).reset_index(drop=True)
    return pd.concat([snapshots.reset_index(drop=True), labels_df], axis=1)


@app.command()
def main(
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("phase3a_v1"),
    max_events: int = typer.Option(0, help="0 = no cap; otherwise stop after this many events per source (fast pilot run)"),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[EventRecord] = []
    for exchange, run_id, start, end in SOURCES:
        events = _load_events_with_trades(run_id)
        if max_events:
            events = events[:max_events]
        # event_id is only unique WITHIN one event-study run (per-symbol
        # sequence numbering restarts at 1 for each run) - the discovery
        # sample and historical OOS sample cover overlapping symbols on
        # bybit, so e.g. "bybit:DOGEUSDT:1" exists in BOTH runs as two
        # DIFFERENT real events. Re-tag with the source run_id here, before
        # anything downstream (event weighting, cluster bootstrap, merges)
        # ever groups by event_id - otherwise unrelated events silently get
        # pooled as if they were one, corrupting every event-clustered stat.
        events = [replace(e, event_id=f"{run_id}::{e.event_id}") for e in events]
        typer.echo(f"[{exchange}/{run_id}] {len(events)} events")
        all_events.extend(events)
    if not all_events:
        typer.echo("ERROR: no events loaded from any source.")
        raise typer.Exit(1)
    typer.echo(f"Total events across all sources: {len(all_events)}")

    symbols_by_exchange: dict[str, set[str]] = {}
    for event in all_events:
        symbols_by_exchange.setdefault(event.exchange, set()).add(event.symbol)

    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    feature_frame_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    btc_ohlcv_by_exchange: dict[str, pd.DataFrame] = {}
    for exchange, symbols in symbols_by_exchange.items():
        since_ms = datetime_to_ms("2021-01-01")
        until_ms = datetime_to_ms(cfg.backtest.end_date)
        btc_ohlcv = storage.read_range(exchange, "ohlcv", "BTCUSDT", timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        btc_ohlcv_by_exchange[exchange] = btc_ohlcv
        ohlcv_by_symbol.update(_load_ohlcv_by_symbol(exchange, symbols, timeframe, since_ms, until_ms))
        feature_frame_by_symbol.update(_load_feature_frame_by_symbol(exchange, symbols, ohlcv_by_symbol, timeframe, since_ms, until_ms, cfg, btc_ohlcv))
        typer.echo(f"[{exchange}] loaded OHLCV/features for {len(symbols)} symbols")

    typer.echo("Building event snapshots (this is the long-running step)...")
    all_snapshot_frames = []
    n_events_with_snapshots = 0
    for i, event in enumerate(all_events):
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        feature_frame = feature_frame_by_symbol.get((event.exchange, event.symbol))
        btc_ohlcv = btc_ohlcv_by_exchange.get(event.exchange)
        if ohlcv is None or feature_frame is None or btc_ohlcv is None or btc_ohlcv.empty:
            continue
        try:
            snapshots = _build_one_event_snapshots(event, ohlcv, feature_frame, btc_ohlcv, timeframe, cfg)
        except Exception as exc:  # one bad event must not abort the whole run
            typer.echo(f"  [{event.event_id}] SKIPPED ({exc.__class__.__name__}: {exc})")
            continue
        if snapshots is not None and not snapshots.empty:
            all_snapshot_frames.append(snapshots)
            n_events_with_snapshots += 1
        if (i + 1) % 50 == 0:
            n_rows = sum(len(f) for f in all_snapshot_frames)
            typer.echo(f"  ...{i + 1}/{len(all_events)} events processed, {n_events_with_snapshots} produced snapshots, {n_rows} snapshot rows so far")

    if not all_snapshot_frames:
        typer.echo("ERROR: no snapshots produced from any event.")
        raise typer.Exit(1)
    snapshots_df = pd.concat(all_snapshot_frames, ignore_index=True)
    typer.echo(f"Total: {len(snapshots_df)} snapshots across {n_events_with_snapshots} events")

    # ---- merge context columns needed for stability splits (point 25) ----
    events_df_for_regime = pd.DataFrame([{"event_id": e.event_id, "btc_return_concurrent": e.btc_return_concurrent, "pump_detected_ts": e.pump_detected_ts} for e in all_events])
    events_with_regime = walk_forward.label_market_regime(events_df_for_regime)
    snapshots_df = snapshots_df.merge(events_with_regime[["event_id", "regime_trend"]], on="event_id", how="left")
    ts_min, ts_max = snapshots_df["pump_detected_timestamp"].min(), snapshots_df["pump_detected_timestamp"].max()
    midpoint = ts_min + (ts_max - ts_min) / 2
    snapshots_df["period"] = np.where(snapshots_df["pump_detected_timestamp"] < midpoint, "early", "late")
    snapshots_df = snapshots_df.copy()  # defragment after the piecemeal column assignments above

    snapshots_df.to_parquet(out_dir / "event_snapshot_dataset.parquet", index=False)
    typer.echo(f"Saved event_snapshot_dataset.parquet ({len(snapshots_df)} rows, {len(snapshots_df.columns)} columns)")

    # ---- Deliverable 2: snapshot_data_quality.md ----
    quality = diagnostics.compute_snapshot_data_quality(snapshots_df)
    (out_dir / "snapshot_data_quality.md").write_text(reports.render_data_quality_report(quality), encoding="utf-8")

    # ---- Deliverable 3: label_distribution.md ----
    distribution_df = diagnostics.compute_label_distribution(snapshots_df)
    (out_dir / "label_distribution.md").write_text(reports.render_label_distribution_report(distribution_df), encoding="utf-8")

    # ---- feature discovery: every numeric column not identity/label ----
    pump_dynamics_features = [
        "return_5m", "return_15m", "return_30m", "return_60m", "velocity_5m", "velocity_15m",
        "acceleration_5_to_15m", "acceleration_15_to_30m", "acceleration_30_to_60m",
        "bars_since_last_new_high", "number_of_new_highs_15m", "number_of_new_highs_30m", "number_of_new_highs_60m",
        "last_high_extension_pct", "previous_high_extension_pct", "latest_time_between_highs_bars", "mean_time_between_highs_bars",
        "high_extension_velocity", "high_extension_deceleration",
        "pullback_depth_pct", "pullback_depth_atr", "pullback_duration_bars", "latest_pullback_depth_pct", "latest_pullback_depth_atr", "average_pullback_depth_pct",
        "price_efficiency_volume_relative", "efficiency_volume_slope", "price_efficiency_oi_relative", "efficiency_oi_slope",
        "continuation_attempt_volume_zscore", "continuation_price_extension", "continuation_efficiency", "failed_continuation_strength",
    ]
    non_feature_prefixes = tuple(diagnostics.IDENTITY_COLUMNS) + ("primary_", "fast_", "deep_", "peak_hazard", "period", "regime_trend")
    all_feature_cols = [
        c for c in snapshots_df.columns
        if c not in diagnostics.IDENTITY_COLUMNS and not any(c.startswith(p) for p in non_feature_prefixes)
        and pd.api.types.is_numeric_dtype(snapshots_df[c])
    ]
    typer.echo(f"Discovered {len(all_feature_cols)} candidate numeric feature columns ({len(pump_dynamics_features)} pump_dynamics)")

    outcome_col = "primary_outcome"

    # ---- Deliverable 4: pump_dynamics_feature_study.md ----
    typer.echo("Computing pump_dynamics feature study...")
    pd_effect_sizes = [ranking.effect_size_report(snapshots_df, f, outcome_col) for f in pump_dynamics_features if f in snapshots_df.columns]
    pd_decile_tables = {f: ranking.decile_analysis(snapshots_df, f, outcome_col) for f in pump_dynamics_features if f in snapshots_df.columns}
    (out_dir / "pump_dynamics_feature_study.md").write_text(reports.render_pump_dynamics_feature_study(pd_effect_sizes, pd_decile_tables), encoding="utf-8")

    # ---- Deliverable 5: feature_decile_analysis.md (full feature set) ----
    typer.echo("Computing full feature decile analysis...")
    all_decile_tables = {f: ranking.decile_analysis(snapshots_df, f, outcome_col) for f in all_feature_cols}
    (out_dir / "feature_decile_analysis.md").write_text(reports.render_feature_decile_analysis(all_decile_tables, "Feature Decile Analysis (Full Set)"), encoding="utf-8")

    # ---- effect sizes for the full set (needed for stability pre-filter + top candidates) ----
    typer.echo("Computing full feature effect sizes...")
    all_effect_sizes = {f: ranking.effect_size_report(snapshots_df, f, outcome_col) for f in all_feature_cols}
    for f in pump_dynamics_features:
        if f in snapshots_df.columns and f not in all_effect_sizes:
            all_effect_sizes[f] = ranking.effect_size_report(snapshots_df, f, outcome_col)

    # ---- Deliverable 6: feature_stability.md (pre-filtered to features clearing a minimum effect size) ----
    typer.echo("Computing feature stability for pre-filtered candidates...")
    candidate_features = [f for f, r in all_effect_sizes.items() if r.get("valid") and pd.notna(r.get("cliffs_delta")) and abs(r["cliffs_delta"]) >= STABILITY_PREFILTER_ABS_CLIFFS_DELTA]
    typer.echo(f"{len(candidate_features)} features clear the |Cliff's delta|>={STABILITY_PREFILTER_ABS_CLIFFS_DELTA} pre-filter for stability analysis")
    stability_by_feature = {}
    for f in candidate_features:
        stability_df = ranking.feature_stability_report(snapshots_df, f, outcome_col, STABILITY_SPLIT_DIMS)
        verdict = ranking.is_stable(stability_df)
        stability_by_feature[f] = (stability_df, verdict)
    (out_dir / "feature_stability.md").write_text(reports.render_feature_stability_report(stability_by_feature), encoding="utf-8")

    # ---- Deliverable 7: top_candidate_features.md ----
    top_rows = []
    for f in candidate_features:
        effect = all_effect_sizes[f]
        _, verdict = stability_by_feature[f]
        if verdict["stable"]:
            top_rows.append({**effect, "stable": verdict["stable"], "fraction_same_direction": verdict["fraction_same_direction"]})
    ranked = pd.DataFrame(top_rows)
    if not ranked.empty:
        ranked["abs_cliffs_delta"] = ranked["cliffs_delta"].abs()
        ranked = ranked.sort_values("abs_cliffs_delta", ascending=False).drop(columns="abs_cliffs_delta")
    (out_dir / "top_candidate_features.md").write_text(reports.render_top_candidate_features(ranked), encoding="utf-8")

    typer.echo(f"Done. 7 deliverables under: {out_dir}")


if __name__ == "__main__":
    app()
