"""CLI: Phase 3A.1 - Directional Separation Audit. Extends Phase 3A's
event-snapshot dataset (run_phase3a_screener_snapshots.py) with the
resolution-power-vs-direction-power split the user asked for after
reviewing Phase 3A's "B - WEAK RANKING EDGE" verdict: several of Phase 3A's
top-ranked features (realized_vol_1h, volume_current_over_climax) may only
predict "the market is about to resolve out of NEUTRAL" rather than "the
market favors REVERSAL over CONTINUATION specifically".

Builds the 7 Phase 3A.1 deliverables:
  1. directional_vs_resolution_analysis.md (points 1-3)
  2. high_state_early_warning_analysis.md (points 5-6)
  3. pullback_deterioration_analysis.md (point 7, user's #1 priority)
  4. failed_continuation_efficiency.md (points 8-9, user's #2 priority)
  5. oi_regime_interactions.md (points 10-11)
  6. cross_exchange_exhaustion.md (point 12, user's #3 priority)
  7. phase_3a1_summary.md (points 18/20 - the KEEP/WEAK/REJECT synthesis)

Still no ML (point 19), still no PnL/entry/risk/exit/live-scanner logic
anywhere in this package. Reuses run_phase3a_screener_snapshots.py's
event/OHLCV/feature-frame loading verbatim (imported, not duplicated) - only
the per-event snapshot assembly and the downstream analysis are new.

python scripts/run_phase3a1_directional_audit.py --out-run-id phase3a1_v1
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import typer

import run_phase3a_screener_snapshots as phase3a

from pump_exhaustion_reversal.backtest import walk_forward
from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.features.structure import atr as compute_atr
from pump_exhaustion_reversal.research import cross_exchange
from pump_exhaustion_reversal.screener_v2 import cross_exchange_features as cef
from pump_exhaustion_reversal.screener_v2 import features, labels, pump_dynamics, ranking, reports, validation
from pump_exhaustion_reversal.screener_v2 import snapshot_builder as sb

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
STABILITY_SPLIT_DIMS = ("period", "exchange", "regime_trend", "symbol", "liquidity_bucket", "pump_age_bucket")
CENSOR_THRESHOLDS = (0.02, 0.03, 0.05)
HIGH_STATE_MIN_POSITION = 0.90
OUTCOME_COL = "primary_outcome"

# Fixed BEFORE looking at any Phase 3A.1 result: the 16 Phase 3A top
# candidates (top_candidate_features.md) plus the new priority-1/2/3
# features this phase adds. Never curated after seeing results.
CANDIDATE_FEATURES = [
    "realized_vol_1h", "realized_vol_15m", "volume_current_over_climax", "volume_peak_to_current_pct",
    "current_return_from_start", "atr_expansion_ratio", "upper_wick_pct", "bars_since_last_new_high",
    "time_since_last_new_high_minutes", "pullback_duration_bars", "pullback_depth_atr", "rejection_atr",
    "average_pullback_depth_pct", "lower_high_strength", "number_of_new_highs_60m", "oi_change_60m",
    "latest_over_previous_pullback_ratio", "latest_over_average_prior_pullback_ratio",
    "pullback_depth_slope", "pullback_duration_slope", "recovery_fraction", "recovery_time_slope",
    "efficiency_change", "efficiency_combined_change", "volume_per_high_extension",
    "marginal_volume_efficiency", "volume_per_return",
    "return_divergence", "volume_divergence", "oi_divergence",
]

PULLBACK_FEATURES = ["latest_over_previous_pullback_ratio", "latest_over_average_prior_pullback_ratio", "pullback_depth_slope", "pullback_duration_slope", "recovery_fraction", "recovery_time_slope"]
CONTINUATION_FEATURES = ["efficiency_change", "efficiency_combined_change", "volume_per_high_extension", "marginal_volume_efficiency", "volume_per_return"]
CROSS_EXCHANGE_FEATURES = ["return_divergence", "volume_divergence", "oi_divergence"]

INTERACTION_SPECS: list[tuple[str, list[tuple[str, int | None]]]] = [
    ("OI regime x pullback depth slope", [("oi_regime", None), ("pullback_depth_slope", 2)]),
    ("OI regime x volume peak-to-current", [("oi_regime", None), ("volume_peak_to_current_pct", 2)]),
    ("Deepening pullback x OI regime", [("latest_over_previous_pullback_ratio", 2), ("oi_regime", None)]),
    ("Failed-continuation efficiency change x OI regime", [("efficiency_change", 2), ("oi_regime", None)]),
]

FEATURE_INTERPRETATIONS: dict[str, str] = {
    "realized_vol_1h": "Rising realized vol - Phase 3A.1 flag: raised BOTH P(reversal) and P(continuation) together in Phase 3A (resolution power); check direction-power columns before trusting it as reversal-specific.",
    "realized_vol_15m": "Same volatility-expansion family as realized_vol_1h - shares its resolution-vs-direction ambiguity.",
    "volume_current_over_climax": "Volume vs the event's own climax bar - another 'market becoming active' candidate flagged by the user; needs direction power to confirm it isn't resolution-only.",
    "volume_peak_to_current_pct": "Same volume-participation family as volume_current_over_climax.",
    "current_return_from_start": "Raw cumulative return since pump start - conflates 'how big was this pump' with any exhaustion-specific signal; not strictly monotonic in Phase 3A's decile table.",
    "atr_expansion_ratio": "Current ATR vs the event's own trailing average - volatility-expansion proxy, same resolution-power caution as realized_vol_*.",
    "upper_wick_pct": "Upper-wick rejection at the current bar - a genuine single-bar exhaustion signature IF it carries direction power, not just 'a big bar happened'.",
    "bars_since_last_new_high": "Time since the last new high - suspected POSITIONAL bias (point 4): a snapshot already deep in a pullback trivially scores high here regardless of real exhaustion.",
    "time_since_last_new_high_minutes": "Same metric as bars_since_last_new_high in minutes - same positional-bias caution.",
    "pullback_duration_bars": "Current in-progress pullback duration - same positional-bias family as bars_since_last_new_high.",
    "pullback_depth_atr": "Current in-progress pullback depth in ATR units - positional bias possible.",
    "rejection_atr": "Distance below the running high in ATR units - Phase 3A's decile table showed this monotonically DECREASING with more rejection, consistent with 'already deep into the move' rather than a leading signal.",
    "average_pullback_depth_pct": "Average of all completed pullbacks so far - an event-maturity proxy more than a single-bar trigger.",
    "lower_high_strength": "How far the current high falls short of the pre-break swing high - a genuine structural-deterioration signal, conditional on a structure break having occurred.",
    "number_of_new_highs_60m": "Count of new highs in the last 60m - a pump-intensity proxy, resolution-power risk similar to volume/volatility features.",
    "oi_change_60m": "OI change over 60m - U-shaped in Phase 3A (both strong growth AND strong decline show elevated reversal rates); see oi_regime_interactions.md instead of treating it as linear.",
    "latest_over_previous_pullback_ratio": "Latest completed pullback depth vs the one before it - user's #1 hypothesis: pullbacks getting DEEPER relative to their own recent history.",
    "latest_over_average_prior_pullback_ratio": "Latest pullback vs the average of ALL prior pullbacks - same deepening-pullback hypothesis, smoothed baseline.",
    "pullback_depth_slope": "Change in completed-pullback depth vs the previous completed pullback - directly operationalizes 'pullbacks becoming deeper'.",
    "pullback_duration_slope": "Change in completed-pullback duration vs the previous one - pullbacks taking LONGER to complete.",
    "recovery_fraction": "How much of the current pullback's drawdown has been recovered so far (in-progress) - low/falling values mean the market is struggling to claw back toward the highs.",
    "recovery_time_slope": "Change in recovery time (episode low to next new high) vs the previous completed pullback - directly operationalizes 'recoveries becoming slower'.",
    "efficiency_change": "Change in continuation_efficiency (price extension per unit volume impulse) between the last two continuation attempts - user's #2 hypothesis: participation arrives while price output per unit shrinks.",
    "efficiency_combined_change": "Same as efficiency_change but with volume+OI impulse combined as the input - NaN whenever OI is unavailable for the event.",
    "volume_per_high_extension": "Volume spent per unit of high-extension achieved at the latest new-high attempt - inverse framing of continuation efficiency, gated by actual new-high events.",
    "marginal_volume_efficiency": "Change in volume_per_high_extension between the last two high-attempts - rising values mean each additional new high is getting more expensive to buy.",
    "volume_per_return": "Quote volume spent per unit of 30m price return - a continuous absorption proxy, not gated to new-high bars specifically.",
    "return_divergence": "This exchange's cumulative return minus the linked exchange's, at the same wall-clock moment - user's #3 hypothesis: non-confirmation across venues.",
    "volume_divergence": "This exchange's volume-over-climax minus the linked exchange's - participation non-confirmation across venues.",
    "oi_divergence": "This exchange's OI change (60m) minus the linked exchange's - positioning non-confirmation across venues.",
}


def _build_one_event_snapshots(event, ohlcv: pd.DataFrame, feature_frame: pd.DataFrame, btc_ohlcv: pd.DataFrame, timeframe: str, cfg) -> pd.DataFrame | None:
    window = sb.build_snapshot_window(event, ohlcv, feature_frame, btc_ohlcv, timeframe, cfg, max_lifecycle_hours=sb.DEFAULT_MAX_LIFECYCLE_HOURS)
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
        pump_dynamics.compute_volume_efficiency_at_highs(window.df_slice, window.bar_interval_ms),
        features.compute_volume_features(window.df_slice, window.cvd_slice, climax_idx_wide, window.bar_interval_ms),
        features.compute_oi_features(window.cvd_slice, window.bar_interval_ms),
        features.compute_oi_regime_features(window.cvd_slice, window.bar_interval_ms),
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

    high_anchored_horizon_bars = max(round(labels.REVERSAL_FROM_HIGH_HORIZON_HOURS * 3_600_000 / window.bar_interval_ms), 1)
    label_rows = []
    for abs_idx, high_so_far in zip(snapshots["snapshot_abs_idx"], snapshots["event_high_so_far"]):
        family_labels = labels.label_snapshot_across_families(ohlcv, int(abs_idx), window.bar_interval_ms)
        peak_hazard = labels.compute_peak_hazard_label(ohlcv, int(abs_idx), float(high_so_far), window.bar_interval_ms)
        high_anchored = labels.compute_high_anchored_label(ohlcv, int(abs_idx), float(high_so_far), high_anchored_horizon_bars)
        row = {}
        for family, result in family_labels.items():
            row[f"{family}_outcome"] = result["outcome"]
            row[f"{family}_bars"] = result["bars"]
            row[f"{family}_right_censored"] = result["right_censored"]
        row["peak_hazard"] = peak_hazard["peak_hazard"]
        row["peak_hazard_right_censored"] = peak_hazard["right_censored"]
        row["high_anchored_outcome"] = high_anchored["outcome"]
        row["high_anchored_bars"] = high_anchored["bars"]
        row["high_anchored_right_censored"] = high_anchored["right_censored"]
        label_rows.append(row)
    labels_df = pd.DataFrame(label_rows).reset_index(drop=True)
    return pd.concat([snapshots.reset_index(drop=True), labels_df], axis=1)


@app.command()
def main(
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("phase3a1_v1"),
    max_events: int = typer.Option(0, help="0 = no cap; otherwise stop after this many events per source (fast pilot run)"),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = []
    events_by_source: dict[str, list] = {}
    for exchange, run_id, start, end in phase3a.SOURCES:
        events = phase3a._load_events_with_trades(run_id)
        if max_events:
            events = events[:max_events]
        events = [replace(e, event_id=f"{run_id}::{e.event_id}") for e in events]
        typer.echo(f"[{exchange}/{run_id}] {len(events)} events")
        events_by_source[run_id] = events
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
        ohlcv_by_symbol.update(phase3a._load_ohlcv_by_symbol(exchange, symbols, timeframe, since_ms, until_ms))
        feature_frame_by_symbol.update(phase3a._load_feature_frame_by_symbol(exchange, symbols, ohlcv_by_symbol, timeframe, since_ms, until_ms, cfg, btc_ohlcv))
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
    typer.echo(f"Total: {len(snapshots_df)} snapshots across {n_events_with_snapshots}/{len(all_events)} events ({n_events_with_snapshots / len(all_events):.1%})")

    # ---- context columns for stability splits (point 13: now includes symbol + liquidity_bucket + pump_age_bucket) ----
    events_df_for_regime = pd.DataFrame([{"event_id": e.event_id, "btc_return_concurrent": e.btc_return_concurrent, "pump_detected_ts": e.pump_detected_ts} for e in all_events])
    events_with_regime = walk_forward.label_market_regime(events_df_for_regime)
    snapshots_df = snapshots_df.merge(events_with_regime[["event_id", "regime_trend"]], on="event_id", how="left").copy()
    ts_min, ts_max = snapshots_df["pump_detected_timestamp"].min(), snapshots_df["pump_detected_timestamp"].max()
    midpoint = ts_min + (ts_max - ts_min) / 2
    snapshots_df["period"] = np.where(snapshots_df["pump_detected_timestamp"] < midpoint, "early", "late")

    liq = snapshots_df["quote_volume_at_pump_start"]
    snapshots_df["liquidity_bucket"] = np.select(
        [liq < cfg.universe.min_24h_quote_volume_usd, liq < cfg.universe.preferred_min_24h_quote_volume_usd, liq >= cfg.universe.preferred_min_24h_quote_volume_usd],
        ["LOW", "MID", "HIGH"], default=None,
    )
    try:
        snapshots_df["pump_age_bucket"] = pd.qcut(snapshots_df["pump_age_minutes"], q=3, labels=["EARLY_AGE", "MID_AGE", "LATE_AGE"], duplicates="drop").astype(str)
    except ValueError:
        snapshots_df["pump_age_bucket"] = None

    typer.echo("Linking cross-exchange events (point 12, priority #3)...")
    binance_disc_events = events_by_source.get("binance_canonical_v2", [])
    bybit_disc_events = events_by_source.get("bybit_canonical_v2", [])
    linked_df = cross_exchange.link_cross_exchange_events(binance_disc_events, bybit_disc_events)
    n_both = int((linked_df["binance_present"] & linked_df["bybit_present"]).sum()) if not linked_df.empty else 0
    typer.echo(f"Cross-exchange: {n_both} linked (both-present) underlying events out of {len(linked_df)} total underlying events")
    cross_exchange_cols = cef.compute_cross_exchange_snapshot_features(snapshots_df, linked_df)
    snapshots_df = pd.concat([snapshots_df.reset_index(drop=True), cross_exchange_cols.reset_index(drop=True)], axis=1)

    snapshots_df = snapshots_df.copy()  # defragment after the piecemeal column assignments above
    snapshots_df.to_parquet(out_dir / "event_snapshot_dataset.parquet", index=False)
    typer.echo(f"Saved event_snapshot_dataset.parquet ({len(snapshots_df)} rows, {len(snapshots_df.columns)} columns)")

    available_candidates = [f for f in CANDIDATE_FEATURES if f in snapshots_df.columns and pd.api.types.is_numeric_dtype(snapshots_df[f])]
    typer.echo(f"{len(available_candidates)}/{len(CANDIDATE_FEATURES)} candidate features present as numeric columns")

    # ---- Deliverable 1: directional_vs_resolution_analysis.md (points 1-3) ----
    typer.echo("Computing resolution vs direction power split...")
    resolution_effects = {f: ranking.resolution_effect_size_report(snapshots_df, f, OUTCOME_COL) for f in available_candidates}
    directional_effects = {f: ranking.directional_effect_size_report(snapshots_df, f, OUTCOME_COL) for f in available_candidates}
    decile_tables = {f: ranking.decile_analysis(snapshots_df, f, OUTCOME_COL) for f in available_candidates}
    (out_dir / "directional_vs_resolution_analysis.md").write_text(
        reports.render_directional_vs_resolution_report(resolution_effects, directional_effects, decile_tables), encoding="utf-8"
    )

    # ---- Deliverable 2: high_state_early_warning_analysis.md (points 5-6) ----
    typer.echo("Computing pre-reversal censoring + high-state cohort analysis...")
    censored_effects: dict[str, dict[float, dict]] = {f: {} for f in available_candidates}
    for thr in CENSOR_THRESHOLDS:
        censored_df = validation.filter_pre_reversal_censored(snapshots_df, thr)
        for f in available_candidates:
            censored_effects[f][thr] = ranking.directional_effect_size_report(censored_df, f, OUTCOME_COL)
    high_state_df = validation.filter_high_state_cohort(snapshots_df, HIGH_STATE_MIN_POSITION)
    typer.echo(f"High-state cohort: {len(high_state_df)} snapshots ({len(high_state_df) / len(snapshots_df):.1%} of all)")
    high_state_effects = {f: ranking.directional_effect_size_report(high_state_df, f, OUTCOME_COL) for f in available_candidates}
    (out_dir / "high_state_early_warning_analysis.md").write_text(
        reports.render_high_state_early_warning_report(censored_effects, high_state_effects, CENSOR_THRESHOLDS), encoding="utf-8"
    )

    # ---- Deliverable 3: pullback_deterioration_analysis.md (point 7, priority #1) ----
    typer.echo("Computing pullback deterioration analysis (priority #1)...")
    pullback_features = [f for f in PULLBACK_FEATURES if f in snapshots_df.columns]
    pullback_directional = {f: ranking.directional_effect_size_report(snapshots_df, f, OUTCOME_COL) for f in pullback_features}
    pullback_deciles = {f: ranking.decile_analysis(snapshots_df, f, OUTCOME_COL) for f in pullback_features}
    (out_dir / "pullback_deterioration_analysis.md").write_text(
        reports.render_feature_group_report(
            "Pullback Deterioration Analysis (Priority #1)",
            "User's #1 reprioritized hypothesis: pullbacks becoming deeper AND recoveries becoming slower/weaker relative to the event's OWN recent history - not absolute pullback depth alone.",
            pullback_directional, pullback_deciles,
        ), encoding="utf-8"
    )

    # ---- Deliverable 4: failed_continuation_efficiency.md (points 8-9, priority #2) ----
    typer.echo("Computing failed-continuation efficiency analysis (priority #2)...")
    continuation_features = [f for f in CONTINUATION_FEATURES if f in snapshots_df.columns]
    continuation_directional = {f: ranking.directional_effect_size_report(snapshots_df, f, OUTCOME_COL) for f in continuation_features}
    continuation_deciles = {f: ranking.decile_analysis(snapshots_df, f, OUTCOME_COL) for f in continuation_features}
    (out_dir / "failed_continuation_efficiency.md").write_text(
        reports.render_feature_group_report(
            "Failed Continuation Efficiency Analysis (Priority #2)",
            "User's #2 reprioritized hypothesis: volume/OI participation keeps arriving while the price output it buys shrinks - closer to an economic definition of exhaustion than raw pullback depth.",
            continuation_directional, continuation_deciles,
        ), encoding="utf-8"
    )

    # ---- Deliverable 5: oi_regime_interactions.md (points 10-11) ----
    typer.echo("Computing OI regime + interaction grids...")
    oi_regime_breakdown = ranking.interaction_grid(snapshots_df, [("oi_regime", None)], OUTCOME_COL) if "oi_regime" in snapshots_df.columns else pd.DataFrame()
    interaction_results = {name: ranking.interaction_grid(snapshots_df, dims, OUTCOME_COL) for name, dims in INTERACTION_SPECS}
    (out_dir / "oi_regime_interactions.md").write_text(
        reports.render_oi_regime_interactions_report(oi_regime_breakdown, interaction_results), encoding="utf-8"
    )

    # ---- Deliverable 6: cross_exchange_exhaustion.md (point 12, priority #3) ----
    typer.echo("Computing cross-exchange exhaustion analysis (priority #3)...")
    cross_exchange_coverage = {
        "n_snapshots_with_other_exchange": int(snapshots_df["other_exchange_present"].sum()),
        "n_snapshots_total": len(snapshots_df),
        "n_linked_underlying_events": n_both,
    }
    cross_exchange_directional = {f: ranking.directional_effect_size_report(snapshots_df, f, OUTCOME_COL) for f in CROSS_EXCHANGE_FEATURES if f in snapshots_df.columns}
    new_high_only_grid = ranking.interaction_grid(snapshots_df, [("one_exchange_new_high_only", None)], OUTCOME_COL) if "one_exchange_new_high_only" in snapshots_df.columns else pd.DataFrame()
    leader_lagger_grid = ranking.interaction_grid(snapshots_df, [("leader_lagger_state", None)], OUTCOME_COL) if "leader_lagger_state" in snapshots_df.columns else pd.DataFrame()
    (out_dir / "cross_exchange_exhaustion.md").write_text(
        reports.render_cross_exchange_exhaustion_report(cross_exchange_coverage, cross_exchange_directional, new_high_only_grid, leader_lagger_grid), encoding="utf-8"
    )

    # ---- Deliverable 7: phase_3a1_summary.md (points 18/20 - KEEP/WEAK/REJECT synthesis) ----
    typer.echo("Computing directional stability + lead-time for the summary table...")
    summary_rows = []
    for f in available_candidates:
        directional_stability_df = ranking.directional_feature_stability_report(snapshots_df, f, OUTCOME_COL, STABILITY_SPLIT_DIMS)
        directional_stability = ranking.is_directionally_stable(directional_stability_df)
        high_state_effect = censored_effects.get(f, {}).get(0.05)
        lead_time = ranking.lead_time_analysis(snapshots_df, f, OUTCOME_COL, "primary_bars", bar_interval_ms=bar_interval_ms, top_quantile=0.9)
        verdict = ranking.classify_feature_verdict(directional_effects[f], directional_stability, high_state_effect)
        summary_rows.append(
            {
                "feature": f, "resolution": resolution_effects[f], "direction": directional_effects[f],
                "high_state": high_state_effect, "lead_time": lead_time, "stability": directional_stability, "verdict": verdict,
            }
        )
    (out_dir / "phase_3a1_summary.md").write_text(reports.render_phase_3a1_summary(summary_rows, FEATURE_INTERPRETATIONS), encoding="utf-8")

    n_keep = sum(1 for r in summary_rows if r["verdict"]["verdict"] == "KEEP")
    n_weak = sum(1 for r in summary_rows if r["verdict"]["verdict"] == "WEAK")
    n_reject = sum(1 for r in summary_rows if r["verdict"]["verdict"] == "REJECT")
    typer.echo(f"Verdicts: {n_keep} KEEP, {n_weak} WEAK, {n_reject} REJECT (of {len(summary_rows)} candidates)")
    typer.echo(f"Done. 7 Phase 3A.1 deliverables under: {out_dir}")


if __name__ == "__main__":
    app()
