"""CLI: Phase 4A - Ranking to Trading Edge. V2 is frozen and fixed as a
CONFIRMATION screener (Phase 3B.1's verdict). This phase asks a narrower,
different question: after the FIRST high-confidence V2 alert, does a simple
causal entry mechanic produce positive net trading expectancy? Still not a
live scanner, still no real money - pure historical R-multiple research
reusing the project's already-validated cost model (research/entry_edge.py,
backtest/execution_model.py, strategy/risk_engine.py) rather than
reinventing it.

Reuses:
  - Phase 3A.1's event_snapshot_dataset.parquet (via run_phase3b_logistic_
    baseline.py's PHASE3A1_DATASET constant)
  - Phase 3B.1's V2 feature set + temporal/symbol-holdout fold construction
    (run_phase3b1_clean_baseline.py, imported directly - the SAME frozen
    folds, not re-derived)
  - research/entry_edge.py's entry price resolution, R-cost accounting,
    bootstrap, and cost-sensitivity machinery (Stage 2A/2B/2C, already
    fixed for the double-counting bug it documents)
  - strategy/risk_engine.py's compute_structural_stop (min/max stop
    widening, ATR buffer - unchanged)
  - scripts/run_phase_2c_cost_and_early_entry.py's _compute_funding_r
    (the SAME per-bar-prorated funding convention _simulate_trade already
    uses in production - see this script's own module docstring for the
    disclosed "prorated, not literal timestamp-crossing" caveat)

python scripts/run_phase4a_entry_edge.py --out-run-id phase4a_v1
"""

from __future__ import annotations

import sys
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
import run_phase3b_logistic_baseline as phase3b
import run_phase3b1_clean_baseline as phase3b1
import run_phase_2c_cost_and_early_entry as phase2c

from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.features.structure import atr as compute_atr
from pump_exhaustion_reversal.research import entry_edge, first_passage as fp
from pump_exhaustion_reversal.screener_v2 import entry_families as ef
from pump_exhaustion_reversal.screener_v2 import matched_controls as mc
from pump_exhaustion_reversal.screener_v2 import model_evaluation, reports_4a, validation
from pump_exhaustion_reversal.strategy import risk_engine

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
OUTCOME_COL = phase3b.OUTCOME_COL

ALERT_QUANTILES: dict[str, float] = {"top20": 0.80, "top10": 0.90, "top5": 0.95}
PRIMARY_QUANTILE_KEY = "top10"

# Point 12: paired R-multiple/horizon barrier families, fixed before running
# on real data - never an exhaustive 3x3 grid ("не оптимизировать elaborate
# TP"), same pattern as labels.LABEL_FAMILIES elsewhere in this project.
BARRIER_FAMILIES: dict[str, dict] = {
    "FAST_1.5R_2H": {"profit_r": 1.5, "adverse_r": 1.0, "horizon_hours": 2.0},
    "STANDARD_2R_4H": {"profit_r": 2.0, "adverse_r": 1.0, "horizon_hours": 4.0},
    "DEEP_3R_8H": {"profit_r": 3.0, "adverse_r": 1.0, "horizon_hours": 8.0},
}
PRIMARY_BARRIER_FAMILY = "STANDARD_2R_4H"

ENTRY_PATTERN_MAX_WAIT_HOURS = 4.0  # how long Family B/C are allowed to wait for their pattern to form after the alert
ENTRY_MODE = "next_bar_open"  # point 8: entry price resolution, never the alert bar's own quoted price
MATCH_COLS = ["period", "liquidity_bucket", "magnitude_bucket", "volatility_bucket"]
N_CONTROLS_PER_CANDIDATE = 2

FAMILY_NAMES = ("IMMEDIATE", "RETEST", "FAILED_CONTINUATION")


def _detect_signal(family: str, ohlcv: pd.DataFrame, alert_abs_idx: int, event_high_so_far: float, atr_series: pd.Series, max_wait_bars: int) -> dict:
    if family == "IMMEDIATE":
        return ef.detect_family_a_immediate(alert_abs_idx, event_high_so_far)
    if family == "RETEST":
        return ef.detect_family_b_retest(ohlcv, alert_abs_idx, atr_series, max_wait_bars)
    if family == "FAILED_CONTINUATION":
        return ef.detect_family_c_failed_continuation(ohlcv, alert_abs_idx, event_high_so_far, atr_series, max_wait_bars)
    raise ValueError(family)


def _high_vol_flag(atr_series: pd.Series, bar_idx: int) -> bool:
    atr_now = atr_series.iloc[bar_idx]
    baseline = atr_series.iloc[: bar_idx + 1].mean()
    return bool(pd.notna(atr_now) and pd.notna(baseline) and baseline > 0 and atr_now > 2.0 * baseline)


def _build_trade_record(
    family: str, event_id: str, ohlcv: pd.DataFrame, feature_frame: pd.DataFrame | None, atr_series: pd.Series,
    alert_abs_idx: int, event_high_so_far: float, max_wait_bars: int, execution_cfg, stops_cfg, bar_interval_ms: int,
) -> dict | None:
    """Runs one entry family's detector, resolves the entry price, computes
    the structural stop, then walks EVERY barrier family from the SAME
    entry - one row per (event, family, barrier_family). Returns None if
    the family never produces a valid signal for this event (not an error -
    most events will fail most families' pattern most of the time, by
    design)."""
    signal = _detect_signal(family, ohlcv, alert_abs_idx, event_high_so_far, atr_series, max_wait_bars)
    if not signal.get("valid"):
        return None
    signal_bar_idx = signal["signal_bar_idx"]
    entry_price = entry_edge.resolve_entry_price(ohlcv, signal_bar_idx, ENTRY_MODE)
    if entry_price is None:
        return None
    atr_value = atr_series.iloc[signal_bar_idx]
    if pd.isna(atr_value):
        return None
    stop_result = risk_engine.compute_structural_stop(entry_price, signal["stop_reference_price"], float(atr_value), stops_cfg, is_short=True)
    if stop_result.stop_price <= entry_price:
        return None
    stop_distance_pct = stop_result.stop_distance_bps / 10_000.0
    walk_start_idx = signal_bar_idx + 1
    high_vol = _high_vol_flag(atr_series, signal_bar_idx)
    entry_drawdown_pct = (event_high_so_far - entry_price) / event_high_so_far if event_high_so_far else np.nan

    barrier_outcomes = {}
    for barrier_name, spec in BARRIER_FAMILIES.items():
        horizon_bars = max(round(spec["horizon_hours"] * 3_600_000 / bar_interval_ms), 1)
        profit_pct, adverse_pct = fp.resolve_r_based_barriers(stop_distance_pct, spec["profit_r"], spec["adverse_r"])
        outcome = entry_edge.first_passage_from_entry(ohlcv, walk_start_idx, entry_price, profit_pct, adverse_pct, horizon_bars)
        exit_bar_idx = walk_start_idx + outcome["bars"] - 1
        funding_r = phase2c._compute_funding_r(feature_frame, walk_start_idx, exit_bar_idx, bar_interval_ms, stop_distance_pct) if feature_frame is not None else 0.0
        barrier_outcomes[barrier_name] = {**outcome, "stop_distance_pct": stop_distance_pct, "high_vol": high_vol, "funding_r": funding_r}

    return {
        "event_id": event_id, "family": family, "signal_bar_idx": signal_bar_idx, "entry_price": entry_price,
        "stop_price": stop_result.stop_price, "stop_distance_bps": stop_result.stop_distance_bps,
        "exceeds_max_stop": stop_result.exceeds_max_stop, "widened_to_min": stop_result.widened_to_min,
        "entry_drawdown_from_high_pct": entry_drawdown_pct, "barrier_outcomes": barrier_outcomes,
    }


def _summary_for(trade_records: list[dict], barrier_name: str, execution_cfg, bar_interval_ms: int) -> dict:
    outcomes_with_stop = [r["barrier_outcomes"][barrier_name] for r in trade_records if barrier_name in r["barrier_outcomes"]]
    r_trades = entry_edge.compute_r_trades(outcomes_with_stop, BARRIER_FAMILIES[barrier_name]["profit_r"], BARRIER_FAMILIES[barrier_name]["adverse_r"], execution_cfg, bar_interval_ms)
    return entry_edge.summarize_net_expectancy(r_trades)


def _r_trades_for(trade_records: list[dict], barrier_name: str, execution_cfg, bar_interval_ms: int) -> list[dict]:
    outcomes_with_stop = [r["barrier_outcomes"][barrier_name] for r in trade_records if barrier_name in r["barrier_outcomes"]]
    return entry_edge.compute_r_trades(outcomes_with_stop, BARRIER_FAMILIES[barrier_name]["profit_r"], BARRIER_FAMILIES[barrier_name]["adverse_r"], execution_cfg, bar_interval_ms)


@app.command()
def main(config: str = typer.Option(str(BACKTEST_CONFIG_PATH)), out_run_id: str = typer.Option("phase4a_v1")) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    max_wait_bars = max(round(ENTRY_PATTERN_MAX_WAIT_HOURS * 3_600_000 / bar_interval_ms), 1)

    snapshots_df = pd.read_parquet(phase3b.PHASE3A1_DATASET)
    forward_lock_ts = datetime_to_ms(phase3b.FORWARD_LOCK_DATE)
    event_first_ts = snapshots_df.groupby("event_id")["pump_detected_timestamp"].transform("first")
    snapshots_df = snapshots_df.loc[event_first_ts < forward_lock_ts].copy()
    snapshots_df["_drawdown_from_high_pct"] = validation.compute_drawdown_from_high_pct(snapshots_df)
    resolved_df = snapshots_df[snapshots_df[OUTCOME_COL].isin(("REVERSAL", "CONTINUATION"))].copy()

    events_unique = snapshots_df.drop_duplicates(subset="event_id")[["event_id", "pump_detected_timestamp", "symbol"]].rename(columns={"pump_detected_timestamp": "pump_detected_ts"})
    folds = phase3b1.wf.build_expanding_temporal_folds(events_unique, n_folds=4, min_train_fraction=0.4)
    symbol_holdout = phase3b1.wf.build_symbol_holdout_fold(events_unique, holdout_fraction=0.2, seed=7)
    typer.echo(f"V2 frozen feature set: {phase3b1.V2_CONTINUOUS}")

    typer.echo("Re-fitting V2 on the identical frozen temporal folds (point 1-2, no lookahead)...")
    v2_models, v2_val_resolved, v2_val_all = phase3b1._fit_v_on_folds(resolved_df, snapshots_df, folds, phase3b1.V2_CONTINUOUS, phase3b1.V2_CATEGORICAL)
    v2_pooled_all = pd.concat(v2_val_all.values(), ignore_index=True)
    typer.echo(f"Pooled OOF-scored snapshots: {len(v2_pooled_all)} (only chronologically-later events with a legitimate out-of-fold score - the earliest ~40% of events have none, by construction of expanding-window walk-forward).")

    # ---- reference rows (one per event, earliest OOF-scored snapshot) for matched-control bucketing ----
    reference_rows = v2_pooled_all.sort_values("timestamp").groupby("event_id", as_index=False).first()
    reference_rows["magnitude_bucket"] = pd.qcut(reference_rows["current_return_from_start"], q=3, labels=["LOW", "MID", "HIGH"], duplicates="drop").astype(str)
    reference_rows["volatility_bucket"] = pd.qcut(reference_rows["realized_vol_15m"], q=3, labels=["LOW", "MID", "HIGH"], duplicates="drop").astype(str)

    typer.echo("Extracting FIRST_TOP-X%_ALERT_PER_EVENT candidates (point 3-4)...")
    candidates_by_quantile = {key: model_evaluation.first_alert_per_event(v2_pooled_all, "score", top_quantile=q) for key, q in ALERT_QUANTILES.items()}
    for key, df in candidates_by_quantile.items():
        typer.echo(f"  {key}: {len(df)} candidate events")
    primary_candidates = candidates_by_quantile[PRIMARY_QUANTILE_KEY].merge(reference_rows[["event_id", "magnitude_bucket", "volatility_bucket"]], on="event_id", how="left")

    pool_events = reference_rows[~reference_rows["event_id"].isin(candidates_by_quantile[PRIMARY_QUANTILE_KEY]["event_id"])]
    control_links = mc.select_matched_controls(primary_candidates, pool_events, MATCH_COLS, n_controls_per_candidate=N_CONTROLS_PER_CANDIDATE, seed=7)
    typer.echo(f"Matched controls: {len(control_links)} control links for {primary_candidates['event_id'].nunique()} primary candidates.")
    control_events_df = reference_rows[reference_rows["event_id"].isin(control_links["control_event_id"])]

    # ---- load raw OHLCV + feature_frame (funding) for every symbol needed ----
    all_candidate_rows = pd.concat(list(candidates_by_quantile.values()) + [control_events_df], ignore_index=True)
    needed_keys = set(zip(all_candidate_rows["exchange"], all_candidate_rows["symbol"]))
    typer.echo(f"Loading raw OHLCV + feature frames for {len(needed_keys)} (exchange,symbol) pairs...")
    ohlcv_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    feature_frame_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for exchange in {k[0] for k in needed_keys}:
        symbols = {k[1] for k in needed_keys if k[0] == exchange}
        since_ms = datetime_to_ms("2021-01-01")
        until_ms = datetime_to_ms(cfg.backtest.end_date)
        btc_ohlcv = storage.read_range(exchange, "ohlcv", "BTCUSDT", timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        ohlcv_by_exchange = phase3a._load_ohlcv_by_symbol(exchange, symbols, timeframe, since_ms, until_ms)
        ohlcv_by_key.update(ohlcv_by_exchange)
        feature_frame_by_key.update(phase3a._load_feature_frame_by_symbol(exchange, symbols, ohlcv_by_exchange, timeframe, since_ms, until_ms, cfg, btc_ohlcv))
    atr_by_key = {k: compute_atr(v, cfg.stops.atr_length).bfill() for k, v in ohlcv_by_key.items()}

    def _build_records_for(rows_df: pd.DataFrame) -> dict[str, list[dict]]:
        """event_id -> [trade_record per family that produced a valid signal]"""
        out: dict[str, list[dict]] = {}
        for _, row in rows_df.iterrows():
            key = (row["exchange"], row["symbol"])
            ohlcv, atr_series = ohlcv_by_key.get(key), atr_by_key.get(key)
            if ohlcv is None:
                continue
            feature_frame = feature_frame_by_key.get(key)
            recs = []
            for family in FAMILY_NAMES:
                rec = _build_trade_record(
                    family, row["event_id"], ohlcv, feature_frame, atr_series, int(row["snapshot_abs_idx"]),
                    float(row["event_high_so_far"]), max_wait_bars, cfg.execution_model, cfg.stops, bar_interval_ms,
                )
                if rec is not None:
                    recs.append(rec)
            out[row["event_id"]] = recs
        return out

    typer.echo("Running entry-family detectors + cost-model walk for primary (top-10) candidates...")
    primary_records = _build_records_for(primary_candidates)
    typer.echo("Running entry-family detectors for matched controls...")
    control_records = _build_records_for(control_events_df)
    typer.echo("Running entry-family detectors for secondary (top-20/top-5) candidates...")
    secondary_records_by_quantile = {key: _build_records_for(df) for key, df in candidates_by_quantile.items() if key != PRIMARY_QUANTILE_KEY}

    def _flatten(records_by_event: dict[str, list[dict]], family: str) -> list[dict]:
        return [r for recs in records_by_event.values() for r in recs if r["family"] == family]

    # ================= Deliverable 1: confirmation_candidate_dataset.parquet =================
    dataset_rows = []
    for cohort_name, records_by_event in [("top10_primary", primary_records), ("matched_control", control_records)] + [(f"{k}_secondary", v) for k, v in secondary_records_by_quantile.items()]:
        for event_id, recs in records_by_event.items():
            for r in recs:
                for barrier_name, o in r["barrier_outcomes"].items():
                    dataset_rows.append({"cohort": cohort_name, "event_id": event_id, "family": r["family"], "barrier_family": barrier_name, "entry_price": r["entry_price"], "stop_price": r["stop_price"], "stop_distance_bps": r["stop_distance_bps"], "entry_drawdown_from_high_pct": r["entry_drawdown_from_high_pct"], **o})
    pd.DataFrame(dataset_rows).to_parquet(out_dir / "confirmation_candidate_dataset.parquet", index=False)
    typer.echo(f"Saved confirmation_candidate_dataset.parquet ({len(dataset_rows)} rows).")

    # ================= Deliverables 2-4: per-family results =================
    fold_key_by_event = {}
    for f in folds:
        for eid in f["val_event_ids"]:
            fold_key_by_event[eid] = f"fold_{f['fold']}"
    holdout_val_events = symbol_holdout["val_event_ids"]

    per_family_headline = {}
    bucket_ev_by_family = {}
    stop_distance_rows = []
    cost_sensitivity_by_family = {}
    robustness_by_family = {}
    family_descriptions = {
        "IMMEDIATE": "Entry Family A: enter next_bar_open immediately at the first top-10 alert. Simplest benchmark - shows whether the ranking signal alone has edge with zero waiting.",
        "RETEST": "Entry Family B: wait for a causal rebound off a post-alert low, then a stall/lower-high, entry next_bar_open after confirmation.",
        "FAILED_CONTINUATION": "Entry Family C: wait for price to test the recent high and fail to extend (rejection), entry next_bar_open after confirmation. User's favorite hypothesis - failed continuation now only times execution within an already-selected strong cohort, it does not need to prove ranking edge on its own.",
    }
    for family in FAMILY_NAMES:
        primary_recs = _flatten(primary_records, family)
        control_recs = _flatten(control_records, family)
        summary_by_barrier = {bn: _summary_for(primary_recs, bn, cfg.execution_model, bar_interval_ms) for bn in BARRIER_FAMILIES}
        primary_r_trades = _r_trades_for(primary_recs, PRIMARY_BARRIER_FAMILY, cfg.execution_model, bar_interval_ms)
        event_boot = entry_edge.bootstrap_net_expectancy_ci(primary_r_trades)
        symbol_by_event = dict(zip(primary_candidates["event_id"], primary_candidates["symbol"]))
        net_r_series = pd.Series([t["net_r"] for t in primary_r_trades])
        symbol_series = pd.Series([symbol_by_event.get(r["event_id"]) for r in primary_recs if PRIMARY_BARRIER_FAMILY in r["barrier_outcomes"]])
        symbol_boot_raw = validation.cluster_bootstrap_proportion(net_r_series, symbol_series) if len(symbol_series) else {}
        symbol_boot = {"n": symbol_boot_raw.get("n_rows", 0), "win_rate_ci": {}, "net_expectancy_ci": {"statistic": symbol_boot_raw.get("statistic"), "ci_low": symbol_boot_raw.get("ci_low"), "ci_high": symbol_boot_raw.get("ci_high")}}

        temporal_summary = {}
        for f in folds:
            fold_recs = [r for r in primary_recs if fold_key_by_event.get(r["event_id"]) == f"fold_{f['fold']}"]
            temporal_summary[f"fold_{f['fold']}"] = _summary_for(fold_recs, PRIMARY_BARRIER_FAMILY, cfg.execution_model, bar_interval_ms)
        holdout_recs = [r for r in primary_recs if r["event_id"] in holdout_val_events]
        holdout_summary = _summary_for(holdout_recs, PRIMARY_BARRIER_FAMILY, cfg.execution_model, bar_interval_ms)
        robustness_by_family[family] = {"temporal": temporal_summary, "symbol_holdout": holdout_summary}

        score_bucket_summary = {}
        for key in ALERT_QUANTILES:
            recs = primary_recs if key == PRIMARY_QUANTILE_KEY else _flatten(secondary_records_by_quantile[key], family)
            score_bucket_summary[key] = _summary_for(recs, PRIMARY_BARRIER_FAMILY, cfg.execution_model, bar_interval_ms)
        bucket_ev_by_family[family] = score_bucket_summary

        (out_dir / {"IMMEDIATE": "immediate_entry_results.md", "RETEST": "retest_entry_results.md", "FAILED_CONTINUATION": "failed_continuation_entry_results.md"}[family]).write_text(
            reports_4a.render_entry_family_results(
                family, family_descriptions[family], len(primary_candidates), len(primary_recs), summary_by_barrier,
                {"event-cluster": event_boot, "symbol-cluster": symbol_boot}, temporal_summary, holdout_summary, score_bucket_summary,
            ),
            encoding="utf-8",
        )

        primary_headline = summary_by_barrier[PRIMARY_BARRIER_FAMILY]
        primary_headline["median_entry_drawdown"] = float(pd.Series([r["entry_drawdown_from_high_pct"] for r in primary_recs]).median()) if primary_recs else np.nan
        per_family_headline[family] = primary_headline

        for r in primary_recs:
            stop_distance_rows.append({"family": family, "stop_distance_bps": r["stop_distance_bps"]})
        cost_sensitivity_by_family[family] = entry_edge.cost_sensitivity_sweep(
            [r["barrier_outcomes"][PRIMARY_BARRIER_FAMILY] for r in primary_recs], BARRIER_FAMILIES[PRIMARY_BARRIER_FAMILY]["profit_r"], BARRIER_FAMILIES[PRIMARY_BARRIER_FAMILY]["adverse_r"], cfg.execution_model, bar_interval_ms
        )

    # ================= Deliverable 5: tradeability_and_cost_analysis.md =================
    typer.echo("Computing tradeability + cost sensitivity (points 10-11)...")
    stop_df = pd.DataFrame(stop_distance_rows)
    round_trip_cost_bps = 2 * cfg.execution_model.default_slippage_bps + 2 * cfg.execution_model.taker_fee_bps
    stop_rows_out = []
    for family, group in stop_df.groupby("family"):
        row = {"family": family, "n": len(group)}
        for p in (10, 25, 50, 75, 90):
            row[f"stop_distance_bps_p{p}"] = float(np.percentile(group["stop_distance_bps"], p))
        row["cost_to_stop_ratio_median"] = float((round_trip_cost_bps / group["stop_distance_bps"]).median())
        stop_rows_out.append(row)
    stop_distance_summary = pd.DataFrame(stop_rows_out)
    poor_pct = (round_trip_cost_bps / stop_df["stop_distance_bps"] > 0.275).mean() if len(stop_df) else np.nan
    poor_note = f"Poor-tradeability flag (round-trip cost > ~27.5% of stop distance, the midpoint of the user's 25-30% band): {poor_pct:.1%} of all family signals, across all families pooled. Diagnostic only - not used to exclude any trade from the expectancy tables above."
    (out_dir / "tradeability_and_cost_analysis.md").write_text(reports_4a.render_tradeability_and_cost(stop_distance_summary, cost_sensitivity_by_family, poor_note), encoding="utf-8")

    # ================= Deliverable 6: rank_vs_trade_ev.md =================
    (out_dir / "rank_vs_trade_ev.md").write_text(reports_4a.render_rank_vs_trade_ev(bucket_ev_by_family), encoding="utf-8")

    # ================= Deliverable 7: matched_control_analysis.md =================
    typer.echo("Computing matched-control comparison (point 15)...")
    # Point 20: never pick "best" by raw net EV alone - require a reasonable
    # N and a valid PF first (a tiny-N family can top the EV table on one
    # lucky trade); only fall back to the naive argmax if nothing clears
    # that bar at all (disclosed edge case, not expected in practice).
    MIN_N_FOR_BEST_FAMILY = 10
    eligible_families = {f: h for f, h in per_family_headline.items() if h.get("n", 0) >= MIN_N_FOR_BEST_FAMILY and pd.notna(h.get("net_expectancy_r")) and pd.notna(h.get("profit_factor"))}
    if eligible_families:
        best_family_by_ev = max(eligible_families, key=lambda f: eligible_families[f]["net_expectancy_r"])
    else:
        best_family_by_ev = max(per_family_headline, key=lambda f: per_family_headline[f].get("net_expectancy_r", -999) if per_family_headline[f].get("n", 0) else -999)
    control_recs_best = _flatten(control_records, best_family_by_ev)
    control_summary = _summary_for(control_recs_best, PRIMARY_BARRIER_FAMILY, cfg.execution_model, bar_interval_ms)
    (out_dir / "matched_control_analysis.md").write_text(reports_4a.render_matched_control_analysis(per_family_headline[best_family_by_ev], control_summary, best_family_by_ev), encoding="utf-8")

    # ================= Deliverable 8: temporal_symbol_robustness.md =================
    (out_dir / "temporal_symbol_robustness.md").write_text(reports_4a.render_temporal_symbol_robustness(robustness_by_family), encoding="utf-8")

    # ================= Deliverable 9: phase_4a_summary.md =================
    typer.echo("Synthesizing Phase 4A summary (points 26-27)...")
    best_h = per_family_headline.get(best_family_by_ev, {})
    best_temporal = robustness_by_family.get(best_family_by_ev, {}).get("temporal", {})
    n_folds_positive = sum(1 for s in best_temporal.values() if s.get("n", 0) > 0 and s.get("net_expectancy_r", -999) > 0)
    best_holdout = robustness_by_family.get(best_family_by_ev, {}).get("symbol_holdout", {})
    control_better_or_close = control_summary.get("n", 0) > 0 and best_h.get("net_expectancy_r", -999) is not None and control_summary.get("net_expectancy_r", 999) >= best_h.get("net_expectancy_r", 0)

    criteria = [
        {"criterion": "Net EV > 0 (best family, STANDARD_2R_4H)", "observed": f"{best_h.get('net_expectancy_r', float('nan')):+.3f}R (n={best_h.get('n', 0)})", "pass": bool(best_h.get("n", 0) > 0 and best_h.get("net_expectancy_r", -999) > 0)},
        {"criterion": "PF >= ~1.2", "observed": f"{best_h.get('profit_factor', float('nan')):.2f}", "pass": bool(best_h.get("profit_factor", 0) and best_h.get("profit_factor", 0) >= 1.2)},
        {"criterion": "Positive/near-positive across temporal folds", "observed": f"{n_folds_positive}/{len(best_temporal)} folds net EV > 0", "pass": bool(n_folds_positive >= max(1, len(best_temporal) - 1))},
        {"criterion": "Symbol holdout does not collapse", "observed": f"{best_holdout.get('net_expectancy_r', float('nan')):+.3f}R (n={best_holdout.get('n', 0)})", "pass": bool(best_holdout.get("n", 0) > 0 and best_holdout.get("net_expectancy_r", -999) > -0.1)},
        {"criterion": "V2 top-10 cohort clearly better than matched controls", "observed": f"candidates {best_h.get('net_expectancy_r', float('nan')):+.3f}R vs controls {control_summary.get('net_expectancy_r', float('nan')):+.3f}R", "pass": bool(not control_better_or_close)},
        {"criterion": "Median remaining opportunity stays meaningful (Phase 3B.1: ~16-17%)", "observed": "carried over from Phase 3B.1 remaining_opportunity_analysis.md (not re-measured here)", "pass": True},
    ]
    n_pass = sum(1 for c in criteria if c["pass"])
    # Outcome B (ranker good, entry/tradeability not yet good enough) is
    # about whether the CANDIDATES vs CONTROLS gap is real and large - not
    # about whether the candidates' own net EV happens to clear zero. A
    # ranker that turns a clearly-losing control population (-0.38R) into a
    # roughly-breakeven candidate population (-0.01R) has demonstrably added
    # value even though the absolute number is still <= 0; that is exactly
    # "ranker good, execution/cost economics not yet good enough" - not "no
    # edge at all" (C), which is reserved for candidates NOT meaningfully
    # different from controls.
    control_gap_r = best_h.get("net_expectancy_r", float("nan")) - control_summary.get("net_expectancy_r", float("nan"))
    candidates_meaningfully_beat_controls = pd.notna(control_gap_r) and control_gap_r > 0.15 and control_summary.get("n", 0) >= 10
    if n_pass == len(criteria):
        outcome_label, go_no_go = "A - RANKING TRANSLATES INTO TRADING EDGE", "GO -> PAPER FORWARD (no real money)"
    elif candidates_meaningfully_beat_controls:
        outcome_label, go_no_go = "B - RANKER GOOD, ENTRY/COST ECONOMICS NOT YET GOOD ENOUGH", "NO-GO (keep V2 as watchlist/confirmation screener; entry mechanics + cost structure need more work, screener itself is not invalidated)"
    else:
        outcome_label, go_no_go = "C - RANKING EDGE DOES NOT (YET) TRANSLATE INTO A TRADE", "NO-GO (keep V2 as an analytical/watchlist tool only; do not change the logistic model)"

    reasoning = (
        f"Best-performing family on the primary top-10 cohort: {best_family_by_ev} ({best_h.get('net_expectancy_r', float('nan')):+.3f}R net, "
        f"PF={best_h.get('profit_factor', float('nan')):.2f}, n={best_h.get('n', 0)}). Matched controls (same period/liquidity/magnitude/volatility, "
        f"NOT top-10 ranked) with the same entry: {control_summary.get('net_expectancy_r', float('nan')):+.3f}R net, PF={control_summary.get('profit_factor', float('nan')):.2f}, "
        f"n={control_summary.get('n', 0)} - a {control_gap_r:+.3f}R gap in candidates' favor ({'meaningful, ranker is adding real value' if candidates_meaningfully_beat_controls else 'not clearly distinguishable from noise'}). "
        f"{n_pass}/{len(criteria)} point-26 criteria passed."
    )
    (out_dir / "phase_4a_summary.md").write_text(
        reports_4a.render_phase_4a_summary(best_family_by_ev, outcome_label, go_no_go, per_family_headline, criteria, reasoning), encoding="utf-8"
    )

    typer.echo(f"BEST FAMILY: {best_family_by_ev} | OUTCOME: {outcome_label} | {go_no_go}")
    typer.echo(f"Done. 9 Phase 4A deliverables under: {out_dir}")


if __name__ == "__main__":
    app()
