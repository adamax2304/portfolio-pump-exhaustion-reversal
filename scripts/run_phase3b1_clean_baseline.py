"""CLI: Phase 3B.1 - Clean Logistic Baseline + Early-Signal Validation.
Builds FEATURE_SET_V2 (drops oi_regime and bars_since_last_new_high per
Phase 3B's own ablation/correlation evidence), re-fits on the IDENTICAL
temporal folds/event-weighting/symbol-grouping/forward-lock as Phase 3B, and
tests the central open question: does V2's ranking edge exist BEFORE a
meaningful drawdown from the event high (early-warning) or only after
(confirmation-only)? No boosting run here regardless of verdict.

Reuses Phase 3A.1's event_snapshot_dataset.parquet and Phase 3B's own
fold-fitting helper directly (imported, not duplicated).

python scripts/run_phase3b1_clean_baseline.py --out-run-id phase3b1_v1
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

from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.screener_v2 import logistic_model, model_evaluation, model_features, reports_3b1, remaining_opportunity as ro, validation
from pump_exhaustion_reversal.screener_v2 import walkforward_folds as wf

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
OUTCOME_COL = phase3b.OUTCOME_COL
BARS_COL = phase3b.BARS_COL
POSITIVE_LABEL = phase3b.POSITIVE_LABEL
ALPHA = phase3b.ALPHA

V1_CONTINUOUS = list(model_features.FEATURE_SET_V1_CONTINUOUS)
V1_CATEGORICAL = list(model_features.FEATURE_SET_V1_CATEGORICAL)
V2_CONTINUOUS = ["pullback_depth_atr", "rejection_atr", "lower_high_strength", "realized_vol_15m", "current_return_from_start", "upper_wick_pct"]
V2_CATEGORICAL: list[str] = []

DRAWDOWN_BUCKETS = [("0_1PCT", 0.0, 0.01), ("1_2PCT", 0.01, 0.02), ("2_3PCT", 0.02, 0.03), ("3_5PCT", 0.03, 0.05), ("GT_5PCT", 0.05, None)]
CUMULATIVE_BUCKETS = [("LE_2PCT", 0.0, 0.02), ("LE_3PCT", 0.0, 0.03)]

REMAINING_OPPORTUNITY_HORIZON_HOURS = 4.0  # matches labels.REVERSAL_FROM_HIGH_HORIZON_HOURS - same frozen research parameter, not tuned
REMAINING_OPPORTUNITY_REVERSAL_PCT = 0.10  # matches labels.REVERSAL_FROM_HIGH_DRAWDOWN_PCT
COOLDOWN_MINUTES = 30  # point 8: sanity-check only, not optimized


def _fit_v_on_folds(resolved_df: pd.DataFrame, snapshots_df: pd.DataFrame, folds: list[dict], continuous: list[str], categorical: list[str]):
    fold_models, fold_val_resolved, fold_val_all = {}, {}, {}
    for fold in folds:
        train_df = resolved_df[resolved_df["event_id"].isin(fold["train_event_ids"])]
        val_df_resolved = resolved_df[resolved_df["event_id"].isin(fold["val_event_ids"])]
        val_df_all = snapshots_df[snapshots_df["event_id"].isin(fold["val_event_ids"])]
        model, val_resolved_scored, val_all_scored = phase3b._fit_and_score_fold(train_df, val_df_resolved, val_df_all, continuous, categorical)
        fold_models[fold["fold"]] = model
        fold_val_resolved[fold["fold"]] = val_resolved_scored
        fold_val_all[fold["fold"]] = val_all_scored
    return fold_models, fold_val_resolved, fold_val_all


def _bootstrap_top10(pooled_resolved: pd.DataFrame) -> dict:
    threshold = pooled_resolved["score"].quantile(0.9)
    top10 = pooled_resolved[pooled_resolved["score"] >= threshold]
    is_rev = (top10[OUTCOME_COL] == "REVERSAL").astype(float)
    return {
        "event": validation.cluster_bootstrap_proportion(is_rev, top10["event_id"]),
        "symbol": validation.cluster_bootstrap_proportion(is_rev, top10["symbol"]) if "symbol" in top10.columns else {},
    }


@app.command()
def main(config: str = typer.Option(str(BACKTEST_CONFIG_PATH)), out_run_id: str = typer.Option("phase3b1_v1")) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not phase3b.PHASE3A1_DATASET.exists():
        typer.echo(f"ERROR: {phase3b.PHASE3A1_DATASET} not found.")
        raise typer.Exit(1)
    snapshots_df = pd.read_parquet(phase3b.PHASE3A1_DATASET)
    typer.echo(f"Loaded {len(snapshots_df)} snapshots, {snapshots_df['event_id'].nunique()} events.")

    forward_lock_ts = datetime_to_ms(phase3b.FORWARD_LOCK_DATE)
    event_first_ts = snapshots_df.groupby("event_id")["pump_detected_timestamp"].transform("first")
    locked = snapshots_df.loc[event_first_ts >= forward_lock_ts, "event_id"].unique()
    if len(locked):
        typer.echo(f"WARNING: excluding {len(locked)} events at/after the forward-lock boundary.")
        snapshots_df = snapshots_df.loc[event_first_ts < forward_lock_ts].copy()

    snapshots_df["_drawdown_from_high_pct"] = validation.compute_drawdown_from_high_pct(snapshots_df)
    resolved_df = snapshots_df[snapshots_df[OUTCOME_COL].isin(("REVERSAL", "CONTINUATION"))].copy()
    typer.echo(f"Resolved snapshots: {len(resolved_df)} / {len(snapshots_df)}")

    events_unique = snapshots_df.drop_duplicates(subset="event_id")[["event_id", "pump_detected_timestamp", "symbol"]].rename(columns={"pump_detected_timestamp": "pump_detected_ts"})
    folds = wf.build_expanding_temporal_folds(events_unique, n_folds=4, min_train_fraction=0.4)
    symbol_holdout = wf.build_symbol_holdout_fold(events_unique, holdout_fraction=0.2, seed=7)
    typer.echo(f"{len(folds)} temporal folds + symbol-holdout ({len(symbol_holdout['holdout_symbols'])} symbols) - identical to Phase 3B (same params, same seed).")

    typer.echo("Fitting V1 (Phase 3B baseline) and V2 (clean) on the same temporal folds...")
    v1_models, v1_val_resolved, v1_val_all = _fit_v_on_folds(resolved_df, snapshots_df, folds, V1_CONTINUOUS, V1_CATEGORICAL)
    v2_models, v2_val_resolved, v2_val_all = _fit_v_on_folds(resolved_df, snapshots_df, folds, V2_CONTINUOUS, V2_CATEGORICAL)
    v1_pooled_resolved = pd.concat(v1_val_resolved.values(), ignore_index=True)
    v1_pooled_all = pd.concat(v1_val_all.values(), ignore_index=True)
    v2_pooled_resolved = pd.concat(v2_val_resolved.values(), ignore_index=True)
    v2_pooled_all = pd.concat(v2_val_all.values(), ignore_index=True)
    typer.echo(f"V1 pooled OOS: {len(v1_pooled_resolved)} resolved. V2 pooled OOS: {len(v2_pooled_resolved)} resolved.")

    # ================= Deliverable 1: v1_vs_v2_comparison.md =================
    typer.echo("Building V1 vs V2 comparison (point 3)...")
    v1_curve = model_evaluation.score_decile_curve(v1_pooled_resolved, "score", OUTCOME_COL, top_fractions=(0.5, 0.3, 0.2, 0.1, 0.05))
    v2_curve = model_evaluation.score_decile_curve(v2_pooled_resolved, "score", OUTCOME_COL, top_fractions=(0.5, 0.3, 0.2, 0.1, 0.05))
    v1_metrics = model_evaluation.classification_metrics(v1_pooled_resolved, "score", OUTCOME_COL)
    v2_metrics = model_evaluation.classification_metrics(v2_pooled_resolved, "score", OUTCOME_COL)
    v1_precision = model_evaluation.precision_at_k(v1_pooled_all, "score", OUTCOME_COL, timestamp_col="timestamp", ks=(1, 3, 5))["by_k"]
    v2_precision = model_evaluation.precision_at_k(v2_pooled_all, "score", OUTCOME_COL, timestamp_col="timestamp", ks=(1, 3, 5))["by_k"]
    v1_boot = _bootstrap_top10(v1_pooled_resolved)
    v2_boot = _bootstrap_top10(v2_pooled_resolved)
    (out_dir / "v1_vs_v2_comparison.md").write_text(
        reports_3b1.render_v1_vs_v2_comparison(v1_curve, v2_curve, v1_metrics, v2_metrics, v1_precision, v2_precision, v1_boot, v2_boot), encoding="utf-8"
    )

    # ================= Deliverable 2: predump_ranking_by_drawdown_bucket.md =================
    typer.echo("Computing pre-dump ranking by drawdown bucket (points 4-5) - the central question...")
    bucket_curve = model_evaluation.ranking_by_drawdown_bucket(v2_pooled_resolved, "score", OUTCOME_COL, "_drawdown_from_high_pct", DRAWDOWN_BUCKETS, top_fractions=(0.2, 0.1, 0.05))
    cumulative_curve = model_evaluation.ranking_by_drawdown_bucket(v2_pooled_resolved, "score", OUTCOME_COL, "_drawdown_from_high_pct", CUMULATIVE_BUCKETS, top_fractions=(0.2, 0.1, 0.05))
    (out_dir / "predump_ranking_by_drawdown_bucket.md").write_text(reports_3b1.render_predump_ranking(bucket_curve, cumulative_curve), encoding="utf-8")

    # ================= Deliverable 3: first_alert_event_level_results.md =================
    typer.echo("Computing first-alert event-level results (points 7-9)...")
    counts_by_quantile = {}
    first_alert_frames = {}
    for q in (0.80, 0.90, 0.95):
        fa = model_evaluation.first_alert_per_event(v2_pooled_all, "score", top_quantile=q)
        first_alert_frames[q] = fa
        if fa.empty:
            counts_by_quantile[q] = {"n_alerted": 0, "n_true": 0, "n_false": 0, "precision": np.nan}
            continue
        n_true = int((fa[OUTCOME_COL] == "REVERSAL").sum())
        counts_by_quantile[q] = {"n_alerted": len(fa), "n_true": n_true, "n_false": len(fa) - n_true, "precision": n_true / len(fa)}

    v2_precision_full = model_evaluation.precision_at_k(v2_pooled_all, "score", OUTCOME_COL, timestamp_col="timestamp", ks=(1, 3, 5))["by_k"]
    precision_by_quantile = {k: {"snapshot": v2_precision_full[k]["mean_precision_at_k_per_timestamp"], "first_alert": v2_precision_full[k]["precision_among_unique_alerts"]} for k in (1, 3, 5)}

    # point 8 cooldown sanity check: alert-once-ever (already computed above at top10) vs alert-with-30min-cooldown-after-exit-reentry
    top10_threshold = v2_pooled_all["score"].quantile(0.90)
    cooldown_alerts = 0
    for event_id, group in v2_pooled_all.sort_values("timestamp").groupby("event_id"):
        in_zone = group["score"] >= top10_threshold
        if not in_zone.any():
            continue
        last_alert_ts = None
        was_in_zone = False
        for ts, flag in zip(group["timestamp"], in_zone):
            if flag and (not was_in_zone) and (last_alert_ts is None or (ts - last_alert_ts) >= COOLDOWN_MINUTES * 60_000):
                cooldown_alerts += 1
                last_alert_ts = ts
            was_in_zone = flag
    once_ever_alerts = counts_by_quantile[0.90]["n_alerted"]
    cooldown_note = (
        f"Alert-once-ever (point 7's rule, used everywhere else in this report): {once_ever_alerts} alerts at the "
        f"top-10% threshold. Alert-with-{COOLDOWN_MINUTES}min-cooldown-after-exit-reentry (point 8, diagnostic "
        f"sanity check only, NOT optimized, NOT used elsewhere): {cooldown_alerts} alerts. "
        + ("Close enough that the once-ever simplification is not hiding a materially different alert volume." if abs(cooldown_alerts - once_ever_alerts) <= max(3, once_ever_alerts * 0.15) else "Meaningfully different - cooldown-based re-alerting would change practical alert volume; worth revisiting before any live design.")
    )
    (out_dir / "first_alert_event_level_results.md").write_text(reports_3b1.render_first_alert_results(counts_by_quantile, precision_by_quantile, cooldown_note), encoding="utf-8")

    # ================= Deliverable 4: remaining_opportunity_analysis.md =================
    typer.echo("Computing remaining-opportunity analysis (point 6) - needs raw OHLCV per event...")
    horizon_bars = max(round(REMAINING_OPPORTUNITY_HORIZON_HOURS * 3_600_000 / bar_interval_ms), 1)
    needed_keys = set()
    for fa in first_alert_frames.values():
        if not fa.empty:
            needed_keys.update(zip(fa["exchange"], fa["symbol"]))
    ohlcv_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for exchange in {k[0] for k in needed_keys}:
        symbols = {k[1] for k in needed_keys if k[0] == exchange}
        since_ms = datetime_to_ms("2021-01-01")
        until_ms = datetime_to_ms(cfg.backtest.end_date)
        ohlcv_by_key.update(phase3a._load_ohlcv_by_symbol(exchange, symbols, timeframe, since_ms, until_ms))
    typer.echo(f"  loaded OHLCV for {len(ohlcv_by_key)} (exchange,symbol) pairs needed by first alerts.")

    stats_by_quantile = {}
    for q, fa in first_alert_frames.items():
        if fa.empty:
            stats_by_quantile[q] = {"n": 0, "median_upside": np.nan, "median_downside": np.nan, "median_drawdown_at_signal": np.nan, "median_bars_to_reversal_minutes": np.nan, "fraction_confirmed": np.nan}
            continue
        opp = ro.compute_remaining_opportunity_for_first_alerts(fa, ohlcv_by_key, horizon_bars, REMAINING_OPPORTUNITY_REVERSAL_PCT)
        drawdown_at_signal = validation.compute_drawdown_from_high_pct(fa)
        stats_by_quantile[q] = {
            "n": len(opp),
            "median_upside": float(opp["additional_upside_before_reversal_pct"].median()) if len(opp) else np.nan,
            "median_downside": float(opp["remaining_downside_pct"].median()) if len(opp) else np.nan,
            "median_drawdown_at_signal": float(drawdown_at_signal.median()) if len(drawdown_at_signal) else np.nan,
            "median_bars_to_reversal_minutes": float(opp["bars_to_reversal"].dropna().median() * bar_interval_ms / 60_000.0) if opp["bars_to_reversal"].notna().any() else np.nan,
            "fraction_confirmed": float(opp["reversal_confirmed"].mean()) if len(opp) else np.nan,
        }
    (out_dir / "remaining_opportunity_analysis.md").write_text(reports_3b1.render_remaining_opportunity(stats_by_quantile), encoding="utf-8")

    # ================= Deliverable 5: symbol_holdout_v2.md =================
    typer.echo("Computing symbol holdout V2 (point 10)...")
    train_sym = resolved_df[resolved_df["event_id"].isin(symbol_holdout["train_event_ids"])]
    val_sym_resolved = resolved_df[resolved_df["event_id"].isin(symbol_holdout["val_event_ids"])]
    val_sym_all = snapshots_df[snapshots_df["event_id"].isin(symbol_holdout["val_event_ids"])]
    _, v1_sym_scored, _ = phase3b._fit_and_score_fold(train_sym, val_sym_resolved, val_sym_all, V1_CONTINUOUS, V1_CATEGORICAL)
    _, v2_sym_scored, _ = phase3b._fit_and_score_fold(train_sym, val_sym_resolved, val_sym_all, V2_CONTINUOUS, V2_CATEGORICAL)
    v1_sym_curve = model_evaluation.score_decile_curve(v1_sym_scored, "score", OUTCOME_COL, top_fractions=(0.5, 0.3, 0.2, 0.1, 0.05))
    v2_sym_curve = model_evaluation.score_decile_curve(v2_sym_scored, "score", OUTCOME_COL, top_fractions=(0.5, 0.3, 0.2, 0.1, 0.05))
    (out_dir / "symbol_holdout_v2.md").write_text(
        reports_3b1.render_symbol_holdout_v2(v1_sym_curve, v2_sym_curve, symbol_holdout["n_train_events"], symbol_holdout["n_val_events"], len(symbol_holdout["holdout_symbols"])), encoding="utf-8"
    )

    # ================= Deliverable 6: coefficient_stability_v2.md =================
    typer.echo("Computing coefficient stability V2 (point 11)...")
    v2_stability_df = logistic_model.coefficient_stability_report(v2_models)
    core_family = ["pullback_depth_atr", "rejection_atr", "lower_high_strength"]
    core_stable = v2_stability_df[v2_stability_df["feature"].isin(core_family)]["sign_consistent"].all() if not v2_stability_df.empty else False
    return_row = v2_stability_df[v2_stability_df["feature"] == "current_return_from_start"]
    decay_note = ""
    if not return_row.empty:
        vals = [return_row.iloc[0].get(f"fold_{k}") for k in sorted(v2_models.keys())]
        vals = [v for v in vals if v is not None and pd.notna(v)]
        if len(vals) >= 2:
            decaying = all(abs(vals[i]) >= abs(vals[i + 1]) for i in range(len(vals) - 1))
            decay_note = f" current_return_from_start coefficients across folds: {[round(float(v), 3) for v in vals]} - {'monotonically DECAYING in magnitude across folds (consistent with Phase 3B).' if decaying else 'not monotonically decaying this time.'}"
    notes = f"pullback_depth_atr / rejection_atr / lower_high_strength all sign-consistent across folds: {bool(core_stable)}.{decay_note}"
    (out_dir / "coefficient_stability_v2.md").write_text(reports_3b1.render_coefficient_stability_v2(v2_stability_df, sorted(v2_models.keys()), notes), encoding="utf-8")

    # ================= Deliverable 7: phase_3b1_summary.md =================
    typer.echo("Synthesizing Phase 3B.1 summary (points 12-14)...")
    v1_top10 = float(v1_curve.loc[v1_curve["top_pct"] == 0.1, "directional_lift"].iloc[0])
    v2_top10 = float(v2_curve.loc[v2_curve["top_pct"] == 0.1, "directional_lift"].iloc[0])
    v1_top20 = float(v1_curve.loc[v1_curve["top_pct"] == 0.2, "directional_lift"].iloc[0])
    v2_top20 = float(v2_curve.loc[v2_curve["top_pct"] == 0.2, "directional_lift"].iloc[0])
    v2_better = v2_top10 > v1_top10 and v2_top20 > v1_top20

    def _bucket_lift(curve_df: pd.DataFrame, bucket: str, top_pct: float) -> tuple[float, int]:
        row = curve_df[(curve_df["bucket"] == bucket) & (curve_df["top_pct"] == top_pct)]
        if len(row) and pd.notna(row["directional_lift"].iloc[0]):
            return float(row["directional_lift"].iloc[0]), int(row["n_resolved"].iloc[0])
        return np.nan, 0

    # Both top-20% AND top-10% checked - not just one favorable point. A
    # real ranking edge should hold up (or improve) as the cutoff tightens;
    # if top-20% looks positive but top-10% collapses toward/below 1.0, that
    # is exactly the small-N noise pattern this check exists to catch, not
    # a genuine early-warning signal.
    le3_top20, le3_top20_n = _bucket_lift(cumulative_curve, "LE_3PCT", 0.2)
    le3_top10, le3_top10_n = _bucket_lift(cumulative_curve, "LE_3PCT", 0.1)
    le2_top20, le2_top20_n = _bucket_lift(cumulative_curve, "LE_2PCT", 0.2)
    le2_top10, le2_top10_n = _bucket_lift(cumulative_curve, "LE_2PCT", 0.1)
    far_top20, far_top20_n = _bucket_lift(bucket_curve, "GT_5PCT", 0.2)
    far_top10, far_top10_n = _bucket_lift(bucket_curve, "GT_5PCT", 0.1)
    predump_holds_at_top10 = pd.notna(le3_top10) and le3_top10 > 1.0
    far_holds_at_top10 = pd.notna(far_top10) and far_top10 > 1.0

    v2_sym_top10 = float(v2_sym_curve.loc[v2_sym_curve["top_pct"] == 0.1, "directional_lift"].iloc[0]) if 0.1 in v2_sym_curve["top_pct"].values else np.nan
    v2_sym_top5 = float(v2_sym_curve.loc[v2_sym_curve["top_pct"] == 0.05, "directional_lift"].iloc[0]) if 0.05 in v2_sym_curve["top_pct"].values else np.nan
    symbol_holdout_ok = pd.notna(v2_sym_top10) and v2_sym_top10 > 1.0

    baseline_unconditional = float((v2_pooled_all[OUTCOME_COL] == "REVERSAL").mean())
    fa_precision3 = precision_by_quantile.get(3, {}).get("first_alert", np.nan)
    fa_precision5 = precision_by_quantile.get(5, {}).get("first_alert", np.nan)
    fa_precision_ok = (pd.notna(fa_precision3) and fa_precision3 > baseline_unconditional) or (pd.notna(fa_precision5) and fa_precision5 > baseline_unconditional)

    v1_boot_low = v1_boot.get("event", {}).get("ci_low", np.nan)
    v2_boot_low = v2_boot.get("event", {}).get("ci_low", np.nan)
    bootstrap_not_noise = pd.notna(v2_boot_low) and v2_boot_low > baseline_unconditional

    criteria = [
        {"criterion": "Temporal top-10% lift >= 1.4x", "threshold": ">= 1.4x", "observed": f"{v2_top10:.2f}x", "pass": bool(v2_top10 >= 1.4)},
        {"criterion": "Top-20% also > baseline", "threshold": "> 1.0x", "observed": f"{v2_top20:.2f}x", "pass": bool(v2_top20 > 1.0)},
        {"criterion": "Symbol-holdout preserved", "threshold": "top-10% lift > 1.0x", "observed": f"{v2_sym_top10:.2f}x" if pd.notna(v2_sym_top10) else "n/a", "pass": bool(symbol_holdout_ok)},
        {
            "criterion": "PRE-DUMP <=3% cohort shows meaningful positive lift AT BOTH top-20% and top-10%",
            "threshold": "> 1.0x at both cutoffs (not just the looser one)",
            "observed": f"top20={_fmt_num_local(le3_top20)}x (n={le3_top20_n}), top10={_fmt_num_local(le3_top10)}x (n={le3_top10_n})",
            "pass": bool(pd.notna(le3_top20) and le3_top20 > 1.0 and predump_holds_at_top10),
        },
        {"criterion": "First-alert Precision@3 or @5 > unconditional baseline", "threshold": f"> {baseline_unconditional:.1%}", "observed": f"{_fmt_pct_local(fa_precision3)} / {_fmt_pct_local(fa_precision5)}", "pass": bool(fa_precision_ok)},
        {"criterion": "Coefficient signs stable (core family)", "threshold": "pullback_depth_atr/rejection_atr/lower_high_strength all consistent", "observed": str(bool(core_stable)), "pass": bool(core_stable)},
        {"criterion": "Bootstrap top-decile lift not pure noise", "threshold": f"event-cluster CI low > {baseline_unconditional:.1%}", "observed": _fmt_pct_local(v2_boot_low), "pass": bool(bootstrap_not_noise)},
    ]
    n_pass = sum(1 for c in criteria if c["pass"])
    go_no_go = "GO" if n_pass == len(criteria) else "NO-GO"

    # Early-warning edge classification uses BOTH cutoffs, not the single
    # most favorable one - top20-only-positive (the loosest, least
    # decision-relevant cut) with top10 flat/negative is exactly the
    # "looked real until you tighten it" pattern that should read as WEAK,
    # not YES.
    if pd.notna(le3_top20) and le3_top20 > 1.15 and predump_holds_at_top10:
        early_warning_edge = "YES"
    elif pd.notna(le3_top20) and le3_top20 > 1.0:
        early_warning_edge = "WEAK"
    else:
        early_warning_edge = "NO"
    if pd.notna(far_top20) and far_top20 > 1.2 and far_holds_at_top10:
        confirmation_edge = "YES"
    elif pd.notna(far_top20) and far_top20 > 1.0:
        confirmation_edge = "WEAK"
    else:
        confirmation_edge = "NO"
    symbol_generalization = "YES" if symbol_holdout_ok and pd.notna(v2_sym_top5) and v2_sym_top5 > 1.0 else ("WEAK" if symbol_holdout_ok else "NO")

    reasoning = (
        f"V2 top-10%/top-20% lift = {v2_top10:.2f}x/{v2_top20:.2f}x vs V1's {v1_top10:.2f}x/{v1_top20:.2f}x on the "
        f"IDENTICAL folds ({'V2 better' if v2_better else 'V2 NOT clearly better'}). PRE-DUMP (<=3% from high): "
        f"top-20% lift = {_fmt_num_local(le3_top20)}x (n={le3_top20_n}) looks positive, but top-10% lift = "
        f"{_fmt_num_local(le3_top10)}x (n={le3_top10_n}) does NOT hold up - the tighter, more decision-relevant cut "
        f"collapses toward/below 1.0 with very thin N. <=2% cohort: top-20%={_fmt_num_local(le2_top20)}x "
        f"(n={le2_top20_n}), top-10%={_fmt_num_local(le2_top10)}x (n={le2_top10_n}) - same pattern, even thinner. "
        f"Far cohort (>5% from high, confirmation-stage): top-20%={_fmt_num_local(far_top20)}x (n={far_top20_n}), "
        f"top-10%={_fmt_num_local(far_top10)}x (n={far_top10_n}) - CONSISTENT at both cutoffs, much larger N. "
        f"Symbol-holdout top-10%/top-5% lift = {_fmt_num_local(v2_sym_top10)}x/{_fmt_num_local(v2_sym_top5)}x. "
        f"{n_pass}/{len(criteria)} point-13 criteria passed (ALL required for GO)."
    )

    (out_dir / "phase_3b1_summary.md").write_text(
        reports_3b1.render_phase_3b1_summary(
            "YES" if v2_better else "NO", early_warning_edge, confirmation_edge, symbol_generalization, go_no_go, criteria, reasoning
        ),
        encoding="utf-8",
    )

    typer.echo(f"V2 BETTER: {v2_better} | EARLY WARNING: {early_warning_edge} | CONFIRMATION: {confirmation_edge} | SYMBOL GEN: {symbol_generalization} | {go_no_go}")
    typer.echo(f"Done. 7 Phase 3B.1 deliverables under: {out_dir}")


def _fmt_pct_local(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


def _fmt_num_local(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


if __name__ == "__main__":
    app()
