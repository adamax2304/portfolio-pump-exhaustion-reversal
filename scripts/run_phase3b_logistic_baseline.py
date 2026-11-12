"""CLI: Phase 3B - Pump Exhaustion Ranking Model, logistic baseline only.
User's explicit gate: Phase 3A.1 proved directional information exists in
snapshot state; Phase 3B tests whether a SMALL, pre-registered, interpretable
feature set can rank active pump states by P(REVERSAL | resolved) with a
regularized logistic regression - still not a trading strategy, no entry/SL/
TP/risk engine, no PnL anywhere in this script.

Reuses Phase 3A.1's already-built event_snapshot_dataset.parquet directly
(data/diagnostics/phase3a1_v1/) - no new snapshot building. Builds the 10
Phase 3B deliverables, ending with an explicit RANKING EDGE / GO-NO-GO
verdict, then stops - no automatic transition to gradient boosting (point 24)
regardless of the verdict.

python scripts/run_phase3b_logistic_baseline.py --out-run-id phase3b_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import typer

from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.screener_v2 import logistic_model, model_evaluation, model_features, ranking, reports_3b, validation
from pump_exhaustion_reversal.screener_v2 import walkforward_folds as wf

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
PHASE3A1_DATASET = DIAGNOSTICS_ROOT / "phase3a1_v1" / "event_snapshot_dataset.parquet"

OUTCOME_COL = "primary_outcome"
BARS_COL = "primary_bars"
POSITIVE_LABEL = "REVERSAL"
ALPHA = 1.0  # L2 strength - fixed once, disclosed, not tuned against any validation metric (no grid search in this baseline pass)
FORWARD_LOCK_DATE = "2026-08-27"  # point 23: absolutely locked, never read by this script
HIGH_STATE_MIN_POSITION = 0.90  # same threshold Phase 3A.1 used, for continuity

# Point 1: Phase 3A.1's KEEP+WEAK candidates, in that exact priority order
# (phase_3a1_summary.md's own row order - KEEP first, then WEAK).
KEEP_WEAK_PRIORITY_ORDER = [
    "realized_vol_15m", "current_return_from_start", "upper_wick_pct", "bars_since_last_new_high",
    "time_since_last_new_high_minutes", "pullback_duration_bars", "pullback_depth_atr", "rejection_atr",
    "lower_high_strength", "realized_vol_1h", "volume_current_over_climax", "volume_peak_to_current_pct",
    "average_pullback_depth_pct", "number_of_new_highs_60m", "oi_change_60m", "volume_per_return",
]


def _fit_and_score_fold(train_df: pd.DataFrame, val_df_resolved: pd.DataFrame, val_df_all: pd.DataFrame, continuous: list[str], categorical: list[str]):
    model = logistic_model.fit_l2_logistic(train_df, continuous, categorical, OUTCOME_COL, POSITIVE_LABEL, alpha=ALPHA)
    val_resolved_scored = val_df_resolved.copy()
    val_resolved_scored["score"] = logistic_model.predict_proba(model, val_resolved_scored)
    val_all_scored = val_df_all.copy()
    val_all_scored["score"] = logistic_model.predict_proba(model, val_all_scored)
    return model, val_resolved_scored, val_all_scored


@app.command()
def main(config: str = typer.Option(str(BACKTEST_CONFIG_PATH)), out_run_id: str = typer.Option("phase3b_v1")) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    bar_interval_ms = parse_timeframe_ms(cfg.backtest.timeframe_base)
    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not PHASE3A1_DATASET.exists():
        typer.echo(f"ERROR: {PHASE3A1_DATASET} not found - run run_phase3a1_directional_audit.py first.")
        raise typer.Exit(1)
    snapshots_df = pd.read_parquet(PHASE3A1_DATASET)
    typer.echo(f"Loaded {len(snapshots_df)} snapshots, {snapshots_df['event_id'].nunique()} events from Phase 3A.1's dataset.")

    # ---- point 23: forward-sample lock guard - never read past this date ----
    forward_lock_ts = datetime_to_ms(FORWARD_LOCK_DATE)
    event_first_ts = snapshots_df.groupby("event_id")["pump_detected_timestamp"].transform("first")
    locked_events = snapshots_df.loc[event_first_ts >= forward_lock_ts, "event_id"].unique()
    if len(locked_events):
        typer.echo(f"WARNING: {len(locked_events)} events detected at/after the {FORWARD_LOCK_DATE} forward-lock boundary - EXCLUDING them, never used for feature/model/threshold selection.")
        snapshots_df = snapshots_df.loc[event_first_ts < forward_lock_ts].copy()
    else:
        typer.echo(f"Forward-lock check: no events at/after {FORWARD_LOCK_DATE} in this dataset (max pump_detected_timestamp = {pd.to_datetime(snapshots_df['pump_detected_timestamp'].max(), unit='ms')}) - lock is a no-op here, by construction.")

    # ---- high-state flag (point 10), computed once up front ----
    position = validation.compute_price_position_in_range_pct(snapshots_df)
    snapshots_df["_high_state"] = position >= HIGH_STATE_MIN_POSITION

    resolved_df = snapshots_df[snapshots_df[OUTCOME_COL].isin(("REVERSAL", "CONTINUATION"))].copy()
    typer.echo(f"Resolved (REVERSAL/CONTINUATION) snapshots for the direction model: {len(resolved_df)} / {len(snapshots_df)} ({len(resolved_df) / len(snapshots_df):.1%})")

    # ================= Deliverable 1: feature_correlation_and_dedup.md =================
    typer.echo("Computing feature correlation + dedup (point 1)...")
    corr_matrix = model_features.compute_feature_correlation_matrix(resolved_df, KEEP_WEAK_PRIORITY_ORDER)
    dedup_result = model_features.deduplicate_correlated_features(corr_matrix, KEEP_WEAK_PRIORITY_ORDER, threshold=0.85)
    (out_dir / "feature_correlation_and_dedup.md").write_text(
        reports_3b.render_feature_correlation_and_dedup(corr_matrix, dedup_result, list(model_features.FEATURE_SET_V1_CONTINUOUS), model_features.KNOWN_ALIAS_FAMILIES),
        encoding="utf-8",
    )

    # ================= temporal + symbol-holdout folds (points 7-9) =================
    events_unique = snapshots_df.drop_duplicates(subset="event_id")[["event_id", "pump_detected_timestamp", "symbol"]].rename(columns={"pump_detected_timestamp": "pump_detected_ts"})
    folds = wf.build_expanding_temporal_folds(events_unique, n_folds=4, min_train_fraction=0.4)
    symbol_holdout = wf.build_symbol_holdout_fold(events_unique, holdout_fraction=0.2, seed=7)
    typer.echo(f"Built {len(folds)} temporal folds + 1 symbol-holdout fold ({len(symbol_holdout['holdout_symbols'])} held-out symbols).")

    continuous_v1 = list(model_features.FEATURE_SET_V1_CONTINUOUS)
    categorical_v1 = list(model_features.FEATURE_SET_V1_CATEGORICAL)

    fold_models: dict = {}
    fold_val_resolved: dict = {}
    fold_val_all: dict = {}
    fold_summaries = []
    typer.echo("Fitting + scoring temporal folds (baseline FEATURE_SET_V1)...")
    for fold in folds:
        train_df = resolved_df[resolved_df["event_id"].isin(fold["train_event_ids"])]
        val_df_resolved = resolved_df[resolved_df["event_id"].isin(fold["val_event_ids"])]
        val_df_all = snapshots_df[snapshots_df["event_id"].isin(fold["val_event_ids"])]
        model, val_resolved_scored, val_all_scored = _fit_and_score_fold(train_df, val_df_resolved, val_df_all, continuous_v1, categorical_v1)
        fold_models[fold["fold"]] = model
        fold_val_resolved[fold["fold"]] = val_resolved_scored
        fold_val_all[fold["fold"]] = val_all_scored
        curve = model_evaluation.score_decile_curve(val_resolved_scored, "score", OUTCOME_COL, top_fractions=(0.5, 0.3, 0.2, 0.1, 0.05))
        fold_summaries.append({"fold": fold["fold"], "n_train_events": fold["n_train_events"], "n_val_events": fold["n_val_events"], "curve": curve})
        typer.echo(f"  fold {fold['fold']}: train={fold['n_train_events']} events, val={fold['n_val_events']} events, converged={model.converged}")

    train_df_sym = resolved_df[resolved_df["event_id"].isin(symbol_holdout["train_event_ids"])]
    val_df_sym_resolved = resolved_df[resolved_df["event_id"].isin(symbol_holdout["val_event_ids"])]
    val_df_sym_all = snapshots_df[snapshots_df["event_id"].isin(symbol_holdout["val_event_ids"])]
    symbol_holdout_model, symbol_val_resolved_scored, symbol_val_all_scored = _fit_and_score_fold(train_df_sym, val_df_sym_resolved, val_df_sym_all, continuous_v1, categorical_v1)
    symbol_holdout_curve = model_evaluation.score_decile_curve(symbol_val_resolved_scored, "score", OUTCOME_COL, top_fractions=(0.5, 0.3, 0.2, 0.1, 0.05))
    symbol_holdout_summary = {"n_train_events": symbol_holdout["n_train_events"], "n_val_events": symbol_holdout["n_val_events"], "holdout_symbols": symbol_holdout["holdout_symbols"], "curve": symbol_holdout_curve}

    # ================= Deliverable 2: temporal_walkforward_results.md =================
    (out_dir / "temporal_walkforward_results.md").write_text(reports_3b.render_temporal_walkforward_results(fold_summaries, symbol_holdout_summary), encoding="utf-8")

    pooled_oos_resolved = pd.concat(fold_val_resolved.values(), ignore_index=True)
    pooled_oos_all = pd.concat(fold_val_all.values(), ignore_index=True)
    typer.echo(f"Pooled out-of-fold (temporal folds only): {len(pooled_oos_resolved)} resolved rows, {len(pooled_oos_all)} total scored rows.")

    # ================= Deliverable 3: score_decile_curve.md =================
    typer.echo("Computing pooled score decile curve (point 11)...")
    pooled_curve = model_evaluation.score_decile_curve(pooled_oos_resolved, "score", OUTCOME_COL, top_fractions=(0.5, 0.3, 0.2, 0.1, 0.05))
    per_fold_curves = {fs["fold"]: fs["curve"] for fs in fold_summaries}
    (out_dir / "score_decile_curve.md").write_text(reports_3b.render_score_decile_curve(pooled_curve, per_fold_curves), encoding="utf-8")

    # ================= Deliverable 4: high_state_ranking.md (point 10) =================
    typer.echo("Fitting HIGH_STATE_ONLY model variant (point 10)...")
    high_state_only_curves = []
    all_state_on_high_state_curves = []
    n_high_state_val_rows = 0
    for fold in folds:
        train_df = resolved_df[resolved_df["event_id"].isin(fold["train_event_ids"])]
        val_df_resolved = resolved_df[resolved_df["event_id"].isin(fold["val_event_ids"])]
        train_hs = train_df[train_df["_high_state"]]
        val_hs = val_df_resolved[val_df_resolved["_high_state"]]
        if train_hs.empty or val_hs.empty:
            continue
        n_high_state_val_rows += len(val_hs)
        hs_model = logistic_model.fit_l2_logistic(train_hs, continuous_v1, categorical_v1, OUTCOME_COL, POSITIVE_LABEL, alpha=ALPHA)
        val_hs_scored_by_hs_model = val_hs.copy()
        val_hs_scored_by_hs_model["score"] = logistic_model.predict_proba(hs_model, val_hs_scored_by_hs_model)
        high_state_only_curves.append(model_evaluation.score_decile_curve(val_hs_scored_by_hs_model, "score", OUTCOME_COL, top_fractions=(0.3, 0.2, 0.1)))

        val_hs_scored_by_all_state_model = val_hs.copy()
        val_hs_scored_by_all_state_model["score"] = logistic_model.predict_proba(fold_models[fold["fold"]], val_hs_scored_by_all_state_model)
        all_state_on_high_state_curves.append(model_evaluation.score_decile_curve(val_hs_scored_by_all_state_model, "score", OUTCOME_COL, top_fractions=(0.3, 0.2, 0.1)))

    def _average_curves(curves: list[pd.DataFrame]) -> pd.DataFrame:
        if not curves:
            return pd.DataFrame()
        combined = pd.concat(curves, ignore_index=True)
        return combined.groupby("top_pct", as_index=False).agg(
            {"n_snapshots": "sum", "n_resolved": "sum", "n_events": "sum", "n_unique_symbols": "sum", "p_reversal_given_resolved": "mean", "directional_lift": "mean"}
        )

    all_state_on_hs_avg = _average_curves(all_state_on_high_state_curves)
    hs_only_on_hs_avg = _average_curves(high_state_only_curves)
    (out_dir / "high_state_ranking.md").write_text(
        reports_3b.render_high_state_ranking(all_state_on_hs_avg, hs_only_on_hs_avg, n_high_state_val_rows, len(pooled_oos_resolved)), encoding="utf-8"
    )

    # ================= Deliverable 5: precision_at_k.md (point 13) =================
    typer.echo("Computing Precision@K (point 13)...")
    precision_result = model_evaluation.precision_at_k(pooled_oos_all, "score", OUTCOME_COL, timestamp_col="timestamp", ks=(1, 3, 5), min_concurrent=2)
    (out_dir / "precision_at_k.md").write_text(reports_3b.render_precision_at_k(precision_result), encoding="utf-8")

    # ================= Deliverable 6: lead_time_analysis.md (point 14) =================
    typer.echo("Computing lead-time analysis (point 14)...")
    lead_time_results = {}
    for q in (0.80, 0.90, 0.95):
        lead_time_results[q] = ranking.lead_time_analysis(pooled_oos_all, "score", OUTCOME_COL, BARS_COL, bar_interval_ms=bar_interval_ms, top_quantile=q)
    (out_dir / "lead_time_analysis.md").write_text(reports_3b.render_lead_time_analysis(lead_time_results), encoding="utf-8")

    # ================= Deliverable 7: coefficient_stability.md (point 15) =================
    typer.echo("Computing coefficient stability (point 15)...")
    stability_df = logistic_model.coefficient_stability_report(fold_models)
    (out_dir / "coefficient_stability.md").write_text(reports_3b.render_coefficient_stability(stability_df, sorted(fold_models.keys())), encoding="utf-8")

    # ================= Deliverable 8: feature_ablation.md (point 16) =================
    typer.echo("Computing feature ablation (point 16)...")
    ablation_results = {}
    for name, spec in model_features.ABLATION_VARIANTS.items():
        variant_val_scores = []
        for fold in folds:
            train_df = resolved_df[resolved_df["event_id"].isin(fold["train_event_ids"])]
            val_df_resolved = resolved_df[resolved_df["event_id"].isin(fold["val_event_ids"])]
            variant_model = logistic_model.fit_l2_logistic(train_df, spec["continuous"], spec["categorical"], OUTCOME_COL, POSITIVE_LABEL, alpha=ALPHA)
            scored = val_df_resolved.copy()
            scored["score"] = logistic_model.predict_proba(variant_model, scored)
            variant_val_scores.append(scored)
        variant_pooled = pd.concat(variant_val_scores, ignore_index=True)
        variant_curve = model_evaluation.score_decile_curve(variant_pooled, "score", OUTCOME_COL, top_fractions=(0.2, 0.1))
        top10 = variant_curve.loc[variant_curve["top_pct"] == 0.1, "directional_lift"]
        top20 = variant_curve.loc[variant_curve["top_pct"] == 0.2, "directional_lift"]
        ablation_results[name] = {
            "n_features": len(spec["continuous"]) + len(spec["categorical"]),
            "top10_lift": float(top10.iloc[0]) if len(top10) else np.nan,
            "top20_lift": float(top20.iloc[0]) if len(top20) else np.nan,
            "n_resolved": int(variant_pooled[OUTCOME_COL].isin(("REVERSAL", "CONTINUATION")).sum()),
        }
        typer.echo(f"  {name}: top10_lift={ablation_results[name]['top10_lift']:.2f}")
    (out_dir / "feature_ablation.md").write_text(reports_3b.render_feature_ablation(ablation_results), encoding="utf-8")

    # ================= Deliverable 9: calibration_and_bootstrap.md (points 21-22, 17) =================
    typer.echo("Computing calibration + bootstrap (points 21-22)...")
    calibration_df = model_evaluation.calibration_curve(pooled_oos_resolved, "score", OUTCOME_COL, n_bins=10)
    top10_threshold = pooled_oos_resolved["score"].quantile(0.9)
    top10_subset = pooled_oos_resolved[pooled_oos_resolved["score"] >= top10_threshold]
    top10_is_reversal = (top10_subset[OUTCOME_COL] == "REVERSAL").astype(float)
    event_bootstrap = validation.cluster_bootstrap_proportion(top10_is_reversal, top10_subset["event_id"])
    symbol_bootstrap = validation.cluster_bootstrap_proportion(top10_is_reversal, top10_subset["symbol"]) if "symbol" in top10_subset.columns else {}

    cross_exchange_note = None
    if "one_exchange_new_high_only" in snapshots_df.columns:
        ce = snapshots_df.dropna(subset=["one_exchange_new_high_only", OUTCOME_COL])
        ce_resolved = ce[ce[OUTCOME_COL].isin(("REVERSAL", "CONTINUATION"))].copy()
        ce_true = ce_resolved[ce_resolved["one_exchange_new_high_only"].astype(bool)]
        if len(ce_true):
            is_rev_true = (ce_true[OUTCOME_COL] == "REVERSAL").astype(float)
            ce_true_boot = validation.cluster_bootstrap_proportion(is_rev_true, ce_true["event_id"])
            baseline_ce = float((ce_resolved[OUTCOME_COL] == "REVERSAL").mean())
            cross_exchange_note = (
                f"N with one_exchange_new_high_only=True (resolved): {len(ce_true)}. Event-cluster bootstrap on its "
                f"reversal rate: {_fmt_pct_local(ce_true_boot.get('statistic'))} "
                f"[{_fmt_pct_local(ce_true_boot.get('ci_low'))}, {_fmt_pct_local(ce_true_boot.get('ci_high'))}] "
                f"(n_clusters={ce_true_boot.get('n_clusters', 0)}) vs {_fmt_pct_local(baseline_ce)} baseline among "
                "all cross-exchange-matched resolved snapshots. Per the user's explicit instruction: NOT used as a "
                "main feature across all events - too few clusters for the bootstrap to mean much at this N (matches "
                "Phase 3A.1's own disclosed N=38 caveat) - left as a future feature for a dedicated linked-events "
                "diagnostic model, not folded into FEATURE_SET_V1."
            )
    (out_dir / "calibration_and_bootstrap.md").write_text(
        reports_3b.render_calibration_and_bootstrap(calibration_df, event_bootstrap, symbol_bootstrap, cross_exchange_note), encoding="utf-8"
    )

    # ================= Deliverable 10: phase_3b_logistic_summary.md (points 18-20, 26) =================
    typer.echo("Synthesizing GO/NO-GO summary (points 18-20)...")
    top10_lift = float(pooled_curve.loc[pooled_curve["top_pct"] == 0.1, "directional_lift"].iloc[0])
    top20_lift = float(pooled_curve.loc[pooled_curve["top_pct"] == 0.2, "directional_lift"].iloc[0])
    monotonic = bool((pooled_curve.sort_values("top_pct")["directional_lift"].diff().dropna() <= 1e-9).all())
    per_fold_top10_lifts = [float(fs["curve"].loc[fs["curve"]["top_pct"] == 0.1, "directional_lift"].iloc[0]) for fs in fold_summaries if 0.1 in fs["curve"]["top_pct"].values]
    n_folds_with_lift_above_1 = sum(1 for v in per_fold_top10_lifts if v > 1.0)
    high_state_lift_10 = float(hs_only_on_hs_avg.loc[hs_only_on_hs_avg["top_pct"] == 0.1, "directional_lift"].iloc[0]) if not hs_only_on_hs_avg.empty and 0.1 in hs_only_on_hs_avg["top_pct"].values else np.nan
    prec3 = precision_result["by_k"].get(3, {})
    prec5 = precision_result["by_k"].get(5, {})
    baseline_rate = float((pooled_oos_resolved[OUTCOME_COL] == "REVERSAL").mean())
    # Precision@K ranks among ALL scored candidates (resolved AND neutral -
    # a live screener does not know in advance which is which), so its
    # naive-baseline comparison must be the UNCONDITIONAL reversal rate over
    # that same all-outcomes population, not the resolved-conditional rate
    # the decile-curve criteria above correctly use.
    baseline_rate_unconditional = float((pooled_oos_all[OUTCOME_COL] == "REVERSAL").mean())
    median_lead_time = lead_time_results[0.90].get("median_time_to_reversal_minutes") if lead_time_results[0.90].get("valid") else np.nan
    top_symbol_share = float(top10_subset["symbol"].value_counts(normalize=True).iloc[0]) if len(top10_subset) and "symbol" in top10_subset.columns else np.nan
    unstable_features = stability_df.loc[~stability_df["sign_consistent"].fillna(False), "feature"].tolist() if "sign_consistent" in stability_df.columns else []
    stable_features = stability_df.loc[stability_df["sign_consistent"].fillna(False), "feature"].tolist() if "sign_consistent" in stability_df.columns else []

    criteria = [
        {"criterion": "Top-10% directional lift", "threshold": ">= 1.4-1.5x", "observed": f"{top10_lift:.2f}x", "pass": bool(top10_lift >= 1.4)},
        {"criterion": "Top-20% directional lift also > baseline", "threshold": "> 1.0x", "observed": f"{top20_lift:.2f}x", "pass": bool(top20_lift > 1.0)},
        {"criterion": "Monotonic score buckets", "threshold": "mostly monotonic", "observed": str(monotonic), "pass": monotonic},
        {"criterion": "Effect across temporal folds", "threshold": ">= 3/4 folds lift > 1.0x", "observed": f"{n_folds_with_lift_above_1}/{len(per_fold_top10_lifts)}", "pass": bool(n_folds_with_lift_above_1 >= 3)},
        {"criterion": "Effect in HIGH_STATE cohort", "threshold": "> 1.0x", "observed": f"{high_state_lift_10:.2f}x" if pd.notna(high_state_lift_10) else "n/a", "pass": bool(pd.notna(high_state_lift_10) and high_state_lift_10 > 1.0)},
        {"criterion": "Precision@3 > baseline", "threshold": f"> {baseline_rate_unconditional:.1%} (unconditional)", "observed": _fmt_pct_local(prec3.get("precision_among_unique_alerts")), "pass": bool(prec3.get("precision_among_unique_alerts") and prec3["precision_among_unique_alerts"] > baseline_rate_unconditional)},
        {"criterion": "Precision@5 > baseline", "threshold": f"> {baseline_rate_unconditional:.1%} (unconditional)", "observed": _fmt_pct_local(prec5.get("precision_among_unique_alerts")), "pass": bool(prec5.get("precision_among_unique_alerts") and prec5["precision_among_unique_alerts"] > baseline_rate_unconditional)},
        {"criterion": "Median lead time practically useful", "threshold": ">= 30 min", "observed": f"{median_lead_time:.0f} min" if pd.notna(median_lead_time) else "n/a", "pass": bool(pd.notna(median_lead_time) and median_lead_time >= 30)},
        {"criterion": "Symbol concentration acceptable", "threshold": "top symbol <= 25% of top-10% alerts", "observed": _fmt_pct_local(top_symbol_share), "pass": bool(pd.notna(top_symbol_share) and top_symbol_share <= 0.25)},
    ]
    n_pass = sum(1 for c in criteria if c["pass"])
    if n_pass >= 7:
        ranking_edge, go_no_go = "YES", "GO"
    elif n_pass >= 4:
        ranking_edge, go_no_go = "WEAK", "NO-GO (revisit feature set / more data first)"
    else:
        ranking_edge, go_no_go = "NO", "NO-GO"
    high_state_edge = "YES" if (pd.notna(high_state_lift_10) and high_state_lift_10 > 1.0) else "NO"
    lead_time_quality = "GOOD" if (pd.notna(median_lead_time) and median_lead_time >= 30) else ("MARGINAL" if pd.notna(median_lead_time) else "UNKNOWN")

    reasoning = (
        f"Pooled out-of-fold top-10% directional lift = {top10_lift:.2f}x the resolved-conditional baseline "
        f"({baseline_rate:.1%} P(reversal|resolved)), top-20% = {top20_lift:.2f}x. Precision@K uses a DIFFERENT, "
        f"unconditional baseline ({baseline_rate_unconditional:.1%} P(reversal) over ALL scored candidates including "
        "NEUTRAL ones - a live screener does not know in advance which candidates will resolve). "
        f"{n_pass}/{len(criteria)} point-20 criteria passed. "
        f"{len(stable_features)}/{len(stability_df)} FEATURE_SET_V1 coefficients kept a consistent sign across all "
        f"temporal folds; {len(unstable_features)} did not ({', '.join(unstable_features) if unstable_features else 'none'})."
    )

    (out_dir / "phase_3b_logistic_summary.md").write_text(
        reports_3b.render_phase_3b_summary(ranking_edge, high_state_edge, lead_time_quality, go_no_go, stable_features, unstable_features, criteria, reasoning),
        encoding="utf-8",
    )

    typer.echo(f"RANKING EDGE: {ranking_edge} | HIGH-STATE EDGE: {high_state_edge} | LEAD-TIME: {lead_time_quality} | {go_no_go}")
    typer.echo(f"Done. 10 Phase 3B deliverables under: {out_dir}")


def _fmt_pct_local(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


if __name__ == "__main__":
    app()
