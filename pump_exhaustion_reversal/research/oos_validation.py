"""Phase 2D: OI_ROLLOVER + VOLUME_DECAY OOS Validation - reusable pipeline
pieces (matched-control study, symbol clustering / cluster bootstrap, regime
and pump-magnitude robustness splits, OI data-quality tiers, and a mechanical
pre-registered-acceptance-criteria check). Built to run against EITHER a
genuine held-out OOS sample (once Bybit accumulates enough forward history
past the freeze date - see config/oi_volume_reversal_candidate_v1.yaml) OR,
clearly labeled as a pipeline dry-run and never as a real OOS verdict, the
existing discovery sample - the two must never be conflated in a report.

Diagnostic-only, same lookahead-safety framing as the rest of research/*
(see research/__init__.py) - nothing here is imported by backtest/simulator.py
or strategy/*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord
from . import stats as research_stats

OI_QUALITY_HIGH_MIN_COVERAGE = 0.95
OI_QUALITY_MEDIUM_MIN_COVERAGE = 0.80


def oi_quality_tier(feature_frame: pd.DataFrame, start_idx: int, end_idx: int) -> str:
    """Phase 2D point 12: HIGH/MEDIUM/LOW/MISSING OI coverage over the
    window that mattered for this signal (climax to signal bar) - never
    collapses "OI missing" into "OI no rollover"; only HIGH/MEDIUM windows
    should count toward the primary OOS result (point 12's "не считать
    missing OI = no rollover")."""
    if "open_interest" not in feature_frame.columns:
        return "MISSING"
    window = feature_frame["open_interest"].iloc[max(0, start_idx) : end_idx + 1]
    if len(window) == 0:
        return "MISSING"
    coverage = float(window.notna().mean())
    if coverage >= OI_QUALITY_HIGH_MIN_COVERAGE:
        return "HIGH"
    if coverage >= OI_QUALITY_MEDIUM_MIN_COVERAGE:
        return "MEDIUM"
    if coverage > 0:
        return "LOW"
    return "MISSING"


def symbol_clustering_summary(candidates_df: pd.DataFrame, value_col: str = "net_r") -> dict:
    """Phase 2D point 9: N events, N unique symbols, max events per symbol,
    top-5 symbols' share of total PnL (sum of value_col) - N events is NOT
    N independent observations if one token fired repeatedly."""
    if candidates_df.empty:
        return {"n_events": 0, "n_unique_symbols": 0, "max_events_per_symbol": 0, "top5_symbol_pnl_share": np.nan}
    by_symbol = candidates_df.groupby("symbol")[value_col].agg(["count", "sum"])
    total_pnl = candidates_df[value_col].sum()
    top5_pnl = by_symbol["sum"].abs().sort_values(ascending=False).head(5).index
    top5_sum = by_symbol.loc[top5_pnl, "sum"].sum()
    return {
        "n_events": len(candidates_df),
        "n_unique_symbols": candidates_df["symbol"].nunique(),
        "max_events_per_symbol": int(by_symbol["count"].max()),
        "top5_symbol_pnl_share": float(top5_sum / total_pnl) if total_pnl else np.nan,
        "per_symbol_counts": by_symbol["count"].sort_values(ascending=False).to_dict(),
    }


def cluster_bootstrap_ci(candidates_df: pd.DataFrame, value_col: str = "net_r", n_resamples: int = 2000, ci: float = 0.95, seed: int = 7) -> dict:
    """Phase 2D point 9: bootstrap by resampling SYMBOLS with replacement
    (not individual events) - if the same token contributed several
    correlated events, an ordinary event-level bootstrap understates the
    true uncertainty. Each resample draws N_symbols symbols with
    replacement and pools ALL of that symbol's events; the statistic is the
    mean of value_col over the pooled resample."""
    if candidates_df.empty:
        return {"n_events": 0, "n_symbols": 0, "median_expectancy": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    symbols = candidates_df["symbol"].unique()
    if len(symbols) < research_stats.MIN_N_FOR_BOOTSTRAP:
        return {"n_events": len(candidates_df), "n_symbols": len(symbols), "median_expectancy": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    by_symbol_values = {s: candidates_df.loc[candidates_df["symbol"] == s, value_col].to_numpy() for s in symbols}
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_resamples):
        drawn = rng.choice(symbols, size=len(symbols), replace=True)
        pooled = np.concatenate([by_symbol_values[s] for s in drawn])
        pooled = pooled[np.isfinite(pooled)]
        if len(pooled):
            boot_means.append(np.mean(pooled))
    if not boot_means:
        return {"n_events": len(candidates_df), "n_symbols": len(symbols), "median_expectancy": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return {
        "n_events": len(candidates_df), "n_symbols": len(symbols), "median_expectancy": float(np.median(boot_means)),
        "ci_low": float(lo), "ci_high": float(hi), "ci_level": ci,
    }


def regime_breakdown(candidates_df: pd.DataFrame, events_with_regime: pd.DataFrame, value_col: str = "net_r") -> pd.DataFrame:
    """Phase 2D point 10: net expectancy split by regime_trend/regime_weekday/
    regime_session (from backtest.walk_forward.label_market_regime, reused
    unchanged - no new regime logic). Diagnostic only - never picks a "best"
    regime, only checks the result isn't concentrated in a single one."""
    if candidates_df.empty or events_with_regime.empty:
        return pd.DataFrame(columns=["dimension", "value", "n", "net_expectancy_r"])
    merged = candidates_df.merge(events_with_regime[["event_id", "regime_trend", "regime_weekday", "regime_session"]], on="event_id", how="left")
    rows = []
    for dimension in ("regime_trend", "regime_weekday", "regime_session"):
        for value, group in merged.groupby(dimension):
            rows.append({"dimension": dimension, "value": value, "n": len(group), "net_expectancy_r": float(group[value_col].mean())})
    return pd.DataFrame(rows)


def pump_magnitude_buckets(candidates_df: pd.DataFrame, magnitude_by_event: dict[str, float], value_col: str = "net_r") -> pd.DataFrame:
    """Phase 2D point 11: net expectancy split by pump-magnitude tercile
    (small/medium/extreme, by whatever pump-return metric the caller
    supplies per event_id - e.g. climax price / pump_start price - 1).
    Diagnostic only - detector thresholds are never touched here."""
    if candidates_df.empty or not magnitude_by_event:
        return pd.DataFrame(columns=["bucket", "n", "net_expectancy_r"])
    df = candidates_df.copy()
    df["pump_magnitude"] = df["event_id"].map(magnitude_by_event)
    df = df.dropna(subset=["pump_magnitude"])
    if df.empty:
        return pd.DataFrame(columns=["bucket", "n", "net_expectancy_r"])
    try:
        df["bucket"] = pd.qcut(df["pump_magnitude"], q=3, labels=["small", "medium", "extreme"], duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["bucket", "n", "net_expectancy_r"])
    rows = []
    for bucket, group in df.groupby("bucket", observed=True):
        rows.append({"bucket": bucket, "n": len(group), "net_expectancy_r": float(group[value_col].mean())})
    return pd.DataFrame(rows)


def match_controls(
    signal_event_ids: set[str], candidate_pool: pd.DataFrame, k: int = 3, feature_cols: tuple[str, ...] = ("pump_score", "volume_zscore", "btc_return_concurrent"),
) -> pd.DataFrame:
    """Phase 2D point 8: for each signal event, the K nearest pump events
    from candidate_pool (events WITHOUT the OI_ROLLOVER+VOLUME_DECAY signal)
    by standardized Euclidean distance over feature_cols - a lightweight
    matched-control set, not a full propensity-score model (small N doesn't
    support one). candidate_pool must already exclude the signal events
    themselves and carry `event_id` + feature_cols. Distance is computed on
    z-scored features (pooled mean/std across candidate_pool) so columns on
    different scales (e.g. pump_score ~0-100 vs btc_return ~-0.05-0.05)
    don't silently dominate the match."""
    pool = candidate_pool[~candidate_pool["event_id"].isin(signal_event_ids)].copy()
    usable_cols = [c for c in feature_cols if c in pool.columns]
    pool = pool.dropna(subset=usable_cols)
    if pool.empty or not usable_cols:
        return pd.DataFrame(columns=["signal_event_id", "control_event_id", "distance", "rank"])
    means, stds = pool[usable_cols].mean(), pool[usable_cols].std().replace(0, np.nan)
    pool_z = (pool[usable_cols] - means) / stds
    pool_z = pool_z.fillna(0.0)

    signal_rows = candidate_pool[candidate_pool["event_id"].isin(signal_event_ids)]
    rows = []
    for _, sig_row in signal_rows.iterrows():
        # sig_row comes from .iterrows() on a DataFrame with mixed dtypes
        # (event_id is a string) - a single row's Series is forced to object
        # dtype even after selecting only the numeric columns, which breaks
        # np.sqrt downstream ("float has no callable sqrt method"); cast
        # back to float explicitly.
        sig_features = sig_row[usable_cols].astype(float)
        sig_z = ((sig_features - means) / stds).fillna(0.0)
        dist = np.sqrt(((pool_z - sig_z) ** 2).sum(axis=1))
        nearest = dist.nsmallest(k)
        for rank, (idx, d) in enumerate(nearest.items(), start=1):
            rows.append({"signal_event_id": sig_row["event_id"], "control_event_id": pool.loc[idx, "event_id"], "distance": float(d), "rank": rank})
    return pd.DataFrame(rows)


def check_acceptance_criteria(
    summary: dict, cluster_bootstrap: dict, symbol_summary: dict, criteria: dict, cost_stress_positive: bool | None = None,
    regime_concentration_ok: bool | None = None,
) -> dict:
    """Phase 2D point 17: mechanical pass/fail against the criteria already
    frozen in config/oi_volume_reversal_candidate_v1.yaml's
    oos_acceptance_criteria block - never re-derives or loosens a threshold
    after seeing the result. Returns per-criterion booleans plus an overall
    outcome in {"A_CONFIRMED_ENOUGH_FOR_PAPER_TRADING", "B_WEAK_UNCERTAIN_WAIT_FOR_MORE_DATA",
    "C_REJECT_CLOSE_BRANCH", "INSUFFICIENT_SAMPLE"}."""
    n = summary.get("n", 0)
    if n < criteria.get("n_signals_preferred_minimum", 50):
        return {"outcome": "INSUFFICIENT_SAMPLE", "n": n, "n_required": criteria.get("n_signals_preferred_minimum", 50)}

    checks = {
        "net_expectancy_positive": summary.get("net_expectancy_r", np.nan) > criteria.get("net_expectancy_r_must_exceed", 0.0),
        "profit_factor_ok": summary.get("profit_factor", 0.0) > criteria.get("profit_factor_must_exceed", 1.15),
        "cluster_bootstrap_median_positive": cluster_bootstrap.get("median_expectancy", np.nan) > criteria.get("cluster_bootstrap_median_expectancy_must_exceed", 0.0),
        "symbol_concentration_ok": symbol_summary.get("top5_symbol_pnl_share", 1.0) <= criteria.get("max_single_symbol_pnl_share_pct", 25.0) / 100.0,
    }
    if cost_stress_positive is not None:
        checks["cost_stress_1_5x_positive"] = bool(cost_stress_positive)
    if regime_concentration_ok is not None:
        checks["regime_concentration_ok"] = bool(regime_concentration_ok)

    all_hard_checks_pass = checks["net_expectancy_positive"] and checks["profit_factor_ok"] and checks["cluster_bootstrap_median_positive"] and checks["symbol_concentration_ok"]
    if not all_hard_checks_pass:
        outcome = "C_REJECT_CLOSE_BRANCH" if summary.get("net_expectancy_r", 0.0) <= 0 or summary.get("profit_factor", 0.0) < 1.0 else "B_WEAK_UNCERTAIN_WAIT_FOR_MORE_DATA"
    else:
        soft_checks_pass = checks.get("cost_stress_1_5x_positive", True) and checks.get("regime_concentration_ok", True)
        outcome = "A_CONFIRMED_ENOUGH_FOR_PAPER_TRADING" if soft_checks_pass else "B_WEAK_UNCERTAIN_WAIT_FOR_MORE_DATA"

    return {"outcome": outcome, "n": n, **checks}
