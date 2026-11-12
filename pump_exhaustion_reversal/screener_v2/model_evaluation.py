"""Phase 3B points 11-14, 21: evaluate an already-computed model SCORE
column against outcomes - ranking (decile curve), the screener-simulation
metric (Precision@K), and calibration. Bootstrap CI reuses
validation.cluster_bootstrap_proportion directly (point 22); lead-time
reuses ranking.lead_time_analysis directly with the score column standing
in for a raw feature (point 14) - neither is re-implemented here.

Every function takes an already-SCORED DataFrame (a score_col already
computed by logistic_model.predict_proba on the caller's chosen
train/validation split) - nothing here fits or trains anything, so a
function here can never see fitting-time data by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_RESOLVED_OUTCOMES = ("REVERSAL", "CONTINUATION")


def score_decile_curve(
    scored_df: pd.DataFrame, score_col: str, outcome_col: str, event_col: str = "event_id",
    symbol_col: str = "symbol", top_fractions: tuple[float, ...] = (0.5, 0.3, 0.2, 0.1, 0.05),
) -> pd.DataFrame:
    """Point 11: for each top-X% threshold (ranked by score_col over ALL
    scored snapshots, not just resolved ones - a real screener does not
    know in advance which candidates will resolve), the RESOLVED subset
    that ranks in the top X% - P(reversal|resolved), P(continuation|
    resolved), directional lift vs the overall resolved baseline, N events,
    N unique symbols. The FIRST row (top_pct=1.0) is that overall baseline,
    included so the curve can be read as one monotonic table rather than
    needing the baseline supplied separately."""
    df = scored_df.dropna(subset=[score_col, outcome_col])
    resolved_all = df[df[outcome_col].isin(_RESOLVED_OUTCOMES)]
    baseline_rate = float((resolved_all[outcome_col] == "REVERSAL").mean()) if len(resolved_all) else np.nan

    rows = [
        {
            "top_pct": 1.0, "score_threshold": float(df[score_col].min()) if len(df) else np.nan,
            "n_snapshots": len(df), "n_resolved": len(resolved_all),
            "n_events": int(df[event_col].nunique()) if event_col in df.columns else np.nan,
            "n_unique_symbols": int(df[symbol_col].nunique()) if symbol_col in df.columns else np.nan,
            "p_reversal_given_resolved": baseline_rate,
            "p_continuation_given_resolved": (1 - baseline_rate) if pd.notna(baseline_rate) else np.nan,
            "directional_lift": 1.0 if pd.notna(baseline_rate) else np.nan,
        }
    ]
    for frac in top_fractions:
        threshold = df[score_col].quantile(1 - frac)
        top = df[df[score_col] >= threshold]
        top_resolved = top[top[outcome_col].isin(_RESOLVED_OUTCOMES)]
        p_rev = float((top_resolved[outcome_col] == "REVERSAL").mean()) if len(top_resolved) else np.nan
        rows.append(
            {
                "top_pct": frac, "score_threshold": float(threshold),
                "n_snapshots": len(top), "n_resolved": len(top_resolved),
                "n_events": int(top[event_col].nunique()) if event_col in top.columns else np.nan,
                "n_unique_symbols": int(top[symbol_col].nunique()) if symbol_col in top.columns else np.nan,
                "p_reversal_given_resolved": p_rev,
                "p_continuation_given_resolved": (1 - p_rev) if pd.notna(p_rev) else np.nan,
                "directional_lift": (p_rev / baseline_rate) if baseline_rate else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ranking_by_drawdown_bucket(
    scored_df: pd.DataFrame, score_col: str, outcome_col: str, drawdown_col: str,
    buckets: list[tuple[str, float, float | None]], event_col: str = "event_id", symbol_col: str = "symbol",
    top_fractions: tuple[float, ...] = (0.2, 0.1, 0.05),
) -> pd.DataFrame:
    """Phase 3B.1 points 4-5: for each drawdown-from-event-high bucket
    (label, low_pct inclusive, high_pct exclusive - None means no upper
    bound), restricts to snapshots in that bucket and reports the same
    decile-curve shape as score_decile_curve. The top-X% threshold is
    computed WITHIN the bucket's own scored population, not the global
    population - a live screener ranking "still near the highs" candidates
    only ever compares them against other still-near-the-highs candidates
    at that moment, never against ones that already dumped. This is what
    answers whether the model is an EARLY-WARNING screener (edge already
    present at low drawdown_pct) or a CONFIRMATION screener (edge only
    appears once drawdown_pct is already large)."""
    rows = []
    for label, low, high in buckets:
        if high is None:
            bucket_df = scored_df[scored_df[drawdown_col] >= low]
        else:
            bucket_df = scored_df[(scored_df[drawdown_col] >= low) & (scored_df[drawdown_col] < high)]
        curve = score_decile_curve(bucket_df, score_col, outcome_col, event_col, symbol_col, top_fractions)
        curve.insert(0, "bucket", label)
        rows.append(curve)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def precision_at_k(
    scored_df: pd.DataFrame, score_col: str, outcome_col: str, event_col: str = "event_id",
    timestamp_col: str = "timestamp", ks: tuple[int, ...] = (1, 3, 5), min_concurrent: int = 2,
) -> dict:
    """Point 13: simulates the screener's actual job - at each timestamp,
    among pump events that ALL have an active scored snapshot at that exact
    bar (genuinely concurrent, not just "active somewhere that day"), rank
    by score_col and check whether the top-K picks go on to REVERSAL.
    Timestamps with fewer than `min_concurrent` simultaneously-scored
    candidates are skipped (nothing to rank). An "alert" is counted once
    per (event, K) the FIRST time that event enters the top-K - a candidate
    sitting in the top-3 for 20 consecutive bars is one real screener
    opportunity, not 20; `mean_precision_at_k_per_timestamp` instead
    reports the bar-level rate directly, since that is exactly what
    Precision@K conventionally means. A "false alert" is a first-alert
    whose eventual outcome was NOT REVERSAL (CONTINUATION or NEUTRAL both
    count as a miss - the screener promised THIS specific candidate)."""
    df = scored_df.dropna(subset=[score_col, timestamp_col, outcome_col, event_col]).sort_values(timestamp_col)
    per_k_bar_hits: dict[int, list[float]] = {k: [] for k in ks}
    already_alerted: dict[int, set] = {k: set() for k in ks}
    alert_outcomes: dict[int, list[str]] = {k: [] for k in ks}
    n_timestamps_used = 0

    for ts, group in df.groupby(timestamp_col):
        if len(group) < min_concurrent:
            continue
        n_timestamps_used += 1
        ranked = group.sort_values(score_col, ascending=False)
        for k in ks:
            top_k = ranked.head(k)
            per_k_bar_hits[k].append(float((top_k[outcome_col] == "REVERSAL").mean()))
            for eid, outcome in zip(top_k[event_col], top_k[outcome_col]):
                if eid not in already_alerted[k]:
                    already_alerted[k].add(eid)
                    alert_outcomes[k].append(outcome)

    ts_min, ts_max = (df[timestamp_col].min(), df[timestamp_col].max()) if len(df) else (0, 0)
    n_days = max((ts_max - ts_min) / 86_400_000.0, 1e-9)

    by_k = {}
    for k in ks:
        outcomes = alert_outcomes[k]
        n_alerts = len(outcomes)
        n_true = sum(1 for o in outcomes if o == "REVERSAL")
        by_k[k] = {
            "mean_precision_at_k_per_timestamp": float(np.mean(per_k_bar_hits[k])) if per_k_bar_hits[k] else np.nan,
            "n_unique_events_alerted": n_alerts,
            "n_true_alerts": n_true,
            "n_false_alerts": n_alerts - n_true,
            "alerts_per_day": n_alerts / n_days,
            "false_alerts_per_day": (n_alerts - n_true) / n_days,
            "precision_among_unique_alerts": (n_true / n_alerts) if n_alerts else np.nan,
        }
    return {"n_timestamps_used": n_timestamps_used, "n_days_span": n_days, "by_k": by_k}


def classification_metrics(scored_df: pd.DataFrame, score_col: str, outcome_col: str, positive_label: str = "REVERSAL") -> dict:
    """Phase 3B.1 point 3: ROC-AUC, PR-AUC (average precision), and Brier
    score among RESOLVED snapshots - shown for completeness (point 12's
    "can show, not the primary criterion"), never the deciding metric for
    this screener. Implemented by hand (rank-based AUC via the Mann-Whitney
    U relation, precision-recall via a sorted cumulative sweep) rather than
    adding scikit-learn - same dependency discipline as logistic_model.py."""
    df = scored_df.dropna(subset=[score_col, outcome_col])
    resolved = df[df[outcome_col].isin(_RESOLVED_OUTCOMES)]
    if resolved.empty:
        return {"n_resolved": 0, "roc_auc": np.nan, "pr_auc": np.nan, "brier_score": np.nan}
    y_true = (resolved[outcome_col] == positive_label).to_numpy().astype(float)
    y_score = resolved[score_col].to_numpy(dtype=float)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())

    if n_pos == 0 or n_neg == 0:
        roc_auc = np.nan
    else:
        ranks = pd.Series(y_score).rank(method="average").to_numpy()
        sum_ranks_pos = ranks[y_true.astype(bool)].sum()
        roc_auc = float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    if n_pos == 0:
        pr_auc = np.nan
    else:
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        cum_tp = np.cumsum(y_sorted)
        precision = cum_tp / np.arange(1, len(y_sorted) + 1)
        recall = cum_tp / n_pos
        recall_prev = np.concatenate([[0.0], recall[:-1]])
        is_pos = y_sorted.astype(bool)
        pr_auc = float(np.sum(precision[is_pos] * (recall[is_pos] - recall_prev[is_pos])))

    brier = float(np.mean((y_score - y_true) ** 2))
    return {"n_resolved": len(resolved), "roc_auc": roc_auc, "pr_auc": pr_auc, "brier_score": brier}


def first_alert_per_event(
    scored_df: pd.DataFrame, score_col: str, event_col: str = "event_id", timestamp_col: str = "timestamp",
    top_quantile: float = 0.9, min_score_rows: int = 30,
) -> pd.DataFrame:
    """Phase 3B.1 point 7: one row per event - the FIRST snapshot (by
    timestamp) where score_col enters its GLOBAL top (1-top_quantile)
    fraction (threshold fixed from the pooled scored population, not
    per-event). Every subsequent snapshot of that same event, even if it
    stays in the top bucket for many more bars, is deliberately NOT a
    second "opportunity" - this is the live-like unit precision_at_k's
    unique-alert counting and the remaining-opportunity analysis both
    build on."""
    df = scored_df.dropna(subset=[score_col, event_col, timestamp_col])
    if len(df) < min_score_rows:
        return pd.DataFrame()
    threshold = df[score_col].quantile(top_quantile)
    signal_rows = df[df[score_col] >= threshold]
    if signal_rows.empty:
        return pd.DataFrame()
    first_entries = signal_rows.sort_values(timestamp_col).groupby(event_col, as_index=False).first()
    first_entries["score_threshold"] = float(threshold)
    return first_entries


def calibration_curve(scored_df: pd.DataFrame, score_col: str, outcome_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Point 21: predicted probability vs actual reversal rate, among
    RESOLVED snapshots only (a calibration claim about "P(reversal)" is
    only meaningful conditional on something having resolved - the model
    itself is fit on resolved-only rows, see logistic_model.fit_l2_logistic)."""
    df = scored_df.dropna(subset=[score_col, outcome_col])
    resolved = df[df[outcome_col].isin(_RESOLVED_OUTCOMES)].copy()
    if resolved.empty:
        return pd.DataFrame()
    try:
        resolved["_bin"] = pd.qcut(resolved[score_col], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for bin_label, group in resolved.groupby("_bin", observed=True):
        rows.append(
            {
                "bin": str(bin_label), "n": len(group),
                "mean_predicted_probability": float(group[score_col].mean()),
                "actual_reversal_rate": float((group[outcome_col] == "REVERSAL").mean()),
            }
        )
    return pd.DataFrame(rows)
