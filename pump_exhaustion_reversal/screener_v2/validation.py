"""Phase 3A points 3, 24, 25, 27-28 (the Phase-3B-oriented pieces built now
because Phase 3A's own effect-size/stability analysis already needs them):
event-level (never snapshot-row) cluster bootstrap, and temporal/event-level
splitting. No random row split anywhere in this package - a snapshot from
one event must never end up split across two different statistical buckets
that are then treated as independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CLUSTERS_FOR_BOOTSTRAP = 5


def event_weights(snapshots_df: pd.DataFrame, event_col: str = "event_id") -> pd.Series:
    """Point 3: sum(snapshot_weights within event) == 1 for every event - a
    pump that produced 100 snapshots gets the same total statistical weight
    as one that produced 10, so long events cannot dominate a pooled
    estimate just by existing longer. Weight = 1 / (snapshots in that
    event)."""
    counts = snapshots_df.groupby(event_col)[event_col].transform("size")
    return 1.0 / counts


def cluster_bootstrap_proportion(
    values: pd.Series, cluster_ids: pd.Series, n_resamples: int = 2000, ci: float = 0.95, seed: int = 7
) -> dict:
    """Event-cluster (or symbol-cluster) bootstrap for a proportion (e.g. P
    (REVERSAL)) - resamples CLUSTERS with replacement (not individual
    snapshot rows), pools every resampled cluster's rows, and reports the
    resulting distribution of the pooled mean. This is the "minimum
    cluster-by-event" bootstrap point 3/24 asks for; pass symbol as
    cluster_ids for the point-3 "additionally cluster-by-symbol" check."""
    df = pd.DataFrame({"value": values.to_numpy(), "cluster": cluster_ids.to_numpy()}).dropna(subset=["value"])
    clusters = df["cluster"].unique()
    if len(clusters) < MIN_CLUSTERS_FOR_BOOTSTRAP:
        return {"n_rows": len(df), "n_clusters": len(clusters), "statistic": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    by_cluster_values = {c: df.loc[df["cluster"] == c, "value"].to_numpy() for c in clusters}
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_resamples):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        pooled = np.concatenate([by_cluster_values[c] for c in drawn])
        if len(pooled):
            boot_means.append(np.mean(pooled))
    if not boot_means:
        return {"n_rows": len(df), "n_clusters": len(clusters), "statistic": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return {
        "n_rows": len(df), "n_clusters": len(clusters), "statistic": float(df["value"].mean()),
        "ci_low": float(lo), "ci_high": float(hi), "ci_level": ci,
    }


def compute_drawdown_from_high_pct(snapshots_df: pd.DataFrame, price_col: str = "current_price", high_col: str = "event_high_so_far") -> pd.Series:
    """Phase 3A.1 point 5 ingredient: how far the CURRENT snapshot already
    sits below the event's own running high - both values already causal
    (event_high_so_far is a cummax up to and including this bar). 0.0 = at
    or above the high, rises as price sits further below it."""
    return (snapshots_df[high_col] - snapshots_df[price_col]) / snapshots_df[high_col]


def filter_pre_reversal_censored(snapshots_df: pd.DataFrame, max_drawdown_from_high_pct: float, price_col: str = "current_price", high_col: str = "event_high_so_far") -> pd.DataFrame:
    """Point 5: restricts to snapshots where price is STILL close to the
    event's running high (drawdown_from_high <= max_drawdown_from_high_pct) -
    tests whether a feature survives BEFORE any real reversal has visibly
    started, not just after. The caller runs this at 2%/3%/5% and reports
    all three side by side - this function deliberately does not pick a
    "best" threshold."""
    drawdown = compute_drawdown_from_high_pct(snapshots_df, price_col, high_col)
    return snapshots_df[drawdown <= max_drawdown_from_high_pct]


def compute_price_position_in_range_pct(snapshots_df: pd.DataFrame, price_col: str = "current_price", high_col: str = "event_high_so_far", low_col: str = "event_low_so_far") -> pd.Series:
    """Where the current price sits within the event's own [low_so_far,
    high_so_far] range so far (both already causal, low_so_far is a running
    cummin) - 1.0 = at the running high, 0.0 = at the running low."""
    rng = snapshots_df[high_col] - snapshots_df[low_col]
    position = (snapshots_df[price_col] - snapshots_df[low_col]) / rng.replace(0, np.nan)
    return position


def filter_high_state_cohort(snapshots_df: pd.DataFrame, min_position_in_range_pct: float = 0.9, price_col: str = "current_price", high_col: str = "event_high_so_far", low_col: str = "event_low_so_far") -> pd.DataFrame:
    """Point 6: restricts to snapshots where price is within the top
    (1 - min_position_in_range_pct) of the event's own range so far - the
    causal proxy for "still near the highs". If a feature's ranking edge
    survives on JUST this cohort, that is much stronger evidence than a lift
    that only shows up after price has already fallen a long way."""
    position = compute_price_position_in_range_pct(snapshots_df, price_col, high_col, low_col)
    return snapshots_df[position >= min_position_in_range_pct]


def split_chronological_by_event(events_df: pd.DataFrame, event_col: str = "event_id", ts_col: str = "pump_detected_ts", fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, set]:
    """Point 27: chronological walk-forward split by EVENT (never by
    snapshot row) - earliest `fractions[0]` events -> train, next
    `fractions[1]` -> validation, latest `fractions[2]` -> test. Every
    snapshot belonging to one event_id always lands in exactly one split.
    Phase 3A itself doesn't train anything, but feature_stability's
    "year/period" check reuses this to compare early vs late sub-periods
    without ever letting one event straddle two periods."""
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1.0"
    unique_events = events_df[[event_col, ts_col]].drop_duplicates(subset=event_col).sort_values(ts_col)
    n = len(unique_events)
    train_end = int(round(n * fractions[0]))
    val_end = train_end + int(round(n * fractions[1]))
    ids = unique_events[event_col].tolist()
    return {"train": set(ids[:train_end]), "validation": set(ids[train_end:val_end]), "test": set(ids[val_end:])}
