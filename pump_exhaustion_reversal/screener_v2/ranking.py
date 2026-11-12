"""Phase 3A points 23-25 - univariate/bivariate feature study BEFORE any
model: decile bucketing (does P(REVERSAL) rise monotonically across
deciles?), effect size (Cliff's delta + reversal-lift/continuation-
suppression + event-cluster bootstrap CI, never picking a feature just for
having the single largest observed lift), and per-slice stability.

Every function respects point 3's event weighting (validation.event_weights)
and never lets a single long event dominate a decile's outcome rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..research import stats as research_stats
from . import validation

OUTCOME_VALUES: tuple[str, ...] = ("REVERSAL", "CONTINUATION", "NEUTRAL")


def decile_analysis(snapshots_df: pd.DataFrame, feature_col: str, outcome_col: str, n_bins: int = 5, event_col: str = "event_id") -> pd.DataFrame:
    """Point 23: bins `feature_col` into n_bins quantile buckets, reports
    event-weighted P(REVERSAL)/P(CONTINUATION)/P(NEUTRAL) per bin (spec's
    own worked example uses quintiles - n_bins=5 is the default, pass 10 for
    true deciles). We are looking for a MONOTONIC trend across bins, not a
    single winning bucket."""
    df = snapshots_df.dropna(subset=[feature_col, outcome_col]).copy()
    if df.empty or df[feature_col].nunique() < n_bins:
        return pd.DataFrame(columns=["bin", "bin_order", "n_snapshots", "n_events"] + [f"p_{o.lower()}" for o in OUTCOME_VALUES])
    try:
        df["bin"] = pd.qcut(df[feature_col], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    weights = validation.event_weights(df, event_col) if event_col in df.columns else pd.Series(1.0, index=df.index)

    rows = []
    for order, (bin_label, group) in enumerate(df.groupby("bin", observed=True)):
        w = weights.loc[group.index]
        total_w = w.sum()
        n_events = group[event_col].nunique() if event_col in group.columns else len(group)
        row = {"bin": str(bin_label), "bin_order": order, "n_snapshots": len(group), "n_events": n_events}
        for outcome_value in OUTCOME_VALUES:
            indicator = (group[outcome_col] == outcome_value).astype(float)
            row[f"p_{outcome_value.lower()}"] = float((w * indicator).sum() / total_w) if total_w else np.nan

        # Phase 3A.1 points 1-2: RESOLUTION power (leaving NEUTRAL at all -
        # already visible above via p_neutral) vs DIRECTION power (among
        # RESOLVED snapshots only, does the feature favor REVERSAL over
        # CONTINUATION) - computed here side by side, never conflated into
        # one number. A feature whose raw p_reversal rises purely because
        # p_neutral falls (both outcomes becoming more likely) will show a
        # FLAT reversal_to_continuation_ratio despite a rising p_reversal -
        # that is the whole point of this column.
        resolved_group = group[group[outcome_col].isin(("REVERSAL", "CONTINUATION"))]
        w_resolved = weights.loc[resolved_group.index]
        total_w_resolved = w_resolved.sum()
        row["n_resolved"] = len(resolved_group)
        if total_w_resolved:
            reversal_indicator = (resolved_group[outcome_col] == "REVERSAL").astype(float)
            p_rev_given_resolved = float((w_resolved * reversal_indicator).sum() / total_w_resolved)
            row["p_reversal_given_resolved"] = p_rev_given_resolved
            row["p_continuation_given_resolved"] = 1.0 - p_rev_given_resolved
            row["reversal_to_continuation_ratio"] = p_rev_given_resolved / (1.0 - p_rev_given_resolved) if p_rev_given_resolved < 1.0 else np.inf
        else:
            row["p_reversal_given_resolved"] = np.nan
            row["p_continuation_given_resolved"] = np.nan
            row["reversal_to_continuation_ratio"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def effect_size_report(snapshots_df: pd.DataFrame, feature_col: str, outcome_col: str, event_col: str = "event_id") -> dict:
    """Point 24: Cliff's delta + Mann-Whitney (feature-value distribution,
    REVERSAL snapshots vs CONTINUATION snapshots), reversal-lift and
    continuation-suppression (top vs bottom tercile of the feature), and an
    event-cluster bootstrap CI on the top tercile's reversal rate - never
    selects a feature by raw observed lift alone."""
    df = snapshots_df.dropna(subset=[feature_col, outcome_col])
    if df.empty or df[outcome_col].nunique() < 2:
        return {"feature": feature_col, "n": len(df), "valid": False}

    reversal_values = df.loc[df[outcome_col] == "REVERSAL", feature_col]
    continuation_values = df.loc[df[outcome_col] == "CONTINUATION", feature_col]
    cliffs_d = research_stats.cliffs_delta(reversal_values, continuation_values)
    mw = research_stats.mann_whitney(reversal_values, continuation_values)

    result = {
        "feature": feature_col, "n": len(df), "valid": True, "cliffs_delta": cliffs_d, "mann_whitney_p": mw["p_value"],
        "reversal_lift_top_vs_bottom_tercile": np.nan, "continuation_suppression_top_vs_bottom_tercile": np.nan,
        "top_tercile_reversal_rate": np.nan, "top_tercile_reversal_rate_ci_low": np.nan, "top_tercile_reversal_rate_ci_high": np.nan,
    }
    try:
        tercile = pd.qcut(df[feature_col], q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return result
    if "high" not in tercile.cat.categories or "low" not in tercile.cat.categories:
        return result

    is_reversal = (df[outcome_col] == "REVERSAL").astype(float)
    is_continuation = (df[outcome_col] == "CONTINUATION").astype(float)
    top_mask, bottom_mask = (tercile == "high").to_numpy(), (tercile == "low").to_numpy()
    top_reversal_rate, bottom_reversal_rate = is_reversal[top_mask].mean(), is_reversal[bottom_mask].mean()
    top_cont_rate, bottom_cont_rate = is_continuation[top_mask].mean(), is_continuation[bottom_mask].mean()
    result["reversal_lift_top_vs_bottom_tercile"] = float(top_reversal_rate / bottom_reversal_rate) if bottom_reversal_rate else np.nan
    result["continuation_suppression_top_vs_bottom_tercile"] = float(top_cont_rate / bottom_cont_rate) if bottom_cont_rate else np.nan

    if event_col in df.columns:
        top_bootstrap = validation.cluster_bootstrap_proportion(is_reversal[top_mask], df.loc[top_mask, event_col])
        result["top_tercile_reversal_rate"] = top_bootstrap["statistic"]
        result["top_tercile_reversal_rate_ci_low"] = top_bootstrap["ci_low"]
        result["top_tercile_reversal_rate_ci_high"] = top_bootstrap["ci_high"]
    return result


def resolution_effect_size_report(snapshots_df: pd.DataFrame, feature_col: str, outcome_col: str) -> dict:
    """Phase 3A.1 point 2, RESOLUTION power half: does feature_col separate
    RESOLVED snapshots (REVERSAL or CONTINUATION - a big move happened) from
    NEUTRAL ones (nothing happened) - regardless of which direction the move
    went. Deliberately never touches REVERSAL-vs-CONTINUATION - that
    question belongs to directional_effect_size_report only, so the two
    powers can never be silently averaged into one number."""
    df = snapshots_df.dropna(subset=[feature_col, outcome_col])
    if df.empty:
        return {"feature": feature_col, "n": len(df), "valid": False}
    resolved_values = df.loc[df[outcome_col] != "NEUTRAL", feature_col]
    neutral_values = df.loc[df[outcome_col] == "NEUTRAL", feature_col]
    if len(resolved_values) == 0 or len(neutral_values) == 0:
        return {"feature": feature_col, "n": len(df), "valid": False}
    cliffs_d = research_stats.cliffs_delta(resolved_values, neutral_values)
    mw = research_stats.mann_whitney(resolved_values, neutral_values)
    return {
        "feature": feature_col, "n": len(df), "valid": True,
        "n_resolved": len(resolved_values), "n_neutral": len(neutral_values),
        "resolution_cliffs_delta": cliffs_d, "resolution_mann_whitney_p": mw["p_value"],
    }


def directional_effect_size_report(snapshots_df: pd.DataFrame, feature_col: str, outcome_col: str, event_col: str = "event_id") -> dict:
    """Phase 3A.1 points 1-3, DIRECTION power half: restricts to RESOLVED
    snapshots FIRST (excludes NEUTRAL entirely from both the tercile split
    and the baseline), then asks whether feature_col favors REVERSAL over
    CONTINUATION. This is what point 3 says re-ranking must use instead of
    effect_size_report's raw reversal_lift_top_vs_bottom_tercile (which
    computes reversal rate against the FULL population including NEUTRAL,
    so a feature that only raises P(leaving NEUTRAL) - without favoring
    either resolved outcome - can still show a large raw lift there).
    `directional_lift_sign` is a +1/-1/0 summary of
    directional_lift_top_vs_bottom_tercile relative to 1.0, kept for
    is_directionally_stable's cross-slice agreement check (mirrors
    effect_size_report's cliffs_delta sign used by is_stable)."""
    resolved = snapshots_df[snapshots_df[outcome_col].isin(("REVERSAL", "CONTINUATION"))]
    df = resolved.dropna(subset=[feature_col])
    if df.empty or df[outcome_col].nunique() < 2:
        return {"feature": feature_col, "n_resolved": len(df), "valid": False}

    is_reversal = (df[outcome_col] == "REVERSAL").astype(float)
    reversal_values = df.loc[df[outcome_col] == "REVERSAL", feature_col]
    continuation_values = df.loc[df[outcome_col] == "CONTINUATION", feature_col]
    cliffs_d = research_stats.cliffs_delta(reversal_values, continuation_values)
    mw = research_stats.mann_whitney(reversal_values, continuation_values)

    result = {
        "feature": feature_col, "n_resolved": len(df), "valid": True,
        "cliffs_delta": cliffs_d, "mann_whitney_p": mw["p_value"],
        "overall_reversal_share_among_resolved": float(is_reversal.mean()),
        "directional_lift_top_vs_bottom_tercile": np.nan, "continuation_suppression_top_vs_bottom_tercile": np.nan,
        "top_tercile_reversal_share": np.nan, "top_tercile_reversal_share_ci_low": np.nan, "top_tercile_reversal_share_ci_high": np.nan,
        "directional_lift_sign": 0, "monotonic": None,
    }
    try:
        tercile = pd.qcut(df[feature_col], q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return result
    if "high" not in tercile.cat.categories or "low" not in tercile.cat.categories:
        return result

    top_mask, bottom_mask = (tercile == "high").to_numpy(), (tercile == "low").to_numpy()
    top_reversal_share, bottom_reversal_share = is_reversal[top_mask].mean(), is_reversal[bottom_mask].mean()
    lift = float(top_reversal_share / bottom_reversal_share) if bottom_reversal_share else np.nan
    result["directional_lift_top_vs_bottom_tercile"] = lift
    result["continuation_suppression_top_vs_bottom_tercile"] = float((1 - top_reversal_share) / (1 - bottom_reversal_share)) if (1 - bottom_reversal_share) else np.nan
    if pd.notna(lift):
        result["directional_lift_sign"] = 1 if lift > 1.0 else (-1 if lift < 1.0 else 0)

    if event_col in df.columns:
        top_bootstrap = validation.cluster_bootstrap_proportion(is_reversal[top_mask], df.loc[top_mask, event_col])
        result["top_tercile_reversal_share"] = top_bootstrap["statistic"]
        result["top_tercile_reversal_share_ci_low"] = top_bootstrap["ci_low"]
        result["top_tercile_reversal_share_ci_high"] = top_bootstrap["ci_high"]

    try:
        quintile = pd.qcut(df[feature_col], q=5, labels=False, duplicates="drop")
        quintile_rates = is_reversal.groupby(quintile).mean()
        diffs = quintile_rates.diff().dropna()
        result["monotonic"] = bool((diffs >= 0).all() or (diffs <= 0).all()) if len(diffs) else None
    except ValueError:
        result["monotonic"] = None
    return result


def directional_feature_stability_report(snapshots_df: pd.DataFrame, feature_col: str, outcome_col: str, split_dims: tuple[str, ...], event_col: str = "event_id") -> pd.DataFrame:
    """Directional counterpart to feature_stability_report: re-runs
    directional_effect_size_report (RESOLVED-only) per slice, instead of
    effect_size_report (which mixes in NEUTRAL via the raw-population
    tercile rate)."""
    rows = []
    for dim in split_dims:
        if dim not in snapshots_df.columns:
            continue
        for value, group in snapshots_df.groupby(dim, observed=True):
            result = directional_effect_size_report(group, feature_col, outcome_col, event_col)
            result["dimension"] = dim
            result["value"] = value
            rows.append(result)
    return pd.DataFrame(rows)


def feature_stability_report(snapshots_df: pd.DataFrame, feature_col: str, outcome_col: str, split_dims: tuple[str, ...], event_col: str = "event_id") -> pd.DataFrame:
    """Point 25: re-runs effect_size_report per slice of each dimension in
    split_dims (e.g. "period", "symbol_liquidity_bucket", "regime_trend",
    "exchange") - only dimensions that already exist as columns on
    snapshots_df are used (the caller merges those in first, e.g. via
    backtest.walk_forward.label_market_regime)."""
    rows = []
    for dim in split_dims:
        if dim not in snapshots_df.columns:
            continue
        for value, group in snapshots_df.groupby(dim, observed=True):
            result = effect_size_report(group, feature_col, outcome_col, event_col)
            result["dimension"] = dim
            result["value"] = value
            rows.append(result)
    return pd.DataFrame(rows)


def interaction_grid(snapshots_df: pd.DataFrame, dims: list[tuple[str, int | None]], outcome_col: str, event_col: str = "event_id", min_snapshots: int = 30) -> pd.DataFrame:
    """Phase 3A.1 point 11: a handful of PRE-SPECIFIED 2-3-dimension
    conditional-probability grids (e.g. oi_regime x volume-decay tercile) -
    never an exhaustive combinatorial search over all features; the caller
    passes a short fixed list of `dims` per named hypothesis. Each dim is
    (column, n_bins): n_bins=None uses the column's own existing categorical
    values as-is (e.g. oi_regime), an int quantile-bins that column via
    pd.qcut. Cells with fewer than min_snapshots are still reported but
    flagged low_n=True rather than hidden - a suppressed cell can look like
    "no signal" when it is really "no data"."""
    df = snapshots_df.dropna(subset=[outcome_col]).copy()
    bin_cols = []
    for col, n_bins in dims:
        if col not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=[col])
        if n_bins is None:
            df[f"_bin_{col}"] = df[col].astype(str)
        else:
            try:
                df[f"_bin_{col}"] = pd.qcut(df[col], q=n_bins, duplicates="drop")
            except ValueError:
                return pd.DataFrame()
        bin_cols.append(f"_bin_{col}")
    if df.empty:
        return pd.DataFrame()

    weights = validation.event_weights(df, event_col) if event_col in df.columns else pd.Series(1.0, index=df.index)
    rows = []
    for combo, group in df.groupby(bin_cols, observed=True):
        combo = combo if isinstance(combo, tuple) else (combo,)
        w = weights.loc[group.index]
        total_w = w.sum()
        resolved = group[group[outcome_col].isin(("REVERSAL", "CONTINUATION"))]
        w_resolved = weights.loc[resolved.index]
        total_w_resolved = w_resolved.sum()

        row = {dims[i][0]: str(combo[i]) for i in range(len(dims))}
        row["n_snapshots"] = len(group)
        row["n_resolved"] = len(resolved)
        row["low_n"] = len(group) < min_snapshots
        row["p_resolved"] = float(total_w_resolved / total_w) if total_w else np.nan
        if total_w_resolved:
            reversal_indicator = (resolved[outcome_col] == "REVERSAL").astype(float)
            row["p_reversal_given_resolved"] = float((w_resolved * reversal_indicator).sum() / total_w_resolved)
        else:
            row["p_reversal_given_resolved"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def lead_time_analysis(
    snapshots_df: pd.DataFrame, feature_col: str, outcome_col: str, bars_col: str, bar_interval_ms: int,
    top_quantile: float = 0.9, event_col: str = "event_id", pump_age_col: str = "pump_age_minutes",
) -> dict:
    """Phase 3A.1 point 17, scoped to what the snapshot table alone can
    honestly answer (no re-touch of raw OHLCV): for each event, finds the
    FIRST snapshot (by pump_age) where feature_col enters its GLOBAL top
    decile (threshold fixed from the pooled population BEFORE this per-event
    walk - never tuned per event or per feature after seeing results). At
    that first-entry snapshot, reads how the SAME-anchor competing-outcome
    label (outcome_col/bars_col) resolved: if REVERSAL,
    median_time_to_reversal_minutes is a genuine lead time (the label is
    forward-looking FROM this exact bar, so the signal necessarily fired
    before the reversal barrier was hit). fraction_already_deep_in_drawdown_
    at_signal is the share of first-entries where price had already fallen
    >=5% from the event's own running high by the time the signal first
    fired - a feature that only ever lights up once the drop is already
    well underway should be deprioritized even with a clean decile/effect-
    size profile, per the user's explicit caution."""
    df = snapshots_df.dropna(subset=[feature_col, event_col, pump_age_col]).copy()
    if df.empty:
        return {"feature": feature_col, "n_events_with_signal": 0, "valid": False}
    threshold = df[feature_col].quantile(top_quantile)
    signal_rows = df[df[feature_col] >= threshold]
    if signal_rows.empty:
        return {"feature": feature_col, "n_events_with_signal": 0, "valid": False, "threshold": float(threshold)}

    first_entries = signal_rows.sort_values(pump_age_col).groupby(event_col, as_index=False).first()
    n_events = len(first_entries)

    reversal_rows = first_entries[first_entries[outcome_col] == "REVERSAL"] if outcome_col in first_entries.columns else first_entries.iloc[0:0]
    if bars_col in first_entries.columns and len(reversal_rows):
        time_to_reversal_minutes = reversal_rows[bars_col] * bar_interval_ms / 60_000.0
    else:
        time_to_reversal_minutes = pd.Series(dtype=float)

    if "current_price" in first_entries.columns and "event_high_so_far" in first_entries.columns:
        drawdown = (first_entries["event_high_so_far"] - first_entries["current_price"]) / first_entries["event_high_so_far"].replace(0, np.nan)
        already_deep = drawdown.ge(0.05).mean()
    else:
        already_deep = np.nan

    return {
        "feature": feature_col, "valid": True, "threshold": float(threshold),
        "n_events_with_signal": n_events,
        "n_first_entries_resolved_reversal": len(reversal_rows),
        "fraction_first_entries_resolved_reversal": float(len(reversal_rows) / n_events) if n_events else np.nan,
        "median_time_to_reversal_minutes": float(time_to_reversal_minutes.median()) if len(time_to_reversal_minutes) else np.nan,
        "fraction_already_deep_in_drawdown_at_signal": float(already_deep) if pd.notna(already_deep) else np.nan,
    }


def classify_feature_verdict(
    directional_effect: dict, directional_stability: dict, high_state_effect: dict | None = None,
    min_abs_cliffs_delta: float = 0.10, min_high_state_abs_cliffs_delta: float = 0.07,
) -> dict:
    """Phase 3A.1 point 18: KEEP/WEAK/REJECT synthesis from already-computed
    pieces (directional_effect_size_report, is_directionally_stable's
    output, and optionally a directional_effect_size_report run on the
    high-state/pre-reversal-censored cohort). A pure decision function over
    its inputs - it runs no analysis itself, so it is trivially testable
    against hand-built dicts and stays honest about exactly which evidence
    drove each verdict.

    KEEP: has real directional lift (not just resolution lift, i.e.
    |Cliff's delta| among RESOLVED snapshots clears min_abs_cliffs_delta),
    is directionally stable across the caller's slices (>=70% same sign,
    per is_directionally_stable), AND - when a high-state cohort result is
    supplied - the SAME sign survives there too (a feature that only shows
    up after price has already fallen a long way is exactly what point 17's
    lead-time caution warns against, even if its pooled numbers look
    clean). WEAK: has directional lift but fails exactly one of stability /
    high-state survival. REJECT: no real directional lift at all, or fails
    both."""
    if not directional_effect.get("valid") or pd.isna(directional_effect.get("cliffs_delta")):
        return {"verdict": "REJECT", "reason": "no valid directional signal (resolution-power-only, or insufficient resolved data)"}

    direction_delta = directional_effect["cliffs_delta"]
    if abs(direction_delta) < min_abs_cliffs_delta:
        return {"verdict": "REJECT", "reason": f"directional Cliff's delta {direction_delta:.3f} below the {min_abs_cliffs_delta} floor - looks like resolution power only"}

    stable = bool(directional_stability.get("stable"))
    pooled_sign = np.sign(direction_delta)

    high_state_ok = True
    high_state_note = "not tested"
    if high_state_effect is not None:
        if not high_state_effect.get("valid") or pd.isna(high_state_effect.get("cliffs_delta")):
            high_state_ok = False
            high_state_note = "no valid signal in the high-state/pre-reversal cohort"
        else:
            hs_delta = high_state_effect["cliffs_delta"]
            same_sign = bool(np.sign(hs_delta) == pooled_sign)
            survives = abs(hs_delta) >= min_high_state_abs_cliffs_delta
            high_state_ok = same_sign and survives
            high_state_note = f"cliffs_delta={hs_delta:.3f} ({'same' if same_sign else 'FLIPPED'} sign, {'survives' if survives else 'too weak'})"

    if stable and high_state_ok:
        return {"verdict": "KEEP", "reason": f"directional (delta={direction_delta:.3f}), stable across slices, high-state: {high_state_note}"}
    if stable or high_state_ok:
        missing = "high-state cohort" if not high_state_ok else "cross-slice stability"
        return {"verdict": "WEAK", "reason": f"directional (delta={direction_delta:.3f}) but fails {missing} (high-state: {high_state_note}, stable={stable})"}
    return {"verdict": "REJECT", "reason": f"directional signal present (delta={direction_delta:.3f}) but neither stable nor survives the high-state cohort (high-state: {high_state_note})"}


def is_stable(stability_df: pd.DataFrame, min_fraction_same_direction: float = 0.7) -> dict:
    """A minimal, disclosed 'stable' criterion (point 25's "хотя бы
    направленно" - at least directionally): the sign of cliffs_delta agrees
    across at least min_fraction_same_direction of the valid slices."""
    valid = stability_df[stability_df.get("valid", pd.Series(dtype=bool)) == True] if "valid" in stability_df.columns else stability_df
    valid = valid.dropna(subset=["cliffs_delta"])
    if valid.empty:
        return {"stable": False, "n_slices": 0, "fraction_same_direction": np.nan}
    overall_sign = np.sign(valid["cliffs_delta"].median())
    same_direction = (np.sign(valid["cliffs_delta"]) == overall_sign).mean()
    return {"stable": bool(same_direction >= min_fraction_same_direction), "n_slices": len(valid), "fraction_same_direction": float(same_direction), "overall_sign": int(overall_sign)}


def is_directionally_stable(directional_stability_df: pd.DataFrame, min_fraction_same_direction: float = 0.7) -> dict:
    """Same >=70% cross-slice agreement criterion as is_stable, but applied
    to directional_feature_stability_report's directional_lift_sign instead
    of cliffs_delta - a feature can pass is_stable (resolution-contaminated)
    while failing this (no real directional agreement), and that gap is
    exactly what Phase 3A.1 point 3 asks the re-ranking to expose."""
    valid = directional_stability_df[directional_stability_df.get("valid", pd.Series(dtype=bool)) == True] if "valid" in directional_stability_df.columns else directional_stability_df
    valid = valid[valid.get("directional_lift_sign", pd.Series(dtype=float)) != 0] if "directional_lift_sign" in valid.columns else valid.iloc[0:0]
    if valid.empty:
        return {"stable": False, "n_slices": 0, "fraction_same_direction": np.nan}
    overall_sign = np.sign(valid["directional_lift_sign"].median())
    if overall_sign == 0:
        return {"stable": False, "n_slices": len(valid), "fraction_same_direction": np.nan, "overall_sign": 0}
    same_direction = (np.sign(valid["directional_lift_sign"]) == overall_sign).mean()
    return {"stable": bool(same_direction >= min_fraction_same_direction), "n_slices": len(valid), "fraction_same_direction": float(same_direction), "overall_sign": int(overall_sign)}
