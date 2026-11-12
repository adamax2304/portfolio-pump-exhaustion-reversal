"""Stage 2A Phase 2B point 6: continuous "quality" characteristics of a
failed-reclaim bar - not just the boolean failed_reclaim_lower_high flag.
Goal: do failed-reclaims whose first-passage outcome later resolves
favorably differ from the rest on these, before any of this is turned into
a score or a gate? Diagnostic-only, same lookahead-safety framing as the
rest of research/* (see research/__init__.py) - every feature here reads
only bars at or before reclaim_idx_abs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import avwap as avwap_feat
from . import stats as research_stats

QUALITY_FEATURE_NAMES: tuple[str, ...] = (
    "reclaim_depth_pct", "rejection_magnitude_pct", "wick_to_body_ratio", "volume_at_reclaim_ratio",
    "volume_contraction_from_climax", "cvd_at_reclaim", "cvd_slope_at_reclaim", "oi_change_climax_to_reclaim_pct",
    "distance_from_climax_pct", "time_climax_to_reclaim_bars", "avwap_distance_pct",
)


def _oi_change(feature_frame: pd.DataFrame, climax_idx_abs: int, reclaim_idx_abs: int) -> float:
    if "open_interest" not in feature_frame.columns:
        return np.nan
    oi_climax = feature_frame["open_interest"].iloc[climax_idx_abs]
    oi_reclaim = feature_frame["open_interest"].iloc[reclaim_idx_abs]
    if pd.isna(oi_climax) or pd.isna(oi_reclaim) or oi_climax == 0:
        return np.nan
    return float((oi_reclaim - oi_climax) / oi_climax)


def compute_reclaim_quality_features(
    ohlcv: pd.DataFrame, feature_frame: pd.DataFrame, climax_idx_abs: int, reclaim_idx_abs: int
) -> dict:
    """One failed-reclaim bar's continuous characteristics. climax_idx_abs/
    reclaim_idx_abs are absolute positions into `ohlcv` (already translated
    from wide/df_slice-local coordinates by the caller - see
    lifecycle_diagnostics._anchor_idx_for/_pump_start_idx_for). Returns {}
    (no partial dict) if the bars are out of range or misordered - callers
    should skip the event rather than half-fill a row."""
    if reclaim_idx_abs <= climax_idx_abs or reclaim_idx_abs >= len(ohlcv) or climax_idx_abs < 0:
        return {}
    climax_high = float(ohlcv["high"].iloc[climax_idx_abs])
    climax_volume = float(ohlcv["volume"].iloc[climax_idx_abs])
    reclaim_bar = ohlcv.iloc[reclaim_idx_abs]
    reclaim_high, reclaim_low, reclaim_close, reclaim_open = (
        float(reclaim_bar["high"]), float(reclaim_bar["low"]), float(reclaim_bar["close"]), float(reclaim_bar["open"]),
    )
    body = abs(reclaim_close - reclaim_open)
    upper_wick = reclaim_high - max(reclaim_close, reclaim_open)
    recent_volume_window = ohlcv["volume"].iloc[max(0, reclaim_idx_abs - 20) : reclaim_idx_abs]
    recent_volume_mean = float(recent_volume_window.mean()) if len(recent_volume_window) else np.nan

    avwap_series = avwap_feat.anchored_vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], climax_idx_abs)
    avwap_at_reclaim = float(avwap_series.iloc[reclaim_idx_abs]) if pd.notna(avwap_series.iloc[reclaim_idx_abs]) else np.nan

    cvd_at_reclaim = np.nan
    cvd_slope_at_reclaim = np.nan
    if reclaim_idx_abs < len(feature_frame):
        cvd_row = feature_frame.iloc[reclaim_idx_abs]
        if pd.notna(cvd_row.get("cvd_rolling")):
            cvd_at_reclaim = float(cvd_row["cvd_rolling"])
        if pd.notna(cvd_row.get("cvd_slope")):
            cvd_slope_at_reclaim = float(cvd_row["cvd_slope"])

    return {
        "reclaim_depth_pct": (reclaim_high - climax_high) / climax_high if climax_high else np.nan,
        "rejection_magnitude_pct": upper_wick / reclaim_close if reclaim_close else np.nan,
        "wick_to_body_ratio": (upper_wick / body) if body > 0 else np.nan,
        "volume_at_reclaim_ratio": (float(reclaim_bar["volume"]) / recent_volume_mean) if recent_volume_mean and recent_volume_mean > 0 else np.nan,
        "volume_contraction_from_climax": (float(reclaim_bar["volume"]) / climax_volume) if climax_volume > 0 else np.nan,
        "cvd_at_reclaim": cvd_at_reclaim,
        "cvd_slope_at_reclaim": cvd_slope_at_reclaim,
        "oi_change_climax_to_reclaim_pct": _oi_change(feature_frame, climax_idx_abs, reclaim_idx_abs),
        "distance_from_climax_pct": (climax_high - reclaim_close) / climax_high if climax_high else np.nan,
        "time_climax_to_reclaim_bars": reclaim_idx_abs - climax_idx_abs,
        "avwap_distance_pct": ((reclaim_close - avwap_at_reclaim) / avwap_at_reclaim) if avwap_at_reclaim else np.nan,
        "reclaim_above_avwap": bool(reclaim_close > avwap_at_reclaim) if pd.notna(avwap_at_reclaim) else None,
    }


MIN_N_FOR_UNIVARIATE_LOGIT = 30


def univariate_logit_check(quality_df: pd.DataFrame, outcome_series: pd.Series) -> pd.DataFrame:
    """Phase 2B point 12: "simple uni/multivariate diagnostic regression
    ONLY if the sample allows it" - deliberately univariate (one feature +
    const per fit), not a combined multivariate model: at N in the dozens, a
    multivariate fit across ~10 correlated quality features would overfit
    and produce unstable coefficients long before it produced anything
    trustworthy. Each row is an independent, purely descriptive/
    hypothesis-generating check - never a score, never combined into a
    single prediction. Skips (no row) below MIN_N_FOR_UNIVARIATE_LOGIT
    complete observations for a given feature."""
    import statsmodels.api as sm

    rows = []
    y_full = (outcome_series == "profit").astype(float)
    for name in QUALITY_FEATURE_NAMES:
        if name not in quality_df.columns:
            continue
        x_raw = pd.to_numeric(quality_df[name], errors="coerce")
        mask = x_raw.notna() & y_full.reindex(quality_df.index).notna()
        x, y = x_raw[mask], y_full.reindex(quality_df.index)[mask]
        n = int(mask.sum())
        if n < MIN_N_FOR_UNIVARIATE_LOGIT or y.nunique() < 2:
            rows.append({"feature": name, "n": n, "coef": np.nan, "p_value": np.nan, "pseudo_r2": np.nan, "note": "insufficient n or no outcome variation"})
            continue
        x_std = (x - x.mean()) / x.std() if x.std() > 0 else x
        design = sm.add_constant(x_std)
        try:
            model = sm.Logit(y, design).fit(disp=0)
            rows.append(
                {
                    "feature": name, "n": n, "coef": float(model.params.iloc[1]), "p_value": float(model.pvalues.iloc[1]),
                    "pseudo_r2": float(model.prsquared), "note": "",
                }
            )
        except Exception as exc:  # perfect/quasi-separation at small N is a real, expected failure mode - not a bug
            rows.append({"feature": name, "n": n, "coef": np.nan, "p_value": np.nan, "pseudo_r2": np.nan, "note": f"fit failed: {exc.__class__.__name__}"})
    return pd.DataFrame(rows)


def compare_quality_by_outcome(quality_df: pd.DataFrame, outcome_series: pd.Series) -> pd.DataFrame:
    """Point 6's actual question: for each continuous feature, does the
    'profit' group (first-passage outcome resolved in the short's favor)
    differ from the rest (adverse+timeout pooled - both represent "the
    reclaim did NOT lead to a clean favorable move")? Uses
    stats.effect_size_report (bootstrap CIs + Mann-Whitney + Cliff's delta) -
    at N in the dozens this is a hypothesis-generation pass, not a
    confirmed finding either way."""
    if quality_df.empty:
        return pd.DataFrame(columns=["feature", "n_a", "n_b", "median_a", "median_b", "cliffs_delta"])
    is_success = (outcome_series == "profit").reindex(quality_df.index).fillna(False)
    rows = []
    for name in QUALITY_FEATURE_NAMES:
        if name not in quality_df.columns:
            continue
        col = pd.to_numeric(quality_df[name], errors="coerce")
        report = research_stats.effect_size_report(col[is_success], col[~is_success], "profit", "not_profit")
        report["feature"] = name
        rows.append(report)
    return pd.DataFrame(rows)
