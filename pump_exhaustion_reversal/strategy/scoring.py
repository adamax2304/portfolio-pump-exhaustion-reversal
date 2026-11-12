"""Pump Score (spec section 8) and Exhaustion Score (spec section 12).

Both are weighted averages of 0-100 normalized components, RENORMALIZED over
only the components actually available on a given bar - a missing input
(e.g. no liquidation data) must never silently score as 0 (spec section 8:
"не ставить 0 автоматически ... пересчитать score по available weights;
снизить data_confidence; пометить missing_liquidation_data=True").

Convention: "missing" is always represented as NaN (never Python None), so
the exact same arithmetic works whether you're scoring one bar (pass
length-1 Series, or use the weighted_score() scalar wrapper) or an entire
symbol's history at once (weighted_score_vectorized() - what pump_detector.py
uses to scan full history efficiently). There is only one implementation of
the renormalization logic; the scalar path is a thin wrapper around it, so
the two can't silently drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scaled_component(value, divisor: float):
    """clip(value/divisor, 0, 1) * 100 - spec section 8. NaN-preserving;
    works on a scalar, a pandas Series, or a numpy array."""
    return np.clip(np.true_divide(value, divisor), 0, 1) * 100


def scaled_abs_component(value, divisor: float):
    return np.clip(np.abs(value) / divisor, 0, 1) * 100


def boolean_component(flag):
    """bool (or Series of bool/None/NaN) -> 100.0 / 0.0 / NaN."""
    if isinstance(flag, pd.Series):
        return flag.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else (100.0 if v else 0.0))
    return np.nan if flag is None else (100.0 if flag else 0.0)


def liquidity_quality_score(quote_volume_24h, preferred_min: float, min_floor: float):
    """0-100, linearly scaled between min_floor (->0) and preferred_min (->100), clipped."""
    if preferred_min <= min_floor:
        return np.nan
    frac = (np.asarray(quote_volume_24h, dtype="float64") - min_floor) / (preferred_min - min_floor)
    return np.clip(frac, 0, 1) * 100


def weighted_score_vectorized(components: dict[str, pd.Series], weights: dict[str, float]) -> pd.DataFrame:
    """components: name -> Series (NaN where unavailable for that row), all
    sharing the same index. weights: name -> weight (spec's *_score_weights,
    validated by config.py to sum to 100). Returns a DataFrame indexed like
    the inputs with: score (0-100), available_weight_fraction (1.0 = every
    component was available), and one missing_{name} bool column per input.
    """
    names = list(components.keys())
    index = next(iter(components.values())).index
    comp_df = pd.DataFrame({k: pd.Series(v, index=index, dtype="float64") for k, v in components.items()})
    weight_arr = np.array([weights[k] for k in names], dtype="float64")
    values = comp_df[names].to_numpy(dtype="float64")
    mask = ~np.isnan(values)

    weighted_vals = np.where(mask, values, 0.0) * weight_arr[None, :]
    available_weight = (mask * weight_arr[None, :]).sum(axis=1)
    total_weight = weight_arr.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_score = np.where(available_weight > 0, weighted_vals.sum(axis=1) / available_weight, 0.0)

    out = pd.DataFrame(index=index)
    out["score"] = np.clip(raw_score, 0, 100)
    out["available_weight_fraction"] = np.where(total_weight > 0, available_weight / total_weight, 0.0)
    for j, name in enumerate(names):
        out[f"missing_{name}"] = ~mask[:, j]
    return out


def weighted_score(components: dict[str, float], weights: dict[str, float]) -> dict:
    """Scalar convenience wrapper around weighted_score_vectorized for a
    single observation - components values are NaN (not None) for "missing".
    """
    series_components = {k: pd.Series([v], dtype="float64") for k, v in components.items()}
    row = weighted_score_vectorized(series_components, weights).iloc[0]
    missing = [k for k in components if bool(row[f"missing_{k}"])]
    return {
        "score": float(row["score"]),
        "missing_components": missing,
        "available_weight_fraction": float(row["available_weight_fraction"]),
    }


PUMP_SCORE_COMPONENT_NAMES = (
    "abnormal_return", "volume_anomaly", "volatility_expansion", "liquidation_spike",
    "oi_anomaly", "perp_spot_imbalance", "liquidity_marketcap_context",
)

EXHAUSTION_SCORE_COMPONENT_NAMES = (
    "structure_break", "failed_reclaim_lower_high", "cvd_bearish_divergence", "oi_rollover",
    "volume_decay", "short_liquidation_exhaustion", "sweep_rejection", "avwap_loss",
    "funding_long_crowding", "rsi_divergence",
)


def compute_pump_score(
    max_return_zscore: float,
    volume_zscore: float,
    realized_vol_ratio: float,
    short_liq_zscore: float,
    oi_zscore: float,
    perp_spot_imbalance_score: float,
    liquidity_score: float,
    weights,
) -> dict:
    """Scalar pump score for one bar/event. Pass np.nan for any unavailable input."""
    components = {
        "abnormal_return": scaled_component(max_return_zscore, 6.0),
        "volume_anomaly": scaled_component(volume_zscore, 6.0),
        "volatility_expansion": scaled_component(realized_vol_ratio, 5.0),
        "liquidation_spike": scaled_component(short_liq_zscore, 8.0),
        "oi_anomaly": scaled_abs_component(oi_zscore, 5.0),
        "perp_spot_imbalance": perp_spot_imbalance_score,
        "liquidity_marketcap_context": liquidity_score,
    }
    result = weighted_score(components, weights.model_dump())
    result["components"] = components
    result["missing_liquidation_data"] = "liquidation_spike" in result["missing_components"]
    return result


def compute_exhaustion_score(flags: dict[str, bool | None], weights) -> dict:
    """flags keys must match EXHAUSTION_SCORE_COMPONENT_NAMES / weights fields."""
    components = {k: boolean_component(v) for k, v in flags.items()}
    result = weighted_score(components, weights.model_dump())
    result["components"] = components
    return result
