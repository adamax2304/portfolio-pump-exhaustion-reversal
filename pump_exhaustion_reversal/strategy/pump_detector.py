"""Pump Detector (spec sections 7-8): finds abnormal pump bars, never opens a
trade. Callers (state_machine) use pump_detected to transition IDLE ->
PUMP_DETECTED. The same function scans full history (event study) or a live
tail (Stage 2) identically - it only ever reads the current row's already
backward-looking feature columns, so there is nothing here that could peek
ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PumpDetectorConfig, PumpScoreWeights
from . import scoring

DATA_CONFIDENCE_HIGH_THRESHOLD = 0.70
_WINDOW_LABELS_WITH_FIXED_GUARDS = ("15m", "1h", "4h", "24h")


def compute_pump_score_frame(
    feature_frame: pd.DataFrame,
    liquidity_score: float,
    cfg_detector: PumpDetectorConfig,
    cfg_weights: PumpScoreWeights,
) -> pd.DataFrame:
    """feature_frame must contain, for each window in cfg_detector.windows:
    return_zscore_{w} and abnormal_return_{w} (see features/returns.py);
    plus volume_zscore, realized_vol_ratio (features/volume.py) and
    optionally short_liq_zscore (features/liquidations.py),
    oi_zscore (features/oi.py) - omit or fill with NaN if unavailable.
    liquidity_score is a single scalar (0-100, from
    scoring.liquidity_quality_score) - Stage 1 has no time-varying
    liquidity feed, just the universe snapshot nearest to the event.

    Returns a frame (same index) with: pump_score, data_confidence
    ('HIGH'/'LOW'), missing_liquidation_data, pump_hard_gate_passed,
    pump_detected (= hard gate passed AND pump_score >= min_pump_score -
    spec section 7: fixed-percentage thresholds are "a secondary guard, не
    как единственный критерий", so both must hold, neither alone suffices).
    """
    index = feature_frame.index
    zscore_cols = [
        f"return_zscore_{w}" for w in cfg_detector.windows if f"return_zscore_{w}" in feature_frame.columns
    ]
    max_return_zscore = (
        feature_frame[zscore_cols].abs().max(axis=1) if zscore_cols else pd.Series(np.nan, index=index)
    )

    def _col(name: str) -> pd.Series:
        return feature_frame[name] if name in feature_frame.columns else pd.Series(np.nan, index=index)

    volume_zscore = _col("volume_zscore")
    realized_vol_ratio = _col("realized_vol_ratio")
    short_liq_zscore = _col("short_liq_zscore")
    oi_zscore = _col("oi_zscore")
    liquidity_series = pd.Series(float(liquidity_score), index=index, dtype="float64")

    components = {
        "abnormal_return": scoring.scaled_component(max_return_zscore, 6.0),
        "volume_anomaly": scoring.scaled_component(volume_zscore, 6.0),
        "volatility_expansion": scoring.scaled_component(realized_vol_ratio, 5.0),
        "liquidation_spike": scoring.scaled_component(short_liq_zscore, 8.0),
        "oi_anomaly": scoring.scaled_abs_component(oi_zscore, 5.0),
        "perp_spot_imbalance": pd.Series(np.nan, index=index),  # Stage 1: no spot connector (project plan)
        "liquidity_marketcap_context": liquidity_series,
    }
    score_frame = scoring.weighted_score_vectorized(components, cfg_weights.model_dump())

    out = pd.DataFrame(index=index)
    out["pump_score"] = score_frame["score"]
    out["pump_score_available_weight_fraction"] = score_frame["available_weight_fraction"]
    out["missing_liquidation_data"] = score_frame["missing_liquidation_spike"]
    out["data_confidence"] = np.where(
        score_frame["available_weight_fraction"] >= DATA_CONFIDENCE_HIGH_THRESHOLD, "HIGH", "LOW"
    )
    out["max_return_zscore"] = max_return_zscore

    any_window_ok = pd.Series(False, index=index)
    for w in _WINDOW_LABELS_WITH_FIXED_GUARDS:
        threshold_field = f"min_abnormal_return_{w}"
        col = f"abnormal_return_{w}"
        if hasattr(cfg_detector, threshold_field) and col in feature_frame.columns:
            any_window_ok = any_window_ok | (feature_frame[col] >= getattr(cfg_detector, threshold_field))
    zscore_ok = max_return_zscore >= cfg_detector.min_return_zscore
    volume_ok = volume_zscore >= cfg_detector.min_volume_zscore

    out["pump_hard_gate_passed"] = (any_window_ok & zscore_ok & volume_ok).fillna(False)
    out["pump_detected"] = out["pump_hard_gate_passed"] & (out["pump_score"] >= cfg_detector.min_pump_score)
    return out
