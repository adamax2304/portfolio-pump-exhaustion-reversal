"""Pump Classifier A-E (spec section 9).

Several of the spec's textbook signatures need data Stage 1 does not collect
(spot orderflow/CVD for class C, cross-exchange confirmation for class E,
long/short account-ratio for class B). Where a signature can't be positively
confirmed, classify_pump() says so in `data_confidence_notes` rather than
silently guessing - SPOT_ORGANIC in particular is a residual/fallback here
(no signature for A/B/D/E matched), not a positive detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class PumpClass(str, Enum):
    SHORT_SQUEEZE = "SHORT_SQUEEZE"
    LEVERAGED_LONG = "LEVERAGED_LONG"
    SPOT_ORGANIC = "SPOT_ORGANIC"
    NEWS_FUNDAMENTAL = "NEWS_FUNDAMENTAL"
    LIQUIDITY_MANIPULATION = "LIQUIDITY_MANIPULATION"


@dataclass
class PumpClassification:
    pump_class: PumpClass
    confidence: str  # "high" | "medium" | "low"
    reasons: list[str] = field(default_factory=list)
    data_confidence_notes: list[str] = field(default_factory=list)


def _nan_or(value: float | None) -> float:
    return np.nan if value is None else float(value)


def classify_pump(
    has_recent_positive_catalyst: bool,
    short_liq_zscore: float | None,
    short_liq_depletion_ratio: float | None,
    oi_pct_change_during_pump: float | None,
    oi_zscore: float | None,
    funding_pre_pump_percentile: float | None,
    funding_direction_rising: bool,
    cvd_confidence: str,
    cvd_slope_positive: bool,
    liquidity_confidence: str,
    upper_wick_body_ratio_climax: float | None,
    pump_duration_bars: float | None,
    typical_pump_duration_bars: float | None,
    catalyst_requires_manual_review: bool = False,
) -> PumpClassification:
    short_liq_zscore = _nan_or(short_liq_zscore)
    short_liq_depletion_ratio = _nan_or(short_liq_depletion_ratio)
    oi_pct_change_during_pump = _nan_or(oi_pct_change_during_pump)
    oi_zscore = _nan_or(oi_zscore)
    funding_pre_pump_percentile = _nan_or(funding_pre_pump_percentile)
    upper_wick_body_ratio_climax = _nan_or(upper_wick_body_ratio_climax)
    pump_duration_bars = _nan_or(pump_duration_bars)
    typical_pump_duration_bars = _nan_or(typical_pump_duration_bars)

    # D: news/fundamental - catalyst presence overrides the mechanism-based classes,
    # since spec's classification is about the pump's CAUSE and a real catalyst is
    # a more reliable known cause than an inferred derivatives signature.
    if has_recent_positive_catalyst:
        reasons = ["recent positive catalyst detected within lookback window"]
        notes = ["major listing/catalyst flagged for manual review"] if catalyst_requires_manual_review else []
        return PumpClassification(PumpClass.NEWS_FUNDAMENTAL, "high", reasons, notes)

    # A: short squeeze - short-liquidation proxy spike, ideally with depletion afterward,
    # funding not already crowded long before the pump started.
    if not np.isnan(short_liq_zscore) and short_liq_zscore >= 5.0:
        reasons = [f"short liquidation proxy z-score {short_liq_zscore:.1f} >= 5.0"]
        notes = []
        if not np.isnan(short_liq_depletion_ratio) and short_liq_depletion_ratio <= 0.5:
            reasons.append(f"short liquidation intensity depleted to {short_liq_depletion_ratio:.0%} of its peak")
        if np.isnan(funding_pre_pump_percentile) or funding_pre_pump_percentile <= 50:
            reasons.append("funding was neutral/negative before the pump")
        confidence = "high" if not np.isnan(short_liq_depletion_ratio) else "medium"
        notes.append("liquidation data is a proxy (OI/price/volume), not a real historical liquidation feed")
        return PumpClassification(PumpClass.SHORT_SQUEEZE, confidence, reasons, notes)

    # B: leveraged long - OI expanding strongly, funding hot and/or perp CVD confirming.
    oi_up_strong = (not np.isnan(oi_pct_change_during_pump) and oi_pct_change_during_pump >= 10) or (
        not np.isnan(oi_zscore) and oi_zscore >= 3.0
    )
    funding_hot = (
        not np.isnan(funding_pre_pump_percentile) and funding_pre_pump_percentile >= 80
    ) or funding_direction_rising
    if oi_up_strong and (funding_hot or cvd_slope_positive):
        reasons = ["OI expanded strongly during the pump"]
        notes = []
        if funding_hot:
            reasons.append("funding rising / overheated")
        if cvd_slope_positive:
            reasons.append("perp CVD trending positive")
        if cvd_confidence == "low":
            notes.append("CVD confidence low (tick-rule proxy, not real taker-side data)")
        return PumpClassification(
            PumpClass.LEVERAGED_LONG, "high" if cvd_confidence == "high" else "medium", reasons, notes
        )

    # E: liquidity/manipulation - thin-liquidity symbol with a large rejection wick and/or
    # unusually fast pump duration. No cross-exchange confirmation available in Stage 1.
    fast_pump = (
        not np.isnan(pump_duration_bars)
        and not np.isnan(typical_pump_duration_bars)
        and typical_pump_duration_bars > 0
        and pump_duration_bars <= typical_pump_duration_bars * 0.5
    )
    big_wick = not np.isnan(upper_wick_body_ratio_climax) and upper_wick_body_ratio_climax >= 2.0
    if liquidity_confidence == "LOW" and (big_wick or fast_pump):
        reasons = ["low liquidity_confidence symbol"]
        if big_wick:
            reasons.append(f"climax candle upper-wick/body ratio {upper_wick_body_ratio_climax:.1f}")
        if fast_pump:
            reasons.append("unusually fast pump duration vs typical")
        notes = ["no cross-exchange confirmation available in Stage 1 - single-venue data only"]
        return PumpClassification(PumpClass.LIQUIDITY_MANIPULATION, "medium", reasons, notes)

    # C: spot-organic - residual fallback. Spec's real signature (spot CVD/volume leading
    # perps) needs spot orderflow data Stage 1 does not collect (linear-perp-only universe).
    reasons = ["no short-squeeze, leveraged-long, catalyst, or manipulation signature matched"]
    notes = [
        "SPOT_ORGANIC is a residual fallback in Stage 1, not a positive detection - "
        "no spot orderflow data is collected to confirm spot-led buying (see README limitations)"
    ]
    return PumpClassification(PumpClass.SPOT_ORGANIC, "low", reasons, notes)
