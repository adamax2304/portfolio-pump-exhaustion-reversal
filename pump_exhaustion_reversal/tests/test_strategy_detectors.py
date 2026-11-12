from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.config import AppConfig, PumpDetectorConfig, PumpScoreWeights
from pump_exhaustion_reversal.strategy import catalyst_filter, exhaustion_detector, pump_classifier, pump_detector


# ---- pump_classifier.py --------------------------------------------------


def _classify_defaults(**overrides):
    base = dict(
        has_recent_positive_catalyst=False,
        short_liq_zscore=np.nan,
        short_liq_depletion_ratio=np.nan,
        oi_pct_change_during_pump=np.nan,
        oi_zscore=np.nan,
        funding_pre_pump_percentile=np.nan,
        funding_direction_rising=False,
        cvd_confidence="low",
        cvd_slope_positive=False,
        liquidity_confidence="HIGH",
        upper_wick_body_ratio_climax=np.nan,
        pump_duration_bars=np.nan,
        typical_pump_duration_bars=np.nan,
    )
    base.update(overrides)
    return pump_classifier.classify_pump(**base)


def test_classify_pump_catalyst_overrides_everything():
    result = _classify_defaults(has_recent_positive_catalyst=True, short_liq_zscore=9.0)
    assert result.pump_class == pump_classifier.PumpClass.NEWS_FUNDAMENTAL
    assert result.confidence == "high"


def test_classify_pump_short_squeeze():
    result = _classify_defaults(short_liq_zscore=6.0, short_liq_depletion_ratio=0.3, funding_pre_pump_percentile=20)
    assert result.pump_class == pump_classifier.PumpClass.SHORT_SQUEEZE
    assert result.confidence == "high"


def test_classify_pump_leveraged_long():
    result = _classify_defaults(oi_pct_change_during_pump=15.0, funding_pre_pump_percentile=90, cvd_confidence="high")
    assert result.pump_class == pump_classifier.PumpClass.LEVERAGED_LONG


def test_classify_pump_liquidity_manipulation():
    result = _classify_defaults(liquidity_confidence="LOW", upper_wick_body_ratio_climax=3.5)
    assert result.pump_class == pump_classifier.PumpClass.LIQUIDITY_MANIPULATION


def test_classify_pump_spot_organic_is_fallback_with_notes():
    result = _classify_defaults()
    assert result.pump_class == pump_classifier.PumpClass.SPOT_ORGANIC
    assert result.confidence == "low"
    assert result.data_confidence_notes  # must explain it's a fallback, not a real detection


# ---- catalyst_filter.py --------------------------------------------------


def test_catalyst_filter_disabled_allows_everything():
    from pump_exhaustion_reversal.config import CatalystFilterConfig

    status = catalyst_filter.check_catalyst(None, "BTCUSDT", 0, CatalystFilterConfig(enabled=False))
    assert status.aggressive_allowed and status.standard_allowed and status.conservative_allowed == "yes"


def test_catalyst_filter_none_df_is_unknown_and_conservative():
    from pump_exhaustion_reversal.config import CatalystFilterConfig

    status = catalyst_filter.check_catalyst(None, "BTCUSDT", 0, CatalystFilterConfig())
    assert status.news_confidence == "UNKNOWN"
    assert not status.aggressive_allowed
    assert not status.standard_allowed
    assert status.conservative_allowed == "only_if_strong_confirmation"


def test_catalyst_filter_blocks_around_positive_news():
    from pump_exhaustion_reversal.config import CatalystFilterConfig

    cfg = CatalystFilterConfig(
        block_aggressive_if_positive_news_within_hours=12,
        block_standard_if_positive_news_within_hours=6,
        allow_conservative_after_hours=3,
    )
    now = 20 * 3_600_000
    catalyst_df = pd.DataFrame(
        {
            "timestamp": [now - 8 * 3_600_000],
            "symbol": ["BTCUSDT"],
            "event_type": ["listing"],
            "sentiment": ["positive"],
            "source": ["manual"],
            "confidence": ["high"],
        }
    )
    status = catalyst_filter.check_catalyst(catalyst_df, "BTCUSDT", now, cfg)
    assert status.major_positive is True
    assert status.aggressive_allowed is False  # 8h < 12h block window
    assert status.standard_allowed is True  # 8h > 6h block window
    assert status.conservative_allowed == "yes"  # 8h > 3h allow-after window


def test_catalyst_filter_no_news_allows_everything():
    from pump_exhaustion_reversal.config import CatalystFilterConfig

    catalyst_df = pd.DataFrame(columns=["timestamp", "symbol", "event_type", "sentiment", "source", "confidence"])
    status = catalyst_filter.check_catalyst(catalyst_df, "BTCUSDT", 100_000, CatalystFilterConfig())
    assert status.aggressive_allowed and status.standard_allowed
    assert status.conservative_allowed == "yes"


# ---- pump_detector.py --------------------------------------------------


def _pump_feature_frame(n=5):
    idx = pd.RangeIndex(n)
    return pd.DataFrame(
        {
            "return_zscore_15m": [0.0, 1.0, 3.5, 4.0, 4.0],
            "abnormal_return_15m": [0.0, 0.02, 0.09, 0.10, 0.10],
            "volume_zscore": [0.0, 1.0, 4.0, 5.0, 5.0],
            "realized_vol_ratio": [1.0, 1.5, 3.0, 3.5, 3.5],
        },
        index=idx,
    )


def test_pump_detector_fires_when_all_gates_pass():
    frame = _pump_feature_frame()
    cfg_detector = PumpDetectorConfig(windows=["15m"], min_return_zscore=3.0, min_volume_zscore=3.0, min_abnormal_return_15m=0.08)
    out = pump_detector.compute_pump_score_frame(frame, liquidity_score=100.0, cfg_detector=cfg_detector, cfg_weights=PumpScoreWeights())
    assert not out["pump_hard_gate_passed"].iloc[:2].any()
    assert out["pump_hard_gate_passed"].iloc[2:].all()
    # pump_detected also needs pump_score >= min_pump_score(70) - only bars 3-4
    # are strong enough for that once liquidation/OI/perp-spot weight is missing
    assert not out["pump_detected"].iloc[2]
    assert out["pump_detected"].iloc[3:].all()
    assert out["missing_liquidation_data"].all()  # no short_liq_zscore column supplied


def test_pump_detector_score_alone_does_not_bypass_hard_gate():
    # huge volume/vol-ratio push the score up, but return z-score never clears
    # min_return_zscore -> pump_hard_gate_passed must stay False (spec section 7:
    # fixed thresholds are a secondary guard, not overridable by score alone)
    frame = pd.DataFrame(
        {
            "return_zscore_15m": [1.0],
            "abnormal_return_15m": [0.01],
            "volume_zscore": [6.0],
            "realized_vol_ratio": [5.0],
        }
    )
    cfg_detector = PumpDetectorConfig(windows=["15m"], min_return_zscore=3.0, min_volume_zscore=3.0, min_abnormal_return_15m=0.08, min_pump_score=1.0)
    out = pump_detector.compute_pump_score_frame(frame, liquidity_score=100.0, cfg_detector=cfg_detector, cfg_weights=PumpScoreWeights())
    assert out["pump_score"].iloc[0] > 50  # score is high...
    assert not out["pump_hard_gate_passed"].iloc[0]  # ...but hard gate correctly blocks it
    assert not out["pump_detected"].iloc[0]


# ---- exhaustion_detector.py --------------------------------------------------


def test_exhaustion_flags_missing_sources_are_none_not_false():
    structure = pd.DataFrame({"structure_break_bar": [False, False, True]})
    failed_reclaim = pd.Series([False, False, False])
    sweep = pd.Series([False, False, False])
    inputs = exhaustion_detector.ExhaustionInputs(structure=structure, failed_reclaim=failed_reclaim, sweep_rejection=sweep)
    cfg = AppConfig()
    flags = exhaustion_detector.compute_exhaustion_flags_at(2, inputs, cfg)
    assert flags.structure_break is True
    assert flags.oi_rollover is None  # not supplied -> unknown, not "no rollover"
    assert flags.cvd_bearish_divergence is None
    assert flags.short_liquidation_exhaustion is None


def test_exhaustion_flags_structure_break_persists_after_first_true():
    structure = pd.DataFrame({"structure_break_bar": [False, True, False, False]})
    failed_reclaim = pd.Series([False, False, False, False])
    sweep = pd.Series([False, False, False, False])
    inputs = exhaustion_detector.ExhaustionInputs(structure=structure, failed_reclaim=failed_reclaim, sweep_rejection=sweep)
    cfg = AppConfig()
    flags_before = exhaustion_detector.compute_exhaustion_flags_at(0, inputs, cfg)
    flags_at = exhaustion_detector.compute_exhaustion_flags_at(1, inputs, cfg)
    flags_after = exhaustion_detector.compute_exhaustion_flags_at(3, inputs, cfg)
    assert flags_before.structure_break is False
    assert flags_at.structure_break is True
    assert flags_after.structure_break is True  # stays broken, even though bar 3 itself isn't the break bar


def test_classify_exhaustion_state_thresholds():
    cfg = AppConfig()
    assert exhaustion_detector.classify_exhaustion_state(10, cfg) == "WATCHING"
    assert exhaustion_detector.classify_exhaustion_state(55, cfg) == "EXHAUSTION_FORMING"
    assert exhaustion_detector.classify_exhaustion_state(70, cfg) == "ARMED_SHORT"
    assert exhaustion_detector.classify_exhaustion_state(90, cfg) == "ARMED_SHORT"


def test_full_pump_with_intact_structure_never_reaches_meaningful_exhaustion():
    """Regression guard for spec section 28's key test at the strategy level:
    a pump with intact structure (no break, no failed reclaim, no sweep) must
    score at most the low-weight confirmation-only components, never enough
    to cross even the 'watch' exhaustion threshold on structure-driven signals."""
    n = 10
    structure = pd.DataFrame({"structure_break_bar": [False] * n})
    failed_reclaim = pd.Series([False] * n)
    sweep = pd.Series([False] * n)
    inputs = exhaustion_detector.ExhaustionInputs(structure=structure, failed_reclaim=failed_reclaim, sweep_rejection=sweep)
    cfg = AppConfig()
    flags = exhaustion_detector.compute_exhaustion_flags_at(n - 1, inputs, cfg)
    result = exhaustion_detector.compute_exhaustion_score_at(flags, cfg)
    # only rsi/funding/oi/cvd/volume/liq weights are even in play (all None here -> excluded),
    # structure_break/failed_reclaim/sweep_rejection are explicitly False -> 0 contribution
    assert result["score"] == 0.0
    assert exhaustion_detector.classify_exhaustion_state(result["score"], cfg) == "WATCHING"
