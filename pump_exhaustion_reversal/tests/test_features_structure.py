from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.config import FailedReclaimConfig, StructureConfig, SweepConfig
from pump_exhaustion_reversal.features import structure as struct_feat


def _zigzag_df(waypoints: list[tuple[int, float]]) -> pd.DataFrame:
    """Piecewise-linear price path through (bar_index, price) waypoints, with
    open=high=low=close (zero intrabar range) so pivot/zone arithmetic is exact
    and easy to reason about by hand.
    """
    xs = [w[0] for w in waypoints]
    ys = [w[1] for w in waypoints]
    n = xs[-1] + 1
    idx = np.arange(n)
    mid = np.interp(idx, xs, ys)
    return pd.DataFrame(
        {"timestamp": idx * 300_000, "open": mid, "high": mid, "low": mid, "close": mid}
    )


DEFAULT_STRUCTURE_CFG = StructureConfig()


def _atr_for(df: pd.DataFrame) -> pd.Series:
    a = struct_feat.atr(df, length=14)
    return a.fillna(1.0)  # avoid NaN-driven distance_ok short-circuits during warmup


# ---- atr / find_pivots -----------------------------------------------------


def test_atr_converges_to_constant_true_range():
    n = 60
    close = pd.Series(100 + np.cumsum(np.zeros(n)))  # flat closes
    high = close + 1.0
    low = close - 1.0
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    a = struct_feat.atr(df, length=14)
    assert a.iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_find_pivots_simple_zigzag():
    df = _zigzag_df([(0, 100), (5, 110), (10, 100), (15, 112)])
    pivots = struct_feat.find_pivots(df["high"], df["low"], left_bars=2, right_bars=2)
    assert pivots.loc[5, "pivot_high"]
    assert pivots.loc[10, "pivot_low"]
    # last right_bars=2 bars can never be confirmed pivots (not knowable yet)
    assert not pivots.loc[len(df) - 1, "pivot_high"]
    assert not pivots.loc[len(df) - 2, "pivot_high"]


# ---- compute_structure: the critical "don't short strength" property ------


def test_intact_uptrend_never_breaks_structure():
    """Spec section 28's key acceptance test at the feature level: a clean
    HH/HL uptrend must never report a structure break, however large the move."""
    df = _zigzag_df([(0, 100), (6, 112), (12, 108), (18, 128), (24, 122), (30, 145), (36, 138), (42, 160)])
    atr_series = _atr_for(df)
    structure = struct_feat.compute_structure(df, atr_series, DEFAULT_STRUCTURE_CFG)

    assert not structure["structure_break_bar"].any()
    assert structure["structure_intact"].all()
    hl_labels = structure["swing_low_label"].dropna().unique().tolist()
    hh_labels = structure["swing_high_label"].dropna().unique().tolist()
    assert "HL" in hl_labels
    assert "HH" in hh_labels
    assert "LL" not in hl_labels
    assert "LH" not in hh_labels


def test_break_then_failed_reclaim_sequence():
    """Up-leg -> up-leg (HH/HL) -> blow-off high -> sharp break of the last HL
    -> weak bounce that pokes into the reclaim zone but fails and closes back
    below -> failed_reclaim must fire; it must not fire before the bounce."""
    df = _zigzag_df(
        [
            (0, 100),
            (5, 110),   # pivot high #1
            (10, 104),  # pivot low #1 ("L")
            (16, 122),  # pivot high #2 (HH)
            (22, 114),  # pivot low #2 ("HL", 114 > 104) - this is the level that breaks
            (28, 132),  # pivot high #3 (HH) -> pre_break_swing_high
            (34, 100),  # sharp drop through 114
            (40, 114.2),  # weak bounce, pokes just into the reclaim zone above 114
            (46, 95),   # fails, drops again
        ]
    )
    atr_series = _atr_for(df)
    structure = struct_feat.compute_structure(df, atr_series, DEFAULT_STRUCTURE_CFG)

    break_bars = structure.index[structure["structure_break_bar"]].tolist()
    assert break_bars, "expected a structure break somewhere on the sharp drop through 114"
    break_bar = break_bars[0]
    assert structure.loc[break_bar, "broken_level"] == pytest.approx(114, abs=0.01)
    assert structure.loc[break_bar, "pre_break_swing_high"] == pytest.approx(132, abs=0.01)
    assert not structure["structure_intact"].iloc[break_bar + 1 :].any()

    failed_reclaim = struct_feat.detect_failed_reclaim(df, structure, FailedReclaimConfig())
    assert not failed_reclaim.iloc[: break_bar + 1].any()  # can't fire before/at the break itself
    assert not failed_reclaim.loc[39]  # still mid-bounce, hasn't closed back below yet
    assert failed_reclaim.loc[41]  # bounce failed and closed back below the broken level


def test_failed_reclaim_disabled_returns_all_false():
    df = _zigzag_df([(0, 100), (5, 110), (10, 90)])
    atr_series = _atr_for(df)
    structure = struct_feat.compute_structure(df, atr_series, DEFAULT_STRUCTURE_CFG)
    cfg = FailedReclaimConfig(enabled=False)
    result = struct_feat.detect_failed_reclaim(df, structure, cfg)
    assert not result.any()


# ---- compute_swing_magnitude -----------------------------------------------


def test_swing_magnitude_hand_computed_values_on_known_zigzag():
    df = _zigzag_df([(0, 100), (5, 110), (10, 104), (16, 122), (22, 114), (28, 132), (34, 126)])
    # Constant ATR isolates the swing_distance_atr/preceding_impulse_atr arithmetic from atr()'s own
    # EWM formula - the bps side of compute_structure's distance gate already clears every swing below
    # regardless of ATR, so this doesn't change which pivots get confirmed.
    atr_series = pd.Series(2.0, index=df.index)
    structure = struct_feat.compute_structure(df, atr_series, DEFAULT_STRUCTURE_CFG)
    magnitude = struct_feat.compute_swing_magnitude(structure, df, atr_series, DEFAULT_STRUCTURE_CFG)

    # Sanity: confirm the pivots landed on the bars/prices this test assumes before checking magnitude.
    assert structure.loc[7, "swing_high_label"] == "H" and structure.loc[7, "swing_high_level"] == 110
    assert structure.loc[12, "swing_low_label"] == "L" and structure.loc[12, "swing_low_level"] == 104
    assert structure.loc[18, "swing_high_label"] == "HH" and structure.loc[18, "swing_high_level"] == 122
    assert structure.loc[24, "swing_low_label"] == "HL" and structure.loc[24, "swing_low_level"] == 114
    assert structure.loc[30, "swing_high_label"] == "HH" and structure.loc[30, "swing_high_level"] == 132

    fresh_bars = [7, 12, 18, 24, 30]
    not_fresh = magnitude.index.difference(fresh_bars)
    assert magnitude.loc[not_fresh].isna().all().all()

    # bar 7 is the very first swing of the whole sequence -> no prior opposite-type swing exists yet
    assert np.isnan(magnitude.loc[7, "swing_distance_atr"])
    assert np.isnan(magnitude.loc[7, "swing_duration_bars"])

    # distance/duration: exact arithmetic off the known round pivot prices/bar indices, atr=2.0 constant
    assert magnitude.loc[12, "swing_distance_atr"] == pytest.approx(abs(104 - 110) / 2.0)
    assert magnitude.loc[12, "swing_duration_bars"] == 12 - 7
    assert magnitude.loc[18, "swing_distance_atr"] == pytest.approx(abs(122 - 104) / 2.0)
    assert magnitude.loc[18, "swing_duration_bars"] == 18 - 12
    assert magnitude.loc[24, "swing_distance_atr"] == pytest.approx(abs(114 - 122) / 2.0)
    assert magnitude.loc[24, "swing_duration_bars"] == 24 - 18
    assert magnitude.loc[30, "swing_distance_atr"] == pytest.approx(abs(132 - 114) / 2.0)
    assert magnitude.loc[30, "swing_duration_bars"] == 30 - 24

    # Impulse fields: derive the expected value directly from df's own close series (independent of the
    # function under test) instead of hand-computing the zigzag's interpolated prices.
    close = df["close"]
    lookback = DEFAULT_STRUCTURE_CFG.pivot_left_bars + DEFAULT_STRUCTURE_CFG.pivot_right_bars
    for bar in fresh_bars:
        start = max(0, bar - lookback)
        expected_pct = (close[bar] - close[start]) / close[start]
        expected_atr = (close[bar] - close[start]) / 2.0
        assert magnitude.loc[bar, "preceding_impulse_pct"] == pytest.approx(expected_pct)
        assert magnitude.loc[bar, "preceding_impulse_atr"] == pytest.approx(expected_atr)


def test_swing_magnitude_no_confirmed_swings_is_all_nan():
    # n=4 with the default pivot_left_bars=pivot_right_bars=2 makes find_pivots's own
    # range(left_bars, n - right_bars) empty, so no bar can ever be flagged a pivot at all -
    # compute_structure confirms zero swings of either type, regardless of price shape.
    df = _zigzag_df([(0, 100), (3, 100)])
    atr_series = _atr_for(df)
    structure = struct_feat.compute_structure(df, atr_series, DEFAULT_STRUCTURE_CFG)
    assert structure["swing_high_level"].isna().all() and structure["swing_low_level"].isna().all()

    magnitude = struct_feat.compute_swing_magnitude(structure, df, atr_series, DEFAULT_STRUCTURE_CFG)
    assert list(magnitude.columns) == [
        "swing_distance_atr", "swing_duration_bars", "preceding_impulse_pct", "preceding_impulse_atr",
    ]
    assert magnitude.index.equals(structure.index)
    assert magnitude.isna().all().all()


def test_swing_magnitude_output_aligned_to_structure_index():
    df = _zigzag_df([(0, 100), (5, 110), (10, 104), (16, 122)])
    atr_series = _atr_for(df)
    structure = struct_feat.compute_structure(df, atr_series, DEFAULT_STRUCTURE_CFG)
    magnitude = struct_feat.compute_swing_magnitude(structure, df, atr_series, DEFAULT_STRUCTURE_CFG)
    assert len(magnitude) == len(structure)
    assert magnitude.index.equals(structure.index)


# ---- detect_sweep_rejection (hand-built structure input) -------------------


def test_sweep_rejection_same_bar_pin():
    df = pd.DataFrame(
        {
            "open": [105.0, 106.0, 108.0],
            "high": [106.0, 107.0, 112.0],
            "low": [104.0, 105.0, 107.8],
            "close": [105.5, 106.5, 108.2],
        }
    )
    structure = pd.DataFrame({"swing_high_level": [110.0, 110.0, 110.0]})
    cfg = SweepConfig()
    result = struct_feat.detect_sweep_rejection(df, structure, cfg)
    assert result.tolist() == [False, False, True]


def test_sweep_rejection_confirmed_a_few_bars_later():
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 109.8, 110.5],
            "high": [101.0, 101.0, 112.0, 110.6],
            "low": [99.0, 99.0, 109.7, 109.0],
            "close": [100.5, 100.5, 110.5, 109.5],
        }
    )
    structure = pd.DataFrame({"swing_high_level": [110.0, 110.0, 110.0, 110.0]})
    cfg = SweepConfig()
    result = struct_feat.detect_sweep_rejection(df, structure, cfg)
    assert result.tolist() == [False, False, False, True]


def test_sweep_rejection_requires_wick_ratio_not_just_new_high():
    # sweeps the high AND closes back below it, but the wick is not big enough
    # relative to the body (ratio 1.67 < the 2.0 minimum) -> must not count
    df = pd.DataFrame(
        {
            "open": [108.0],
            "high": [112.0],
            "low": [107.9],
            "close": [109.5],  # body=1.5, upper_wick=112-109.5=2.5, ratio=1.67
        }
    )
    structure = pd.DataFrame({"swing_high_level": [110.0]})
    result = struct_feat.detect_sweep_rejection(df, structure, SweepConfig())
    assert result.tolist() == [False]
