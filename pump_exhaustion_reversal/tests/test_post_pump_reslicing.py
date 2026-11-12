"""Stage 2A core fix: proves _build_post_pump_inputs's climax-anchored
reslicing actually excludes a stale, pre-climax structure break that the
OLD wide (whole-df_slice) computation would incorrectly latch onto - this
is the exact bug the user diagnosed (see project plan "Этап 2A") and the
reason _run_single_event no longer reads a single wide `structure` frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest.simulator import _build_post_pump_inputs
from pump_exhaustion_reversal.config import AppConfig
from pump_exhaustion_reversal.features import structure as struct_feat

# (idx, open, high, low, close). Two independent HL-forms-then-breaks
# patterns: an EARLY one (bars 0-11, breaks at bar 11) that must become
# invisible once climax-anchored past it, and a POST-CLIMAX one (climax at
# bar 22, HL forms at bar 29, breaks at bar 32) that must be found.
_ROWS = [
    (0, 50, 50, 50, 50), (1, 50, 50, 50, 50),
    (2, 50, 55, 50, 54),
    (3, 54, 54, 52, 52.5),
    (4, 52.5, 53, 48, 48.5),   # low1 = 48
    (5, 48.5, 51, 48.5, 50.5),
    (6, 50.5, 52, 50, 51.5),
    (7, 51.5, 53, 51, 52.5),
    (8, 52.5, 53, 49, 49.5),   # low2 = 49, higher than low1 -> HL
    (9, 49.5, 51, 49, 50.5),
    (10, 50.5, 51, 49.5, 50),
    (11, 50, 50, 46, 46.5),    # closes below HL(49) -> EARLY break (irrelevant to any real pump leg)
    (12, 46.5, 47, 45, 45.5),
    (13, 45.5, 46, 45, 45.2),
    (14, 45.2, 46, 45, 45.5),
    (15, 46, 55, 45, 54), (16, 54, 65, 53, 64), (17, 64, 78, 63, 77),
    (18, 77, 95, 76, 94), (19, 94, 115, 93, 113), (20, 113, 125, 112, 123),
    (21, 123, 129, 122, 128), (22, 128, 131, 127, 129),  # climax: high=131 at bar 22
    (23, 129, 129, 120, 121), (24, 121, 122, 116, 117),
    (25, 117, 118, 115, 115.5),   # low1 = 115
    (26, 115.5, 119, 115, 118.5),
    (27, 118.5, 121, 118, 120.5),
    (28, 120.5, 121, 119, 119.5),
    (29, 119.5, 120, 118, 118.5),  # low2 = 118, higher than low1(115) -> HL
    (30, 118.5, 119.5, 118, 119),
    (31, 119, 119.5, 118.5, 119),
    (32, 119, 119, 114, 114.5),    # closes below HL(118) -> POST-CLIMAX break (must be found)
    (33, 114.5, 115, 113, 113.5),
    (34, 113.5, 114, 113, 113.2),
]
CLIMAX_IDX = 22
EARLY_BREAK_WIDE_IDX = 11
POST_CLIMAX_BREAK_WIDE_IDX = 32


def _two_break_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [r[1] for r in _ROWS], "high": [r[2] for r in _ROWS],
            "low": [r[3] for r in _ROWS], "close": [r[4] for r in _ROWS],
        }
    )


def _no_cvd_no_rsi_cfg() -> AppConfig:
    cfg = AppConfig()
    return cfg.model_copy(
        update={"cvd": cfg.cvd.model_copy(update={"enabled": False}), "rsi": cfg.rsi.model_copy(update={"enabled": False})}
    )


def _empty_cvd_slice(n: int) -> pd.DataFrame:
    return pd.DataFrame({"cvd_rolling": [np.nan] * n, "cvd_slope": [np.nan] * n, "open_interest": [np.nan] * n, "rsi": [np.nan] * n})


def test_wide_structure_latches_onto_the_early_irrelevant_break():
    """Sanity check on the fixture itself: confirms the OLD bug is real on
    this data before proving the fix - the one-time break latch in
    compute_structure means the wide, whole-series computation finds ONLY
    the early break and never even considers the later, real one."""
    df = _two_break_ohlc()
    cfg = _no_cvd_no_rsi_cfg()
    atr = struct_feat.atr(df, cfg.stops.atr_length).bfill().fillna(df["high"] - df["low"])
    wide_structure = struct_feat.compute_structure(df, atr, cfg.structure)
    break_bars = wide_structure.index[wide_structure["structure_break_bar"]].tolist()
    assert break_bars == [EARLY_BREAK_WIDE_IDX]


def test_post_pump_inputs_finds_the_post_climax_break_not_the_early_one():
    df = _two_break_ohlc()
    cvd = _empty_cvd_slice(len(df))
    cfg = _no_cvd_no_rsi_cfg()

    result = _build_post_pump_inputs(df, cvd, CLIMAX_IDX, 300_000, cfg)

    assert result.climax_idx == CLIMAX_IDX
    assert result.structure_start_idx == CLIMAX_IDX - cfg.structure.pivot_left_bars

    break_local_idxs = result.structure.index[result.structure["structure_break_bar"]].tolist()
    assert len(break_local_idxs) == 1
    break_wide_idx = break_local_idxs[0] + result.structure_start_idx
    assert break_wide_idx == POST_CLIMAX_BREAK_WIDE_IDX

    broken_row = result.structure.iloc[break_local_idxs[0]]
    assert broken_row["broken_level"] == pytest.approx(118.0)
    assert broken_row["pre_break_swing_high"] == pytest.approx(121.0)


def test_post_pump_inputs_structure_start_idx_never_negative_near_series_start():
    """climax_idx near 0 must clamp structure_start_idx to 0, not go negative."""
    df = _two_break_ohlc().iloc[:5].reset_index(drop=True)
    cvd = _empty_cvd_slice(len(df))
    cfg = _no_cvd_no_rsi_cfg()
    result = _build_post_pump_inputs(df, cvd, 0, 300_000, cfg)
    assert result.structure_start_idx == 0
