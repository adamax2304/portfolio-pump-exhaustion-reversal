from __future__ import annotations

import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest.simulator import _post_break_invalidation_reason
from pump_exhaustion_reversal.config import AppConfig


def _base_structure(n: int, broken_at: int | None, broken_level: float, pre_break_high: float) -> pd.DataFrame:
    structure_break_bar = [False] * n
    broken_level_col = [float("nan")] * n
    pre_break_high_col = [float("nan")] * n
    bars_since_break = [float("nan")] * n
    swing_low_level = [float("nan")] * n
    swing_low_label = [None] * n
    if broken_at is not None:
        structure_break_bar[broken_at] = True
        for i in range(broken_at, n):
            broken_level_col[i] = broken_level
            pre_break_high_col[i] = pre_break_high
            bars_since_break[i] = i - broken_at
    return pd.DataFrame(
        {
            "structure_break_bar": structure_break_bar,
            "broken_level": broken_level_col,
            "pre_break_swing_high": pre_break_high_col,
            "bars_since_structure_break": bars_since_break,
            "swing_low_level": swing_low_level,
            "swing_low_label": swing_low_label,
        }
    )


def _flat_ohlc(n: int, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"open": [price] * n, "high": [price] * n, "low": [price] * n, "close": [price] * n})


def test_no_break_yet_returns_none():
    n = 10
    structure = _base_structure(n, broken_at=None, broken_level=100.0, pre_break_high=110.0)
    df_slice = _flat_ohlc(n, 105.0)
    assert _post_break_invalidation_reason(5, structure, df_slice, AppConfig()) is None


def test_new_pump_high_invalidates():
    n = 10
    broken_at = 3
    structure = _base_structure(n, broken_at=broken_at, broken_level=100.0, pre_break_high=110.0)
    df_slice = _flat_ohlc(n, 95.0)
    df_slice.loc[7, "high"] = 111.0  # exceeds pre_break_swing_high (110)
    result = _post_break_invalidation_reason(7, structure, df_slice, AppConfig())
    assert result is not None
    code, text = result
    assert code == "new_pump_high"
    assert "new pump high" in text


def test_price_below_pre_break_high_does_not_invalidate():
    n = 10
    broken_at = 3
    structure = _base_structure(n, broken_at=broken_at, broken_level=100.0, pre_break_high=110.0)
    df_slice = _flat_ohlc(n, 95.0)
    df_slice.loc[7, "high"] = 105.0  # still well below pre_break_swing_high (110)
    assert _post_break_invalidation_reason(7, structure, df_slice, AppConfig()) is None


def test_bullish_reformation_invalidates():
    n = 10
    broken_at = 3
    structure = _base_structure(n, broken_at=broken_at, broken_level=100.0, pre_break_high=110.0)
    structure.loc[7, "swing_low_level"] = 102.0  # above broken_level (100) -> a real HL formed
    structure.loc[7, "swing_low_label"] = "HL"
    df_slice = _flat_ohlc(n, 95.0)  # price itself stays weak, but structure says HL formed
    result = _post_break_invalidation_reason(7, structure, df_slice, AppConfig())
    assert result is not None
    code, text = result
    assert code == "bullish_hl_reformation"
    assert "higher low" in text


def test_hl_below_broken_level_does_not_invalidate():
    n = 10
    broken_at = 3
    structure = _base_structure(n, broken_at=broken_at, broken_level=100.0, pre_break_high=110.0)
    structure.loc[7, "swing_low_level"] = 90.0  # HL but still BELOW the broken level - no bullish reformation
    structure.loc[7, "swing_low_label"] = "HL"
    df_slice = _flat_ohlc(n, 95.0)
    assert _post_break_invalidation_reason(7, structure, df_slice, AppConfig()) is None


def _cfg_with_confirmation_bars(n: int) -> AppConfig:
    return AppConfig(structure={"close_confirmation_bars": n})


def test_strong_reclaim_invalidates():
    cfg = _cfg_with_confirmation_bars(3)
    ccb = cfg.structure.close_confirmation_bars
    n = 10
    broken_at = 3
    structure = _base_structure(n, broken_at=broken_at, broken_level=100.0, pre_break_high=110.0)
    df_slice = _flat_ohlc(n, 95.0)
    zone_top = 100.0 * (1 + cfg.failed_reclaim.reclaim_zone_bps / 10_000)
    for i in range(8 - ccb + 1, 9):
        df_slice.loc[i, "close"] = zone_top + 1.0  # 3 consecutive bars above the reclaim zone
    result = _post_break_invalidation_reason(8, structure, df_slice, cfg)
    assert result is not None
    code, text = result
    assert code == "strong_reclaim"
    assert "reclaim" in text


def test_brief_poke_into_zone_without_confirmation_does_not_invalidate():
    cfg = _cfg_with_confirmation_bars(3)
    n = 10
    broken_at = 3
    structure = _base_structure(n, broken_at=broken_at, broken_level=100.0, pre_break_high=110.0)
    df_slice = _flat_ohlc(n, 95.0)
    zone_top = 100.0 * (1 + cfg.failed_reclaim.reclaim_zone_bps / 10_000)
    df_slice.loc[7, "close"] = zone_top + 1.0  # one bar only
    df_slice.loc[8, "close"] = 95.0  # immediately fails back below - not a "strong" (3-bar) reclaim
    assert _post_break_invalidation_reason(8, structure, df_slice, cfg) is None
