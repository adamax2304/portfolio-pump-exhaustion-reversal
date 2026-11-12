from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import labels


def _ohlcv(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    # rows: (open, high, low, close)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_reversal_wins_when_only_reversal_barrier_touched():
    ohlcv = _ohlcv([(100, 100, 100, 100), (100, 100, 89, 95)])  # low=89 -> -11% from 100, past a 10% reversal barrier
    result = labels.compute_competing_outcome_label(ohlcv, anchor_idx=0, reversal_pct=0.10, continuation_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "REVERSAL"


def test_continuation_wins_when_only_continuation_barrier_touched():
    ohlcv = _ohlcv([(100, 100, 100, 100), (100, 106, 100, 103)])  # high=106 -> +6% from 100, past a 5% continuation barrier
    result = labels.compute_competing_outcome_label(ohlcv, anchor_idx=0, reversal_pct=0.10, continuation_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "CONTINUATION"


def test_same_candle_ambiguity_continuation_wins_ties():
    # one bar touches BOTH the +5% continuation barrier (106) and the -10% reversal barrier (90)
    ohlcv = _ohlcv([(100, 100, 100, 100), (100, 106, 88, 100)])
    result = labels.compute_competing_outcome_label(ohlcv, anchor_idx=0, reversal_pct=0.10, continuation_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "CONTINUATION"


def test_neutral_when_neither_barrier_touched():
    ohlcv = _ohlcv([(100, 100, 100, 100)] * 6)
    result = labels.compute_competing_outcome_label(ohlcv, anchor_idx=0, reversal_pct=0.10, continuation_pct=0.05, horizon_bars=5)
    assert result["outcome"] == "NEUTRAL"


def test_label_snapshot_across_families_covers_primary_fast_deep():
    n = 300
    rng = np.random.default_rng(2)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})
    result = labels.label_snapshot_across_families(ohlcv, anchor_idx=50, bar_interval_ms=300_000)
    assert set(result) == {"primary", "fast", "deep"}
    for r in result.values():
        assert r["outcome"] in ("REVERSAL", "CONTINUATION", "NEUTRAL")


def test_peak_hazard_positive_when_no_new_high_and_genuine_drawdown():
    # event_high_so_far = 110; window never exceeds 110*1.02=112.2, and drops
    # from a window-high of 108 down to 98 -> drawdown = (108-98)/108 = 9.3% >= 7.5%.
    ohlcv = _ohlcv([(100, 100, 100, 100), (105, 108, 105, 107), (107, 108, 98, 99), (99, 100, 95, 96)])
    result = labels.compute_peak_hazard_label(ohlcv, anchor_idx=0, event_high_so_far=110.0, bar_interval_ms=300_000, horizon_hours=1.0)
    assert result["peak_hazard"] is True


def test_peak_hazard_negative_when_new_high_extends_meaningfully():
    # window DOES exceed 110*1.02=112.2 (reaches 120) -> disqualified regardless of any later drawdown.
    ohlcv = _ohlcv([(100, 100, 100, 100), (105, 120, 105, 118), (118, 118, 100, 105)])
    result = labels.compute_peak_hazard_label(ohlcv, anchor_idx=0, event_high_so_far=110.0, bar_interval_ms=300_000, horizon_hours=1.0)
    assert result["peak_hazard"] is False


def test_peak_hazard_negative_when_no_genuine_drawdown():
    # stays flat/slightly down but never gives a 7.5%+ drawdown from its own window peak.
    ohlcv = _ohlcv([(100, 100, 100, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
    result = labels.compute_peak_hazard_label(ohlcv, anchor_idx=0, event_high_so_far=110.0, bar_interval_ms=300_000, horizon_hours=1.0)
    assert result["peak_hazard"] is False


def test_high_anchored_label_fixes_positional_bias():
    # a snapshot already ~8.3% below the event high (120, so around 110):
    # the high-anchored drawdown barrier is a FIXED 108 (10% below 120)
    # regardless of the snapshot's own current price - reversal fires as
    # soon as price crosses 108, not after a fresh -10% from its own level.
    event_high_so_far = 120.0
    ohlcv = _ohlcv([(110, 110, 110, 110), (110, 110, 109, 109.5), (109.5, 109.5, 105, 106)])
    result = labels.compute_high_anchored_label(ohlcv, anchor_idx=0, event_high_so_far=event_high_so_far, horizon_bars=5)
    assert result["outcome"] == "REVERSAL"
    assert result["bars"] == 2  # bar1's low=109 doesn't cross drawdown_price=108 yet; bar2's low=105 does


def test_high_anchored_label_continuation_when_extension_hit_first():
    ohlcv = _ohlcv([(100, 100, 100, 100), (100, 116, 100, 110)])  # high=116 >= 110*1.05=115.5 -> CONTINUATION
    result = labels.compute_high_anchored_label(ohlcv, anchor_idx=0, event_high_so_far=110.0, horizon_bars=5)
    assert result["outcome"] == "CONTINUATION"


def test_high_anchored_label_none_when_already_resolved_at_anchor_bar():
    # anchor bar's own low is ALREADY <= the drawdown price (event_high=120, drawdown_price=108, anchor low=105)
    ohlcv = _ohlcv([(110, 111, 105, 106), (106, 108, 104, 105)])
    result = labels.compute_high_anchored_label(ohlcv, anchor_idx=0, event_high_so_far=120.0, horizon_bars=5)
    assert result["outcome"] is None


def test_high_anchored_label_none_when_event_high_missing():
    ohlcv = _ohlcv([(100, 100, 100, 100), (100, 101, 99, 100)])
    result = labels.compute_high_anchored_label(ohlcv, anchor_idx=0, event_high_so_far=np.nan, horizon_bars=5)
    assert result["outcome"] is None


def test_peak_hazard_none_when_event_high_so_far_missing():
    ohlcv = _ohlcv([(100, 100, 100, 100), (100, 101, 99, 100)])
    result = labels.compute_peak_hazard_label(ohlcv, anchor_idx=0, event_high_so_far=np.nan, bar_interval_ms=300_000)
    assert result["peak_hazard"] is None
