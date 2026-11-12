"""Open-interest features (spec sections 4.3, 11.6).

OI history depth is exchange-limited (Binance ~30d retention on
openInterestHist, Bybit similarly shallow - see connectors/binance.py,
connectors/bybit.py, project plan). Older bars will simply have NaN OI:
callers must treat missing OI as missing data (lower confidence), never as
"open interest is zero".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def oi_change(oi: pd.Series, periods: int) -> pd.Series:
    return oi.diff(periods)


def oi_pct_change(oi: pd.Series, periods: int) -> pd.Series:
    return oi.pct_change(periods)


def oi_zscore(oi: pd.Series, lookback: int) -> pd.Series:
    """Current bar vs the PRIOR lookback bars' mean/std - see
    features/volume.py:volume_zscore for why the current bar is excluded
    from its own baseline (confirmed dampening a real spike's z-score
    during the Stage 1 pilot run)."""
    prior = oi.shift(1)
    mean = prior.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
    std = prior.rolling(lookback, min_periods=max(2, lookback // 2)).std()
    return (oi - mean) / std.replace(0, np.nan)


def oi_slope(oi: pd.Series, periods: int) -> pd.Series:
    return oi.diff(periods) / periods


def detect_oi_rollover(
    price_close: pd.Series,
    oi: pd.Series,
    lookback_bars: int,
    min_oi_drop_pct_after_peak: float,
    danger_oi_rise_pct_on_drop: float,
) -> pd.DataFrame:
    """spec section 11.6. `oi_rollover` fires once OI has dropped
    min_oi_drop_pct_after_peak% off its lookback_bars-window peak - a single
    proxy standing in for the several OR'd scenarios spec describes (price up
    & OI flat/falling, price flat & OI falling, price down & OI falling all
    look the same on the OI side: it stopped supporting new positions).

    `oi_danger_new_shorts` flags the opposite, genuinely dangerous case (spec:
    "price down, OI aggressively up = new shorts entering") - exhaustion
    scoring must NOT reward this even though price is falling, since it can
    fuel a squeeze back up.
    """
    rolling_peak = oi.rolling(lookback_bars, min_periods=1).max()
    drop_pct_from_peak = (rolling_peak - oi) / rolling_peak.replace(0, np.nan) * 100
    price_change = price_close.diff(lookback_bars)
    oi_change_pct = oi.pct_change(lookback_bars) * 100

    rollover = drop_pct_from_peak >= min_oi_drop_pct_after_peak
    danger = (price_change < 0) & (oi_change_pct >= danger_oi_rise_pct_on_drop)

    return pd.DataFrame(
        {
            "oi_peak_in_lookback": rolling_peak,
            "oi_drop_pct_from_peak": drop_pct_from_peak,
            "oi_rollover": rollover.fillna(False),
            "oi_danger_new_shorts": danger.fillna(False),
        },
        index=oi.index,
    )
