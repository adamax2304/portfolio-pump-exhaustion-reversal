"""Volume anomaly + volume-decay features (spec sections 7-8, 11.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_zscore(volume: pd.Series, lookback: int) -> pd.Series:
    """Current bar's volume vs the PRIOR lookback bars' mean/std (shift(1)
    before rolling) - deliberately excludes the current bar from its own
    baseline. Including it (a naive .rolling().mean()/.std() on the same
    series being scored) badly dampens the z-score of exactly the spikes
    this is meant to catch: a single 30x volume bar pulls its own rolling
    mean and std up so much that its z-score comes out under 5 instead of
    the 50+ a genuine 30x-vs-baseline spike should show. Confirmed against
    real Bybit/Binance data during the Stage 1 pilot run, not just in theory.
    """
    prior = volume.shift(1)
    mean = prior.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
    std = prior.rolling(lookback, min_periods=max(2, lookback // 2)).std()
    return (volume - mean) / std.replace(0, np.nan)


def realized_vol_ratio(current_vol: pd.Series, baseline_vol: pd.Series) -> pd.Series:
    """Volatility-expansion ratio: current realized vol / longer-lookback baseline vol."""
    return current_vol / baseline_vol.replace(0, np.nan)


def volume_decay_ratio(current_volume: pd.Series, pump_peak_volume: float) -> pd.Series:
    """current_5m_volume / max_5m_volume_during_pump - spec section 11.4.

    `pump_peak_volume` is a scalar fixed once the pump event's peak bar is
    known (event-relative, not a rolling window) - callers pass the event's
    recorded peak, not a global max.
    """
    if pump_peak_volume is None or pd.isna(pump_peak_volume) or pump_peak_volume == 0:
        return pd.Series(np.nan, index=current_volume.index)
    return current_volume / pump_peak_volume


def volume_decay_flags(decay_ratio: pd.Series, threshold: float, strong_threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "volume_decayed": decay_ratio <= threshold,
            "volume_strongly_decayed": decay_ratio <= strong_threshold,
        },
        index=decay_ratio.index,
    )
