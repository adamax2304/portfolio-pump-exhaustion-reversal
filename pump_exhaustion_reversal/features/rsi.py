"""RSI + bearish divergence - low-weight confirmation only (spec section 11.10,
max_score_weight=2). Not allowed to trigger an entry on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss  # avg_loss==0 -> +inf (correctly saturates RSI at 100 below)
    return 100 - (100 / (1 + rs))


def rsi_bearish_divergence(high: pd.Series, rsi_values: pd.Series, lookback_bars: int) -> pd.Series:
    """True where price prints a new high over lookback_bars while RSI does not
    (i.e. RSI's prior rolling max exceeds the current RSI reading) - a classic
    bearish divergence flag, evaluated bar-by-bar with no forward lookahead.
    """
    rolling_price_max = high.rolling(lookback_bars, min_periods=1).max()
    rolling_rsi_max_prior = rsi_values.shift(1).rolling(lookback_bars, min_periods=1).max()
    price_new_high = high >= rolling_price_max
    rsi_not_new_high = rsi_values < rolling_rsi_max_prior
    return (price_new_high & rsi_not_new_high).fillna(False)
