"""Funding-rate context features (spec sections 4.4, 11.9). Context only -
never a standalone entry trigger, per spec: "Funding — это context, не
standalone trigger" / "Не разрешать вход только по funding."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def funding_percentile(funding_rate: pd.Series, lookback: int) -> pd.Series:
    """Rolling percentile rank (0-100) of the current funding rate vs its own history."""
    return funding_rate.rolling(lookback, min_periods=max(10, lookback // 4)).rank(pct=True) * 100


def funding_zscore(funding_rate: pd.Series, lookback: int) -> pd.Series:
    """Current bar vs the PRIOR lookback bars' mean/std - see
    features/volume.py:volume_zscore for why the current bar is excluded
    from its own baseline."""
    prior = funding_rate.shift(1)
    mean = prior.rolling(lookback, min_periods=max(10, lookback // 4)).mean()
    std = prior.rolling(lookback, min_periods=max(10, lookback // 4)).std()
    return (funding_rate - mean) / std.replace(0, np.nan)


def funding_heat_flags(percentile: pd.Series, overheated_pct: float, extreme_pct: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_overheated": percentile >= overheated_pct,
            "funding_extreme": percentile >= extreme_pct,
        },
        index=percentile.index,
    )
