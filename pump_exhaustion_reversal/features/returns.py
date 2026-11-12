"""Return / volatility / BTC-adjusted abnormal-return features (spec section 7).

Every function is backward-looking only (pandas rolling/shift never peek
forward), so the exact same code path is safe for both the historical event
study/backtest and a future live scanner - no separate "replay-safe" variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def log_return(close: pd.Series, periods: int) -> pd.Series:
    return np.log(close / close.shift(periods))


def realized_vol(bar_log_returns: pd.Series, lookback: int) -> pd.Series:
    return bar_log_returns.rolling(lookback, min_periods=max(2, lookback // 2)).std()


def return_zscore(window_log_return: pd.Series, bar_realized_vol: pd.Series, window_bars: int) -> pd.Series:
    """log_return_w / max(realized_vol_w * sqrt(window_minutes), eps) - spec section 7."""
    denom = (bar_realized_vol * np.sqrt(window_bars)).clip(lower=EPS)
    return window_log_return / denom


def rolling_beta(coin_bar_return: pd.Series, btc_bar_return: pd.Series, lookback: int) -> pd.Series:
    cov = coin_bar_return.rolling(lookback, min_periods=max(2, lookback // 2)).cov(btc_bar_return)
    var = btc_bar_return.rolling(lookback, min_periods=max(2, lookback // 2)).var()
    return cov / var.replace(0, np.nan)


def abnormal_return(coin_window_return: pd.Series, btc_window_return: pd.Series, beta: pd.Series) -> pd.Series:
    """abnormal_return = coin_return - beta_to_btc * btc_return - spec section 7."""
    return coin_window_return - beta * btc_window_return


def bars_for_window(window_ms: int, bar_interval_ms: int) -> int:
    return max(round(window_ms / bar_interval_ms), 1)


def compute_return_features(
    coin_close: pd.Series,
    btc_close: pd.Series,
    bar_interval_ms: int,
    windows_ms: dict[str, int],
    beta_lookback_days: float,
) -> pd.DataFrame:
    """Per spec section 7: multi-window abnormal (BTC-adjusted) return z-scores.

    coin_close / btc_close must share the same index (same bar timestamps,
    already aligned/forward-filled by the caller). Beta is estimated on
    overlapping 1h log-returns sampled at the base bar frequency (spec says
    "rolling по 1h/4h returns за 14-30 дней") rather than on raw bar-to-bar
    returns, which would be far noisier at 1m/5m granularity.

    Returns one row per input bar with columns: beta_to_btc, and for each
    key in windows_ms: return_{key}, abnormal_return_{key}, return_zscore_{key}.
    """
    hourly_periods = bars_for_window(3_600_000, bar_interval_ms)
    coin_hourly_ret = log_return(coin_close, hourly_periods)
    btc_hourly_ret = log_return(btc_close, hourly_periods)
    beta_lookback_bars = bars_for_window(beta_lookback_days * 24 * 3_600_000, bar_interval_ms)
    beta = rolling_beta(coin_hourly_ret, btc_hourly_ret, beta_lookback_bars)

    bar_return = log_return(coin_close, 1)
    bar_vol = realized_vol(bar_return, lookback=max(hourly_periods * 4, 20))

    out = pd.DataFrame(index=coin_close.index)
    out["beta_to_btc"] = beta
    for label, window_ms in windows_ms.items():
        periods = bars_for_window(window_ms, bar_interval_ms)
        coin_ret = log_return(coin_close, periods)
        btc_ret = log_return(btc_close, periods)
        out[f"return_{label}"] = coin_ret
        out[f"abnormal_return_{label}"] = abnormal_return(coin_ret, btc_ret, beta)
        out[f"return_zscore_{label}"] = return_zscore(coin_ret, bar_vol, periods)
    return out
