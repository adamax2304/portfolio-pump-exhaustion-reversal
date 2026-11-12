"""Cumulative volume delta (spec sections 4.2, 11.5).

Source priority, highest confidence first:
  1. real public trades with a known taker side (connectors ccxt_fallback.fetch_trades_window) - "high"
  2. Binance's native taker_buy_base_volume field riding along on OHLCV klines - "high"
     (real exchange-computed taker-side data, just bar-aggregated rather than tick-level)
  3. a close-vs-open tick-rule proxy - "low" (spec 4.2: "если side недоступен - cvd confidence low")
Bybit klines never carry taker_buy_base_volume, so outside an event-anchored
trades fetch, Bybit CVD is always the "low"-confidence proxy - see project plan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_delta_from_taker_buy_volume(volume: pd.Series, taker_buy_base_volume: pd.Series) -> pd.Series:
    """buy_volume=taker_buy_base_volume, sell_volume=volume-buy_volume, delta=buy-sell=2*buy-volume."""
    return 2 * taker_buy_base_volume - volume


def compute_delta_proxy_from_ohlc(open_: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Crude tick-rule proxy when no real taker-side data exists at all: sign(close-open)*volume."""
    direction = np.sign(close - open_)
    return volume * direction


def aggregate_trades_to_bar_delta(
    trades: pd.DataFrame, bar_timestamps: pd.Series, bar_interval_ms: int
) -> pd.Series:
    """Buckets event-anchored public trades (columns timestamp, amount, side)
    into the OHLCV frame's bars and returns buy_volume - sell_volume per bar.
    Trades with unknown side (side is None) contribute 0 to delta - excluded,
    not guessed."""
    if trades.empty:
        return pd.Series(np.nan, index=bar_timestamps.index)
    edges = np.append(bar_timestamps.to_numpy(), bar_timestamps.iloc[-1] + bar_interval_ms)
    buy_amt = trades["amount"].where(trades["side"] == "buy", 0.0)
    sell_amt = trades["amount"].where(trades["side"] == "sell", 0.0)
    signed = buy_amt - sell_amt
    bucket = pd.cut(trades["timestamp"], bins=edges, right=False, labels=False)
    grouped = signed.groupby(bucket).sum()  # groupby drops NaN keys (trades outside [start, end))
    result = pd.Series(0.0, index=range(len(bar_timestamps)))
    result.loc[grouped.index.astype(int)] = grouped.to_numpy()
    result.index = bar_timestamps.index
    return result


def anchored_cvd(delta: pd.Series, anchor_idx: int) -> pd.Series:
    """Cumulative delta reset to 0 at anchor_idx (e.g. the pump-start bar)."""
    result = pd.Series(np.nan, index=delta.index)
    result.iloc[anchor_idx:] = delta.iloc[anchor_idx:].cumsum().to_numpy()
    return result


def cvd_bearish_divergence(high: pd.Series, cvd: pd.Series, lookback_bars: int, tolerance: float) -> pd.Series:
    """price_hh & cvd fails to confirm (spec 11.5): current CVD high must clear
    the prior CVD high by more than `tolerance` to NOT count as divergence."""
    rolling_price_max = high.rolling(lookback_bars, min_periods=1).max()
    rolling_cvd_max_prior = cvd.shift(1).rolling(lookback_bars, min_periods=1).max()
    price_new_high = high >= rolling_price_max
    cvd_not_confirming = cvd <= rolling_cvd_max_prior * (1 + tolerance)
    return (price_new_high & cvd_not_confirming).fillna(False)


def cvd_divergence_magnitude(high: pd.Series, cvd: pd.Series, lookback_bars: int) -> pd.Series:
    """Continuous companion to cvd_bearish_divergence: on bars where price makes
    a new rolling high, how far CVD's current value sits below its own PRIOR
    rolling high, as a fraction of that prior high's absolute value (positive =
    divergence, larger = stronger). NaN where price isn't making a new high
    (the concept doesn't apply there) or the prior CVD high is ~0 (fraction
    undefined) - not a confident 0, matching this codebase's missing-data
    convention (see scoring.py's renormalization)."""
    rolling_price_max = high.rolling(lookback_bars, min_periods=1).max()
    rolling_cvd_max_prior = cvd.shift(1).rolling(lookback_bars, min_periods=1).max()
    price_new_high = high >= rolling_price_max
    gap_fraction = (rolling_cvd_max_prior - cvd) / rolling_cvd_max_prior.abs()
    return gap_fraction.where(price_new_high & (rolling_cvd_max_prior.abs() > 0))


def cvd_rollover(cvd_slope: pd.Series) -> pd.Series:
    """CVD slope flips from non-negative to negative - a simple rollover flag."""
    prev_non_negative = cvd_slope.shift(1) >= 0
    now_negative = cvd_slope < 0
    return (prev_non_negative & now_negative).fillna(False)


def compute_cvd_features(
    df: pd.DataFrame,
    lookback_bars: int,
    trades: pd.DataFrame | None = None,
    bar_interval_ms: int | None = None,
) -> pd.DataFrame:
    """df needs columns open, close, volume, and optionally taker_buy_base_volume.

    Returns cvd_delta, cvd_rolling (rolling-window cumulative delta over
    lookback_bars - matches spec's "rolling CVD"), cvd_slope, and a single
    cvd_confidence label ('high'|'low') applying to the whole frame (the
    decision of which source to use is made once, not per-bar).
    """
    if trades is not None and not trades.empty and trades["side"].isin(["buy", "sell"]).any():
        if bar_interval_ms is None:
            raise ValueError("bar_interval_ms is required when trades are supplied")
        delta = aggregate_trades_to_bar_delta(trades, df["timestamp"], bar_interval_ms)
        confidence = "high"
    elif "taker_buy_base_volume" in df.columns and df["taker_buy_base_volume"].notna().any():
        delta = compute_delta_from_taker_buy_volume(df["volume"], df["taker_buy_base_volume"])
        confidence = "high"
    else:
        delta = compute_delta_proxy_from_ohlc(df["open"], df["close"], df["volume"])
        confidence = "low"

    cvd_rolling = delta.rolling(lookback_bars, min_periods=1).sum()
    slope_step = max(lookback_bars // 4, 1)
    slope = cvd_rolling.diff(slope_step) / slope_step

    out = pd.DataFrame(index=df.index)
    out["cvd_delta"] = delta
    out["cvd_rolling"] = cvd_rolling
    out["cvd_slope"] = slope
    out["cvd_confidence"] = confidence
    return out
