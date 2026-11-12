"""Anchored VWAP (spec section 11.8): pump_start / breakout_candle / climax_candle."""

from __future__ import annotations

import numpy as np
import pandas as pd


def anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, anchor_idx: int) -> pd.Series:
    """VWAP computed from anchor_idx forward only (NaN before the anchor)."""
    typical_price = (high + low + close) / 3
    result = pd.Series(np.nan, index=close.index)
    tp_vol_cum = (typical_price * volume).iloc[anchor_idx:].cumsum()
    vol_cum = volume.iloc[anchor_idx:].cumsum().replace(0, np.nan)
    result.iloc[anchor_idx:] = (tp_vol_cum / vol_cum).to_numpy()
    return result


def avwap_loss(close: pd.Series, avwap: pd.Series) -> pd.Series:
    """True where close is below the anchored VWAP ("lost" that AVWAP as support)."""
    return (close < avwap).fillna(False)


def compute_avwap_features(df: pd.DataFrame, anchors: dict[str, int | None]) -> pd.DataFrame:
    """anchors: {"pump_start": idx, "breakout_candle": idx, "climax_candle": idx},
    any value may be None if that anchor isn't known yet for the current event -
    that anchor's columns are simply omitted, not filled with a guess.
    Emits avwap_{name} and avwap_loss_{name} per known anchor.
    """
    out = pd.DataFrame(index=df.index)
    for name, idx in anchors.items():
        if idx is None:
            continue
        vwap = anchored_vwap(df["high"], df["low"], df["close"], df["volume"], idx)
        out[f"avwap_{name}"] = vwap
        out[f"avwap_loss_{name}"] = avwap_loss(df["close"], vwap)
    return out
