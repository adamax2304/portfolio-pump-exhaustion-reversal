"""Liquidation proxy (spec sections 4.5, 11.7).

No free historical liquidation feed exists on Binance or Bybit (only
real-time WS liquidation streams - no historical REST endpoint), so
ProxyLiquidationProvider is the only LiquidationDataProvider mode actually
implemented in Stage 1 (see connectors/base.py for the interface and the
"real_liquidation_data" / "paid_provider_placeholder" stubs). It estimates
short-liquidation-like activity from OI-drop + price-spike + volume-spike
coinciding: a sudden OI drop alongside an up-spike in price and volume looks
like short covering/liquidation, not organic long entries. Always
confidence="proxy" - never conflate this with a real liquidation feed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..connectors.base import LIQUIDATION_COLUMNS, LiquidationDataProvider
from . import returns as returns_feat
from . import volume as volume_feat


def estimate_liquidation_proxy(df: pd.DataFrame, oi: pd.Series, lookback_bars: int) -> pd.DataFrame:
    """df needs high, low, close, volume; oi must share df's index (already
    aligned/forward-filled by the caller - OI has its own, sparser timestamps).
    Returns liq_proxy_intensity (raw heuristic, >=0) and short_liq_proxy_zscore.

    Both are NaN (missing), not a confirmed 0, whenever OI isn't actually
    dropping: the proxy's whole precondition (short covering) isn't met, so
    it has no reading to offer, as opposed to having measured genuinely low
    liquidation activity. Feeding a confident 0 into a 20%-weighted pump-score
    component for the (common) case of a leveraged-long/organic pump - where
    OI rises, not falls - made pump_score essentially unreachable for
    anything but a textbook short squeeze, confirmed against real 6-month
    Binance/Bybit data where zero symbols reached min_pump_score despite many
    100%+ moves (see project notes). NaN lets scoring.py renormalize around
    it instead (spec section 8), same as a genuinely missing OI feed.
    """
    price_ret = returns_feat.log_return(df["close"], 1)
    vol_z = volume_feat.volume_zscore(df["volume"], lookback_bars)
    oi_drop_frac = (-oi.diff(1) / oi.shift(1).replace(0, np.nan)).clip(lower=0)

    raw_intensity = oi_drop_frac * 100 * price_ret.clip(lower=0) * vol_z.clip(lower=0)
    raw_intensity = raw_intensity.where(oi_drop_frac > 0)

    prior = raw_intensity.shift(1)
    mean = prior.rolling(lookback_bars, min_periods=max(2, lookback_bars // 2)).mean()
    std = prior.rolling(lookback_bars, min_periods=max(2, lookback_bars // 2)).std()
    z = (raw_intensity - mean) / std.replace(0, np.nan)

    return pd.DataFrame(
        {"liq_proxy_intensity": raw_intensity, "short_liq_proxy_zscore": z},
        index=df.index,
    )


def liquidation_depletion_ratio(intensity: pd.Series, peak_intensity: float) -> pd.Series:
    """current_short_liq / peak_short_liq_during_pump - spec section 11.7."""
    if peak_intensity is None or pd.isna(peak_intensity) or peak_intensity == 0:
        return pd.Series(np.nan, index=intensity.index)
    return intensity / peak_intensity


LoaderFn = Callable[[str, int, int], tuple[pd.DataFrame, pd.Series]]


class ProxyLiquidationProvider(LiquidationDataProvider):
    """mode="proxy_mode". `load_ohlcv_and_oi(symbol, start_ms, end_ms)` must
    return (ohlcv_df, oi_series_aligned_to_ohlcv_df.index) - callers in
    event_study/exhaustion_detector already hold this data, so this class
    exists mainly to satisfy the LiquidationDataProvider interface contract
    (spec section 4.5); most Stage-1 code calls estimate_liquidation_proxy()
    directly instead of going through get_liquidations().
    """

    mode = "proxy_mode"

    def __init__(self, load_ohlcv_and_oi: LoaderFn, lookback_bars: int = 20, zscore_threshold: float = 3.0):
        self._load = load_ohlcv_and_oi
        self.lookback_bars = lookback_bars
        self.zscore_threshold = zscore_threshold

    def get_liquidations(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        df, oi = self._load(symbol, start_ms, end_ms)
        if df.empty:
            return pd.DataFrame(columns=LIQUIDATION_COLUMNS)
        proxy = estimate_liquidation_proxy(df, oi, self.lookback_bars)
        mask = (proxy["short_liq_proxy_zscore"] >= self.zscore_threshold).to_numpy()
        rows = pd.DataFrame(
            {
                "timestamp": df["timestamp"].to_numpy(),
                "side": "short",
                "notional_usd_estimate": np.nan,
                "confidence": "proxy",
            }
        )
        return rows.loc[mask].reset_index(drop=True)[LIQUIDATION_COLUMNS]
