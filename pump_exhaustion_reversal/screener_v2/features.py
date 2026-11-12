"""Phase 3A points 10, 12-15, 17-20 - the broader continuous feature set
(volume dynamics, OI dynamics, price x OI regime, funding, structure
deterioration, wick/rejection, volatility, pump age/stage, BTC/market
context). Point 21 (cross-exchange) is NOT built in this pass - it needs a
linked-event join (research/cross_exchange.py) that most single-exchange
snapshots won't have; deferred, disclosed in snapshot_data_quality.md rather
than silently skipped.

Same contract as pump_dynamics.py: every function takes df_slice/cvd_slice
(an event's post-pump-start WIDE-indexed OHLCV/feature slice) and returns a
DataFrame aligned to the same index, vectorized, causal (row i depends only
on rows [0, i]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import _PostPumpInputs
from ..features.structure import _bars_since

_EPS = 1e-9


def _bars_for(minutes: float, bar_interval_ms: int) -> int:
    return max(round(minutes * 60_000 / bar_interval_ms), 1)


def compute_volume_features(df_slice: pd.DataFrame, cvd_slice: pd.DataFrame, climax_idx_wide: int, bar_interval_ms: int) -> pd.DataFrame:
    """Point 10: continuous volume state, not just volume_decay's boolean.
    volume_per_1pct_price_extension: how much volume is now required to buy
    each additional 1% of price extension beyond the climax - the same
    "absorption" hypothesis as pump_dynamics.compute_efficiency_features,
    from the volume side specifically."""
    volume = df_slice["volume"]
    close = df_slice["close"]
    n = len(df_slice)
    climax_idx_wide = min(max(climax_idx_wide, 0), n - 1)
    climax_volume = float(volume.iloc[climax_idx_wide])
    climax_price = float(df_slice["high"].iloc[climax_idx_wide])

    b5, b15 = (_bars_for(m, bar_interval_ms) for m in (5, 15))
    out = pd.DataFrame(index=df_slice.index)
    out["volume_current_over_climax"] = volume / climax_volume if climax_volume else np.nan
    out["volume_5m_z"] = cvd_slice["volume_zscore"] if "volume_zscore" in cvd_slice.columns else np.nan
    out["volume_15m_mean_z"] = out["volume_5m_z"].rolling(b15, min_periods=1).mean()
    out["volume_slope"] = volume.diff(b5)
    out["volume_acceleration"] = out["volume_slope"].diff(b5)
    out["volume_peak_to_current_pct"] = (volume - climax_volume) / climax_volume if climax_volume else np.nan

    price_extension_pct_beyond_climax = (close - climax_price) / climax_price if climax_price else pd.Series(np.nan, index=df_slice.index)
    rolling_volume = volume.rolling(b5, min_periods=1).sum()
    out["volume_per_1pct_price_extension"] = rolling_volume / (price_extension_pct_beyond_climax.diff(b5).abs() * 100).replace(0, np.nan)

    is_new_high = df_slice["high"] > df_slice["high"].cummax().shift(1).fillna(-np.inf)
    volume_on_new_high = volume.where(is_new_high).ffill()
    prev_volume_on_new_high = volume.where(is_new_high).ffill().shift(1).where(is_new_high).ffill()
    out["volume_on_new_high"] = volume_on_new_high
    out["previous_volume_on_new_high"] = prev_volume_on_new_high
    return out


def compute_oi_features(cvd_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Point 12: continuous OI trajectory (accelerating -> slowing -> flat ->
    declining), not oi_rollover's boolean. Missing OI stays NaN throughout -
    never coerced to 0 (0 would silently mean "OI collapsed to zero", a
    materially different and almost always false statement)."""
    out = pd.DataFrame(index=cvd_slice.index)
    if "open_interest" not in cvd_slice.columns or not cvd_slice["open_interest"].notna().any():
        for col in ("oi_change_5m", "oi_change_15m", "oi_change_30m", "oi_change_60m", "oi_slope", "oi_acceleration", "oi_peak_so_far", "oi_drawdown_from_peak"):
            out[col] = np.nan
        return out

    oi = cvd_slice["open_interest"]
    b5, b15, b30, b60 = (_bars_for(m, bar_interval_ms) for m in (5, 15, 30, 60))
    out["oi_change_5m"] = oi.pct_change(b5)
    out["oi_change_15m"] = oi.pct_change(b15)
    out["oi_change_30m"] = oi.pct_change(b30)
    out["oi_change_60m"] = oi.pct_change(b60)
    out["oi_slope"] = oi.diff(b5)
    out["oi_acceleration"] = out["oi_slope"].diff(b5)
    peak = oi.cummax()
    out["oi_peak_so_far"] = peak
    out["oi_drawdown_from_peak"] = (peak - oi) / peak.replace(0, np.nan)
    return out


OI_REGIME_STRONG_DOWN_THRESHOLD = -0.15
OI_REGIME_MILD_DOWN_THRESHOLD = -0.05
OI_REGIME_MILD_UP_THRESHOLD = 0.05
OI_REGIME_STRONG_UP_THRESHOLD = 0.15


def compute_oi_regime_features(cvd_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Phase 3A.1 point 10: Phase 3A found oi_change_60m U-shaped (both
    strong OI growth AND strong OI decline showing elevated reversal rates
    vs a flat middle) - forcing it into one linear/monotonic "rule" would
    hide that shape. Buckets into 5 pre-registered regimes (thresholds fixed
    before looking at any regime-conditional result - never tuned per
    outcome) so downstream ranking can report P(resolution), P(reversal|
    resolved), P(continuation|resolved) PER BUCKET without assuming
    linearity or picking a side of the U as "the" signal."""
    out = pd.DataFrame(index=cvd_slice.index)
    if "open_interest" not in cvd_slice.columns or not cvd_slice["open_interest"].notna().any():
        out["oi_regime"] = None
        return out
    b60 = _bars_for(60, bar_interval_ms)
    oi_change_60m = cvd_slice["open_interest"].pct_change(b60)
    regime = np.select(
        [
            oi_change_60m < OI_REGIME_STRONG_DOWN_THRESHOLD,
            oi_change_60m < OI_REGIME_MILD_DOWN_THRESHOLD,
            oi_change_60m <= OI_REGIME_MILD_UP_THRESHOLD,
            oi_change_60m <= OI_REGIME_STRONG_UP_THRESHOLD,
            oi_change_60m > OI_REGIME_STRONG_UP_THRESHOLD,
        ],
        ["OI_STRONG_DOWN", "OI_MILD_DOWN", "OI_FLAT", "OI_MILD_UP", "OI_STRONG_UP"],
        default=None,
    )
    out["oi_regime"] = np.where(oi_change_60m.notna(), regime, None)
    return out


def compute_price_oi_regime_features(df_slice: pd.DataFrame, cvd_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Point 13: price x OI quadrant (up/up, up/down, down/up, down/down) as
    a categorical label PLUS the continuous magnitude behind it - regime is
    diagnostic labeling only here, not turned into a rule."""
    b15 = _bars_for(15, bar_interval_ms)
    price_change = df_slice["close"].pct_change(b15)
    out = pd.DataFrame(index=df_slice.index)
    if "open_interest" not in cvd_slice.columns or not cvd_slice["open_interest"].notna().any():
        out["price_oi_regime"] = None
        out["price_change_vs_oi_change"] = np.nan
        out["oi_per_price_extension"] = np.nan
        return out
    oi_change = cvd_slice["open_interest"].pct_change(b15)
    regime = np.select(
        [
            (price_change > 0) & (oi_change > 0), (price_change > 0) & (oi_change <= 0),
            (price_change <= 0) & (oi_change > 0), (price_change <= 0) & (oi_change <= 0),
        ],
        ["price_up_oi_up", "price_up_oi_down", "price_down_oi_up", "price_down_oi_down"],
        default=None,
    )
    out["price_oi_regime"] = np.where(price_change.notna() & oi_change.notna(), regime, None)
    out["price_change_vs_oi_change"] = price_change - oi_change
    out["oi_per_price_extension"] = oi_change / price_change.abs().replace(0, np.nan)
    return out


def compute_funding_features(cvd_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Point 14: funding as continuous secondary context, never a hard
    exhaustion condition."""
    b15 = _bars_for(15, bar_interval_ms)
    out = pd.DataFrame(index=cvd_slice.index)
    out["funding_current"] = cvd_slice["funding_rate"] if "funding_rate" in cvd_slice.columns else np.nan
    out["funding_percentile"] = cvd_slice["funding_percentile"] if "funding_percentile" in cvd_slice.columns else np.nan
    if "funding_rate" in cvd_slice.columns:
        rate = cvd_slice["funding_rate"]
        rolling_mean = rate.expanding(min_periods=1).mean()
        rolling_std = rate.expanding(min_periods=2).std()
        out["funding_zscore"] = (rate - rolling_mean) / rolling_std.replace(0, np.nan)
        out["funding_change"] = rate.diff(b15)
        out["funding_acceleration"] = out["funding_change"].diff(b15)
    else:
        out["funding_zscore"] = np.nan
        out["funding_change"] = np.nan
        out["funding_acceleration"] = np.nan
    return out


def compute_structure_deterioration_features(df_slice: pd.DataFrame, post_pump: _PostPumpInputs, atr_series: pd.Series) -> pd.DataFrame:
    """Point 15: structure state as continuous deterioration, not a hard
    required=True gate. Reuses post_pump.structure (already climax-anchored,
    causal - see backtest.simulator._build_post_pump_inputs) unchanged;
    adds the continuous companions on top. Aligned to df_slice's WIDE index
    (post_pump's own frames start at structure_start_idx, offset here)."""
    n = len(df_slice)
    structure = post_pump.structure
    s = post_pump.structure_start_idx
    out = pd.DataFrame(index=df_slice.index)

    swing_low = pd.Series(np.nan, index=df_slice.index)
    swing_low.iloc[s : s + len(structure)] = structure["swing_low_level"].to_numpy()
    swing_low = swing_low.ffill()
    close = df_slice["close"]
    out["distance_to_last_meaningful_hl_atr"] = (close - swing_low) / atr_series.replace(0, np.nan)
    out["depth_below_micro_hl"] = ((swing_low - close) / swing_low.replace(0, np.nan)).clip(lower=0)

    structure_break = pd.Series(False, index=df_slice.index)
    structure_break.iloc[s : s + len(structure)] = structure["structure_break_bar"].to_numpy()
    out["number_of_failed_high_attempts"] = structure_break.astype(float).cumsum()

    failed_reclaim = pd.Series(False, index=df_slice.index)
    if post_pump.failed_reclaim is not None:
        failed_reclaim.iloc[s : s + len(post_pump.failed_reclaim)] = post_pump.failed_reclaim.to_numpy()
    reclaim_high = df_slice["high"].where(failed_reclaim)
    broken_level = pd.Series(np.nan, index=df_slice.index)
    broken_level.iloc[s : s + len(structure)] = structure["broken_level"].to_numpy()
    out["reclaim_strength"] = ((reclaim_high - broken_level) / atr_series.replace(0, np.nan)).ffill().where(failed_reclaim.cummax())

    pre_break_high = pd.Series(np.nan, index=df_slice.index)
    pre_break_high.iloc[s : s + len(structure)] = structure["pre_break_swing_high"].to_numpy()
    out["lower_high_strength"] = ((pre_break_high - df_slice["high"]) / atr_series.replace(0, np.nan)).where(structure_break.cummax())

    deterioration = structure_break.astype(float).cumsum() + failed_reclaim.astype(float).cumsum()
    out["structure_deterioration_score"] = deterioration / deterioration.expanding().max().replace(0, np.nan)
    return out


def compute_wick_rejection_features(df_slice: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Point 17: continuous wick/rejection state - never a single shooting-
    star pattern trigger."""
    open_, high, low, close = df_slice["open"], df_slice["high"], df_slice["low"], df_slice["close"]
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)

    out = pd.DataFrame(index=df_slice.index)
    out["upper_wick_pct"] = upper_wick / close.replace(0, np.nan)
    out["upper_wick_to_body"] = upper_wick / body.replace(0, np.nan)
    out["close_location_in_range"] = (close - low) / full_range  # 1 = closed at the high, 0 = closed at the low
    running_max = high.cummax()
    out["rejection_from_high_pct"] = (running_max - close) / running_max.replace(0, np.nan)
    out["rejection_atr"] = (running_max - close) / atr_series.replace(0, np.nan)
    is_rejection_bar = (out["close_location_in_range"] < 0.5) & (upper_wick > body)
    out["consecutive_upper_rejections"] = is_rejection_bar.groupby((~is_rejection_bar).cumsum()).cumsum()
    return out


def compute_volatility_features(df_slice: pd.DataFrame, atr_series: pd.Series, climax_idx_wide: int, bar_interval_ms: int) -> pd.DataFrame:
    """Point 18: covers both hypothesized exhaustion signatures - a
    volatility CLIMAX (expansion peaking then fading) and volatility
    COMPRESSION near highs (contraction after the climax) - report both,
    pick neither as "the" pattern."""
    close = df_slice["close"]
    n = len(df_slice)
    b15, b60 = (_bars_for(m, bar_interval_ms) for m in (15, 60))
    log_ret = np.log(close / close.shift(1))

    out = pd.DataFrame(index=df_slice.index)
    out["atr_current"] = atr_series
    baseline_atr = atr_series.expanding(min_periods=1).mean()
    out["atr_expansion_ratio"] = atr_series / baseline_atr.replace(0, np.nan)
    out["realized_vol_15m"] = log_ret.rolling(b15, min_periods=2).std()
    out["realized_vol_1h"] = log_ret.rolling(b60, min_periods=2).std()
    out["vol_acceleration"] = out["realized_vol_15m"].diff(b15)

    climax_idx_wide = min(max(climax_idx_wide, 0), n - 1)
    vol_at_climax = out["realized_vol_15m"].iloc[climax_idx_wide]
    if pd.notna(vol_at_climax) and vol_at_climax:
        vol_after_climax_ratio = out["realized_vol_15m"] / vol_at_climax
    else:
        vol_after_climax_ratio = pd.Series(np.nan, index=df_slice.index)
    on_or_after_climax = pd.Series(np.arange(n), index=df_slice.index) >= climax_idx_wide
    out["vol_after_climax_ratio"] = vol_after_climax_ratio.where(on_or_after_climax)
    return out


def compute_pump_age_stage_features(df_slice: pd.DataFrame, detect_idx_wide: int, climax_idx_wide: int, bar_interval_ms: int) -> pd.DataFrame:
    """Point 19: pure timing/fraction-of-range descriptors. Stage labels
    (IGNITION/ACCELERATION/CLIMAX/STALL/DISTRIBUTION/BREAKDOWN) are
    deliberately NOT assigned here - the spec keeps stage heuristic/
    unsupervised and explicitly not a mandatory rule; only the raw
    ingredients a later manual/unsupervised pass would need are computed."""
    n = len(df_slice)
    idx_arr = np.arange(n)
    close = df_slice["close"]
    start_price = float(df_slice["open"].iloc[0])
    running_high = df_slice["high"].cummax()
    running_low = df_slice["low"].cummin()

    out = pd.DataFrame(index=df_slice.index)
    out["minutes_since_pump_start"] = idx_arr * bar_interval_ms / 60_000.0
    out["minutes_since_detection"] = (idx_arr - detect_idx_wide) * bar_interval_ms / 60_000.0
    out["minutes_since_climax"] = (idx_arr - climax_idx_wide) * bar_interval_ms / 60_000.0
    out["current_return_from_start"] = (close - start_price) / start_price if start_price else np.nan
    pump_range = (running_high - running_low).replace(0, np.nan)
    out["fraction_of_current_pump_range"] = (close - running_low) / pump_range
    return out


def compute_market_context_features(df_slice: pd.DataFrame, btc_ohlcv: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Point 20: BTC/market context, aligned via as-of merge onto df_slice's
    own timestamps (never a naive positional join - btc_ohlcv is a full
    separate series, not pre-sliced to this event's window)."""
    b5, b15, b60, b240 = (_bars_for(m, bar_interval_ms) for m in (5, 15, 60, 240))
    btc_aligned = pd.merge_asof(
        df_slice[["timestamp"]], btc_ohlcv[["timestamp", "close"]].rename(columns={"close": "btc_close"}), on="timestamp", direction="backward"
    )
    btc_close = btc_aligned["btc_close"]

    out = pd.DataFrame(index=df_slice.index)
    out["btc_return_5m"] = btc_close.pct_change(b5)
    out["btc_return_15m"] = btc_close.pct_change(b15)
    out["btc_return_1h"] = btc_close.pct_change(b60)
    out["btc_return_4h"] = btc_close.pct_change(b240)
    log_ret = np.log(btc_close / btc_close.shift(1))
    out["btc_vol"] = log_ret.rolling(b60, min_periods=2).std()
    out["btc_trend"] = np.where(out["btc_return_1h"] > 0.005, "up", np.where(out["btc_return_1h"] < -0.005, "down", "flat"))

    coin_return_15m = df_slice["close"].pct_change(b15)
    out["coin_return_minus_btc_return"] = coin_return_15m - out["btc_return_15m"]
    return out
