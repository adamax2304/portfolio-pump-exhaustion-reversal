"""Phase 3A points 7, 8, 9, 11, 16 - the NEW priority-1 hypotheses: model the
pump LOSING its ability to continue, not a single-candle top pattern.
Deceleration, high-extension exhaustion, deepening pullbacks, declining
participation efficiency, and failed continuation attempts.

Every function here takes `df_slice`/`cvd_slice` (an event's post-pump-start
WIDE-indexed OHLCV/feature slice - see snapshot_builder.SnapshotWindow) and
returns a DataFrame of per-bar columns aligned to the SAME index, computed
with pandas vectorized ops (rolling/expanding/shift/cumsum) - never a
per-bar Python loop, and every value at row i depends only on rows [0, i]
(see tests/test_screener_v2_pump_dynamics.py's causality checks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def _bars_for(minutes: float, bar_interval_ms: int) -> int:
    return max(round(minutes * 60_000 / bar_interval_ms), 1)


def _windowed_velocity(close: pd.Series, near_bars_ago: int, far_bars_ago: int) -> pd.Series:
    """Return-per-bar over the window from far_bars_ago to near_bars_ago
    (both measured backward from 'now'); near_bars_ago < far_bars_ago."""
    span = far_bars_ago - near_bars_ago
    near_price = close.shift(near_bars_ago)
    far_price = close.shift(far_bars_ago)
    return ((near_price - far_price) / far_price.replace(0, np.nan)) / span


def compute_velocity_features(df_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Point 7: price velocity + acceleration across adjacent horizons.
    acceleration_X_to_Ym = velocity over the most recent X-Y window minus
    velocity over the PRECEDING window of the same width ("is the pump still
    rising but slowing down") - not a single fixed-horizon return."""
    close = df_slice["close"]
    b5, b15, b30, b60 = (_bars_for(m, bar_interval_ms) for m in (5, 15, 30, 60))

    out = pd.DataFrame(index=df_slice.index)
    out["return_5m"] = close.pct_change(b5)
    out["return_15m"] = close.pct_change(b15)
    out["return_30m"] = close.pct_change(b30)
    out["return_60m"] = close.pct_change(b60)

    v_0_5 = _windowed_velocity(close, 0, b5)
    v_5_15 = _windowed_velocity(close, b5, b15)
    v_15_30 = _windowed_velocity(close, b15, b30)
    v_30_60 = _windowed_velocity(close, b30, b60)
    out["velocity_5m"] = v_0_5
    out["velocity_15m"] = v_5_15
    out["acceleration_5_to_15m"] = v_0_5 - v_5_15
    out["acceleration_15_to_30m"] = v_5_15 - v_15_30
    out["acceleration_30_to_60m"] = v_15_30 - v_30_60
    return out


def compute_high_extension_features(df_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Point 8: is the pump still printing meaningfully bigger new highs, or
    are successive highs shrinking (HH1 +8% -> HH2 +4% -> HH3 +1.2% -> none)?
    A "new high" here is any bar whose high exceeds the max of every prior
    bar in df_slice (pump-start-anchored, matching event_high_so_far in
    snapshot_builder) - ties do not count (must be a strict new record)."""
    high = df_slice["high"]
    n = len(df_slice)
    idx_arr = np.arange(n)

    running_max_prior = high.cummax().shift(1)
    is_new_high = (high > running_max_prior.fillna(-np.inf)).astype(bool)

    last_high_idx = pd.Series(np.where(is_new_high, idx_arr, np.nan), index=df_slice.index).ffill()
    bars_since_last_new_high = idx_arr - last_high_idx

    b15, b30, b60 = (_bars_for(m, bar_interval_ms) for m in (15, 30, 60))
    is_new_high_f = is_new_high.astype(float)
    number_of_new_highs_15m = is_new_high_f.rolling(b15, min_periods=1).sum()
    number_of_new_highs_30m = is_new_high_f.rolling(b30, min_periods=1).sum()
    number_of_new_highs_60m = is_new_high_f.rolling(b60, min_periods=1).sum()

    new_high_value = high.where(is_new_high)
    prev_new_high_value_at_event = new_high_value.ffill().shift(1).where(is_new_high)
    extension_at_new_high = (new_high_value - prev_new_high_value_at_event) / prev_new_high_value_at_event.replace(0, np.nan)
    last_high_extension_pct = extension_at_new_high.ffill()

    extension_events = extension_at_new_high.dropna()
    previous_high_extension_pct = extension_events.shift(1).reindex(df_slice.index).ffill()

    position_at_new_high = pd.Series(idx_arr, index=df_slice.index).where(is_new_high).dropna()
    gap_between_highs = position_at_new_high.diff()
    latest_time_between_highs = gap_between_highs.reindex(df_slice.index).ffill()
    mean_time_between_highs = gap_between_highs.expanding().mean().reindex(df_slice.index).ffill()

    high_extension_velocity = last_high_extension_pct / latest_time_between_highs.replace(0, np.nan)
    velocity_events = (extension_events / gap_between_highs.replace(0, np.nan)).dropna()
    high_extension_deceleration = (velocity_events - velocity_events.shift(1)).reindex(df_slice.index).ffill()

    out = pd.DataFrame(index=df_slice.index)
    out["is_new_high"] = is_new_high
    out["bars_since_last_new_high"] = bars_since_last_new_high
    out["time_since_last_new_high_minutes"] = bars_since_last_new_high * bar_interval_ms / 60_000.0
    out["number_of_new_highs_15m"] = number_of_new_highs_15m
    out["number_of_new_highs_30m"] = number_of_new_highs_30m
    out["number_of_new_highs_60m"] = number_of_new_highs_60m
    out["last_high_extension_pct"] = last_high_extension_pct
    out["previous_high_extension_pct"] = previous_high_extension_pct
    out["latest_time_between_highs_bars"] = latest_time_between_highs
    out["mean_time_between_highs_bars"] = mean_time_between_highs
    out["high_extension_velocity"] = high_extension_velocity
    out["high_extension_deceleration"] = high_extension_deceleration
    return out


def compute_pullback_features(df_slice: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Point 9 (Phase 3A) + Phase 3A.1 point 7 (user's #1 reprioritized
    hypothesis, "pullbacks become deeper AND recoveries become slower/
    weaker" being closer to real exhaustion than absolute pullback depth
    alone): are pullbacks getting deeper, are recoveries getting weaker/
    slower? A "pullback" episode runs from one new high to the next; its
    depth is the max drawdown reached anywhere in that episode (causal
    cummax within the episode, so a bar mid-pullback only ever sees the
    drawdown SO FAR, never the eventual full depth of its own episode -
    only COMPLETED prior episodes ever feed the latest/previous/average/
    slope columns below)."""
    high, close = df_slice["high"], df_slice["close"]
    n = len(df_slice)
    idx_arr = np.arange(n)

    running_max = high.cummax()
    running_max_prior = running_max.shift(1)
    is_new_high = (high > running_max_prior.fillna(-np.inf)).astype(bool)

    drawdown_from_high_pct = (running_max - close) / running_max.replace(0, np.nan)
    drawdown_from_high_atr = (running_max - close) / atr_series.replace(0, np.nan)

    episode = is_new_high.cumsum()
    pullback_depth_pct = drawdown_from_high_pct.groupby(episode).cummax()
    pullback_depth_atr = drawdown_from_high_atr.groupby(episode).cummax()

    completed_depth_pct = pullback_depth_pct.shift(1).where(is_new_high)
    completed_depth_atr = pullback_depth_atr.shift(1).where(is_new_high)
    latest_pullback_depth_pct = completed_depth_pct.ffill()
    latest_pullback_depth_atr = completed_depth_atr.ffill()
    average_pullback_depth_pct = completed_depth_pct.dropna().expanding().mean().reindex(df_slice.index).ffill()

    # --- Phase 3A.1 point 7: latest-vs-previous, latest-vs-average-prior,
    # and deterioration slopes - all built on the SAME completed-episode
    # event sequence (mirrors previous_high_extension_pct's pattern), so
    # "previous"/"average_prior" always exclude the latest (still-forming
    # or just-completed) pullback itself.
    completed_pct_events = completed_depth_pct.dropna()
    previous_pullback_depth_pct = completed_pct_events.shift(1).reindex(df_slice.index).ffill()
    average_prior_pullback_depth_pct = completed_pct_events.expanding().mean().shift(1).reindex(df_slice.index).ffill()
    latest_over_previous_pullback_ratio = latest_pullback_depth_pct / previous_pullback_depth_pct.replace(0, np.nan)
    latest_over_average_prior_pullback_ratio = latest_pullback_depth_pct / average_prior_pullback_depth_pct.replace(0, np.nan)
    pullback_depth_slope = completed_pct_events.diff().reindex(df_slice.index).ffill()

    last_high_idx = pd.Series(np.where(is_new_high, idx_arr, np.nan), index=df_slice.index).ffill()
    position_at_new_high = pd.Series(idx_arr, index=df_slice.index).where(is_new_high).dropna()
    completed_duration_events = position_at_new_high.diff()
    latest_pullback_duration_bars = completed_duration_events.reindex(df_slice.index).ffill()
    previous_pullback_duration_bars = completed_duration_events.shift(1).reindex(df_slice.index).ffill()
    pullback_duration_slope = completed_duration_events.diff().reindex(df_slice.index).ffill()

    # recovery: causal running low WITHIN the episode (bar index of the
    # lowest close seen so far in this episode, updated via ffill using only
    # past bars) lets us measure, bar by bar, how much of the drawdown-so-far
    # has been clawed back (recovery_fraction) and, at each completed
    # episode, how many bars the full recovery took (recovery_time).
    episode_low_so_far = close.groupby(episode).cummin()
    is_new_episode_low = close <= episode_low_so_far
    low_bar_idx = pd.Series(np.where(is_new_episode_low, idx_arr, np.nan), index=df_slice.index)
    running_low_bar_idx = low_bar_idx.groupby(episode).ffill()
    recovery_time_so_far_bars = idx_arr - running_low_bar_idx
    completed_recovery_time = recovery_time_so_far_bars.shift(1).where(is_new_high)
    completed_recovery_time_events = completed_recovery_time.dropna()
    latest_recovery_time_bars = completed_recovery_time.ffill()
    recovery_time_slope = completed_recovery_time_events.diff().reindex(df_slice.index).ffill()
    recovery_fraction = (close - episode_low_so_far) / (running_max - episode_low_so_far).replace(0, np.nan)

    out = pd.DataFrame(index=df_slice.index)
    out["pullback_depth_pct"] = pullback_depth_pct  # in-progress (current episode), causal cummax
    out["pullback_depth_atr"] = pullback_depth_atr
    out["pullback_duration_bars"] = idx_arr - last_high_idx
    out["latest_pullback_depth_pct"] = latest_pullback_depth_pct  # last COMPLETED pullback only
    out["latest_pullback_depth_atr"] = latest_pullback_depth_atr
    out["average_pullback_depth_pct"] = average_pullback_depth_pct
    out["previous_pullback_depth_pct"] = previous_pullback_depth_pct  # the completed pullback BEFORE the latest one
    out["average_prior_pullback_depth_pct"] = average_prior_pullback_depth_pct  # average of all completed pullbacks EXCLUDING the latest
    out["latest_over_previous_pullback_ratio"] = latest_over_previous_pullback_ratio
    out["latest_over_average_prior_pullback_ratio"] = latest_over_average_prior_pullback_ratio
    out["pullback_depth_slope"] = pullback_depth_slope  # positive = pullbacks getting DEEPER
    out["latest_pullback_duration_bars"] = latest_pullback_duration_bars
    out["previous_pullback_duration_bars"] = previous_pullback_duration_bars
    out["pullback_duration_slope"] = pullback_duration_slope  # positive = pullbacks taking LONGER
    out["recovery_fraction"] = recovery_fraction  # 0 at the episode low, ->1 as price claws back toward the peak
    out["latest_recovery_time_bars"] = latest_recovery_time_bars
    out["recovery_time_slope"] = recovery_time_slope  # positive = recovery from the low is taking LONGER than last time
    return out


def compute_efficiency_features(df_slice: pd.DataFrame, cvd_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Point 11: price extension achieved per unit of participation (volume/
    OI), normalized against the event's OWN expanding history (never a raw
    cross-event-comparable number) - the "absorption" hypothesis: volume
    stays high but the price extension it buys keeps shrinking."""
    close = df_slice["close"]
    quote_volume = df_slice["quote_volume"] if "quote_volume" in df_slice.columns else df_slice["volume"] * close
    b30 = _bars_for(30, bar_interval_ms)

    price_change_window = close.diff(b30).abs()
    volume_window = quote_volume.rolling(b30, min_periods=1).sum()
    efficiency_volume_raw = price_change_window / volume_window.replace(0, np.nan)
    efficiency_volume_baseline = efficiency_volume_raw.expanding(min_periods=1).mean()
    price_efficiency_volume = efficiency_volume_raw / efficiency_volume_baseline.replace(0, np.nan)
    efficiency_volume_slope = efficiency_volume_raw.diff(b30)

    out = pd.DataFrame(index=df_slice.index)
    out["price_efficiency_volume_raw"] = efficiency_volume_raw
    out["price_efficiency_volume_relative"] = price_efficiency_volume  # >1 = more efficient than event's own average so far
    out["efficiency_volume_slope"] = efficiency_volume_slope

    if "open_interest" in cvd_slice.columns and cvd_slice["open_interest"].notna().any():
        oi = cvd_slice["open_interest"]
        oi_change_window = oi.diff(b30).abs()
        efficiency_oi_raw = price_change_window / oi_change_window.replace(0, np.nan)
        efficiency_oi_baseline = efficiency_oi_raw.expanding(min_periods=1).mean()
        out["price_efficiency_oi_raw"] = efficiency_oi_raw
        out["price_efficiency_oi_relative"] = efficiency_oi_raw / efficiency_oi_baseline.replace(0, np.nan)
        out["efficiency_oi_slope"] = efficiency_oi_raw.diff(b30)
    else:
        out["price_efficiency_oi_raw"] = np.nan
        out["price_efficiency_oi_relative"] = np.nan
        out["efficiency_oi_slope"] = np.nan
    return out


def compute_failed_continuation_features(df_slice: pd.DataFrame, cvd_slice: pd.DataFrame, bar_interval_ms: int, volume_zscore_threshold: float = 1.0) -> pd.DataFrame:
    """Point 16 (Phase 3A) + Phase 3A.1 point 8 (user's #2 reprioritized
    hypothesis, closer to an economic definition of exhaustion: volume/OI
    participation keeps arriving while the price output it buys shrinks): a
    fresh push toward/past the running high WITH real participation (volume
    impulse above the event's own trailing baseline) that fails to
    meaningfully extend the high - distinct from failed reclaim (which
    requires price to have already broken structure and reclaimed it; this
    can fire on ANY attempted new high, even before any structure break).
    "failed_continuation_strength" is high when participation was large but
    achieved extension was small."""
    high = df_slice["high"]
    n = len(df_slice)
    running_max_prior = high.cummax().shift(1)
    is_new_high = (high > running_max_prior.fillna(-np.inf)).astype(bool)

    volume_zscore = cvd_slice["volume_zscore"] if "volume_zscore" in cvd_slice.columns else pd.Series(np.nan, index=df_slice.index)
    has_participation = (volume_zscore >= volume_zscore_threshold).fillna(False)
    continuation_attempt_detected = is_new_high & has_participation

    new_high_value = high.where(is_new_high)
    prev_new_high_value = new_high_value.ffill().shift(1).where(is_new_high)
    extension_pct = (new_high_value - prev_new_high_value) / prev_new_high_value.replace(0, np.nan)

    continuation_price_extension = extension_pct.where(continuation_attempt_detected)
    continuation_efficiency = (continuation_price_extension / volume_zscore.where(continuation_attempt_detected).abs().clip(lower=_EPS))
    failed_continuation_strength = (volume_zscore.where(continuation_attempt_detected).abs() / continuation_price_extension.abs().clip(lower=_EPS))

    has_oi = "open_interest" in cvd_slice.columns and cvd_slice["open_interest"].notna().any()
    if has_oi:
        oi = cvd_slice["open_interest"]
        oi_change = oi.diff()
        oi_change_mean = oi_change.expanding(min_periods=5).mean()
        oi_change_std = oi_change.expanding(min_periods=5).std()
        oi_impulse_zscore = (oi_change - oi_change_mean) / oi_change_std.replace(0, np.nan)
    else:
        oi_change = pd.Series(np.nan, index=df_slice.index)
        oi_impulse_zscore = pd.Series(np.nan, index=df_slice.index)

    # Phase 3A.1 point 8: a combined normalized participation "input"
    # (volume impulse + OI impulse, both clipped to their expansionary side)
    # against price-extension "output". NaN whenever OI is unavailable for
    # the WHOLE event - never silently zero-filled, which would understate
    # input for OI-less events and make them look artificially more
    # efficient than events where OI data actually exists.
    volume_impulse = volume_zscore.where(continuation_attempt_detected).clip(lower=0)
    if has_oi:
        oi_impulse = oi_impulse_zscore.where(continuation_attempt_detected).clip(lower=0)
        continuation_input_combined = volume_impulse + oi_impulse
        continuation_efficiency_combined = continuation_price_extension / continuation_input_combined.clip(lower=_EPS)
    else:
        continuation_input_combined = pd.Series(np.nan, index=df_slice.index)
        continuation_efficiency_combined = pd.Series(np.nan, index=df_slice.index)

    # event-collapsed attempt sequence -> previous-attempt comparison,
    # mirrors compute_pullback_features' completed-episode pattern.
    efficiency_events = continuation_efficiency.dropna()
    efficiency_now = continuation_efficiency.ffill()
    efficiency_previous_attempt = efficiency_events.shift(1).reindex(df_slice.index).ffill()
    efficiency_change = (efficiency_events - efficiency_events.shift(1)).reindex(df_slice.index).ffill()

    if has_oi:
        efficiency_combined_events = continuation_efficiency_combined.dropna()
        efficiency_combined_now = continuation_efficiency_combined.ffill()
        efficiency_combined_previous_attempt = efficiency_combined_events.shift(1).reindex(df_slice.index).ffill()
        efficiency_combined_change = (efficiency_combined_events - efficiency_combined_events.shift(1)).reindex(df_slice.index).ffill()
    else:
        efficiency_combined_now = pd.Series(np.nan, index=df_slice.index)
        efficiency_combined_previous_attempt = pd.Series(np.nan, index=df_slice.index)
        efficiency_combined_change = pd.Series(np.nan, index=df_slice.index)

    out = pd.DataFrame(index=df_slice.index)
    out["continuation_attempt_detected"] = continuation_attempt_detected
    out["continuation_attempt_volume_zscore"] = volume_zscore.where(continuation_attempt_detected).ffill()
    out["continuation_price_extension"] = continuation_price_extension.ffill()
    out["continuation_efficiency"] = continuation_efficiency.ffill()
    out["failed_continuation_strength"] = failed_continuation_strength.ffill()
    out["continuation_input_combined"] = continuation_input_combined.ffill() if has_oi else np.nan
    out["continuation_efficiency_combined"] = continuation_efficiency_combined.ffill() if has_oi else np.nan
    out["efficiency_now"] = efficiency_now
    out["efficiency_previous_attempt"] = efficiency_previous_attempt
    out["efficiency_change"] = efficiency_change  # negative = efficiency FELL vs the previous attempt (deteriorating)
    out["efficiency_combined_now"] = efficiency_combined_now
    out["efficiency_combined_previous_attempt"] = efficiency_combined_previous_attempt
    out["efficiency_combined_change"] = efficiency_combined_change
    out["continuation_attempt_oi_change"] = oi_change.where(continuation_attempt_detected).ffill() if has_oi else np.nan
    return out


def compute_volume_efficiency_at_highs(df_slice: pd.DataFrame, bar_interval_ms: int) -> pd.DataFrame:
    """Phase 3A.1 point 9: volume spent PER UNIT of high-extension achieved,
    and the change in that ratio between the last two high-attempts
    (marginal_volume_efficiency) - distinct from continuation_efficiency
    (which only looks at attempts gated by an elevated volume_zscore); this
    looks at EVERY new high, weighting participation by the event's own
    trailing baseline rather than a hard z-score gate."""
    high, close = df_slice["high"], df_slice["close"]
    running_max_prior = high.cummax().shift(1)
    is_new_high = (high > running_max_prior.fillna(-np.inf)).astype(bool)

    quote_volume = df_slice["quote_volume"] if "quote_volume" in df_slice.columns else df_slice["volume"] * close
    b15, b30 = (_bars_for(m, bar_interval_ms) for m in (15, 30))
    volume_at_attempt = quote_volume.rolling(b15, min_periods=1).sum()
    volume_baseline = volume_at_attempt.expanding(min_periods=1).mean()
    normalized_volume_at_attempt = volume_at_attempt / volume_baseline.replace(0, np.nan)

    new_high_value = high.where(is_new_high)
    prev_new_high_value = new_high_value.ffill().shift(1).where(is_new_high)
    extension_pct = (new_high_value - prev_new_high_value) / prev_new_high_value.replace(0, np.nan)

    volume_per_high_extension = normalized_volume_at_attempt.where(is_new_high) / extension_pct.abs().clip(lower=_EPS)
    events = volume_per_high_extension.dropna()
    volume_per_high_extension_now = volume_per_high_extension.ffill()
    volume_per_high_extension_previous_attempt = events.shift(1).reindex(df_slice.index).ffill()
    marginal_volume_efficiency = (events - events.shift(1)).reindex(df_slice.index).ffill()

    price_return_30m = close.pct_change(b30).abs()
    volume_window_30m = quote_volume.rolling(b30, min_periods=1).sum()
    volume_per_return = volume_window_30m / price_return_30m.replace(0, np.nan)

    out = pd.DataFrame(index=df_slice.index)
    out["volume_per_high_extension"] = volume_per_high_extension_now
    out["volume_per_high_extension_previous_attempt"] = volume_per_high_extension_previous_attempt
    out["marginal_volume_efficiency"] = marginal_volume_efficiency  # positive = volume cost per unit extension is RISING (less efficient than last attempt)
    out["volume_per_return"] = volume_per_return
    return out
