"""Market structure: pivots, HH/HL sequencing, structure break, sweep/rejection,
failed reclaim (spec sections 11.1-11.3). This is the core of "don't short
strength, short failed strength" - a pump with intact HH/HL structure must
never report a structure break, no matter how large the pump's return is.

Every detector here is deliberately backward-looking only: at bar j it may
use bars [0..j] and any pivot confirmed by bar j (a fractal pivot at index i
is only knowable once bar i+pivot_right_bars has printed - see find_pivots).
This makes the exact same functions safe for both the historical event
study/backtest and a future live scanner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FailedReclaimConfig, StructureConfig, SweepConfig

REQUIRED_OHLC_COLUMNS = ("open", "high", "low", "close")


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def find_pivots(high: pd.Series, low: pd.Series, left_bars: int, right_bars: int) -> pd.DataFrame:
    """Fractal pivots: bar i is a pivot high if high[i] is the max over
    [i-left_bars, i+right_bars]. A pivot at i is only KNOWABLE once bar
    i+right_bars has printed - the last right_bars bars of the input can
    never be marked as pivots, which is the correct no-lookahead behaviour
    (we simply don't know yet), not a bug.
    """
    n = len(high)
    pivot_high = np.full(n, False)
    pivot_low = np.full(n, False)
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    for i in range(left_bars, n - right_bars):
        lo, hi = i - left_bars, i + right_bars + 1
        if high_arr[i] == high_arr[lo:hi].max():
            pivot_high[i] = True
        if low_arr[i] == low_arr[lo:hi].min():
            pivot_low[i] = True
    return pd.DataFrame({"pivot_high": pivot_high, "pivot_low": pivot_low}, index=high.index)


def _bars_since(flag: pd.Series) -> pd.Series:
    n = len(flag)
    positions = np.where(flag.to_numpy(), np.arange(n), -1)
    last_true = pd.Series(positions, index=flag.index).replace(-1, np.nan).ffill()
    return pd.Series(np.arange(n), index=flag.index) - last_true


def compute_structure(df: pd.DataFrame, atr_series: pd.Series, cfg: StructureConfig) -> pd.DataFrame:
    """df: ascending-time OHLC frame. atr_series: same index, e.g. atr(df, cfg_stops.atr_length).

    Returns a per-bar frame with the running structure state:
      pivot_high, pivot_low, swing_high_level/_label ('HH'|'LH'|'H'),
      swing_low_level/_label ('HL'|'LL'|'L'), structure_intact,
      structure_break_bar, bars_since_structure_break, broken_level
      (the HL level that was broken - fixed at break time, NaN before any
      break), pre_break_swing_high (the swing high in force at break time,
      used by failed-reclaim's lower-high check).
    """
    for col in REQUIRED_OHLC_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"compute_structure requires column {col!r}")

    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr_arr = atr_series.to_numpy()

    pivots = find_pivots(df["high"], df["low"], cfg.pivot_left_bars, cfg.pivot_right_bars)
    pivot_high = pivots["pivot_high"].to_numpy()
    pivot_low = pivots["pivot_low"].to_numpy()

    swing_high_level = np.full(n, np.nan)
    swing_high_label = np.full(n, None, dtype=object)
    swing_low_level = np.full(n, np.nan)
    swing_low_label = np.full(n, None, dtype=object)
    structure_intact = np.full(n, True)
    structure_break_bar = np.full(n, False)
    broken_level_arr = np.full(n, np.nan)
    pre_break_high_arr = np.full(n, np.nan)

    last_swing_high: float | None = None
    last_swing_low: float | None = None
    cur_swing_high = np.nan
    cur_swing_high_label = None
    cur_swing_low = np.nan
    cur_swing_low_label = None
    broken = False
    broken_level_val = np.nan
    pre_break_high_val = np.nan
    close_below_count = 0

    def _distance_ok(price: float, reference: float | None, atr_at_confirm: float) -> bool:
        if reference is None:
            return True
        bps_dist = abs(price - reference) / reference * 10_000 if reference else 0.0
        atr_dist = abs(price - reference) / atr_at_confirm if atr_at_confirm and not np.isnan(atr_at_confirm) else 0.0
        return bps_dist >= cfg.min_swing_distance_bps or atr_dist >= cfg.min_swing_distance_atr

    for i in range(n):
        confirm_idx = i - cfg.pivot_right_bars
        if confirm_idx >= 0:
            if pivot_high[confirm_idx]:
                price = high[confirm_idx]
                if _distance_ok(price, last_swing_low, atr_arr[confirm_idx]):
                    if last_swing_high is None:
                        label = "H"
                    else:
                        label = "HH" if price > last_swing_high else "LH"
                    last_swing_high = price
                    cur_swing_high, cur_swing_high_label = price, label
            if pivot_low[confirm_idx]:
                price = low[confirm_idx]
                if _distance_ok(price, last_swing_high, atr_arr[confirm_idx]):
                    if last_swing_low is None:
                        label = "L"
                    else:
                        label = "HL" if price > last_swing_low else "LL"
                    last_swing_low = price
                    cur_swing_low, cur_swing_low_label = price, label

        swing_high_level[i] = cur_swing_high
        swing_high_label[i] = cur_swing_high_label
        swing_low_level[i] = cur_swing_low
        swing_low_label[i] = cur_swing_low_label

        if not broken and not np.isnan(cur_swing_low) and cur_swing_low_label == "HL":
            breached = (low[i] < cur_swing_low) if cfg.allow_wick_break else (close[i] < cur_swing_low)
            close_below_count = close_below_count + 1 if breached else 0
            if close_below_count >= cfg.close_confirmation_bars:
                broken = True
                structure_break_bar[i] = True
                broken_level_val = cur_swing_low
                pre_break_high_val = cur_swing_high

        structure_intact[i] = not broken
        broken_level_arr[i] = broken_level_val if broken else np.nan
        pre_break_high_arr[i] = pre_break_high_val if broken else np.nan

    out = pd.DataFrame(
        {
            "pivot_high": pivot_high,
            "pivot_low": pivot_low,
            "swing_high_level": swing_high_level,
            "swing_high_label": swing_high_label,
            "swing_low_level": swing_low_level,
            "swing_low_label": swing_low_label,
            "structure_intact": structure_intact,
            "structure_break_bar": structure_break_bar,
            "broken_level": broken_level_arr,
            "pre_break_swing_high": pre_break_high_arr,
        },
        index=df.index,
    )
    out["bars_since_structure_break"] = _bars_since(out["structure_break_bar"])
    return out


def compute_swing_magnitude(
    structure: pd.DataFrame, df: pd.DataFrame, atr_series: pd.Series, cfg: StructureConfig
) -> pd.DataFrame:
    """Continuous companion to compute_structure's categorical swing_high_label/
    swing_low_label (research/diagnostic use - not read by entry_engine/risk/
    exit). compute_structure's own _distance_ok() gate computes a continuous
    bps/ATR distance to decide whether a pivot is promoted to a confirmed
    swing at all, then keeps only the pass/fail categorical label and throws
    that magnitude away. This function re-derives the magnitude from
    compute_structure's already-computed output columns - it does NOT
    recompute pivots or re-run _distance_ok - plus the same df/atr_series
    that produced `structure`, so callers get "how big / how fast / how
    long" without duplicating compute_structure's pivot-confirmation loop.

    A bar counts as a freshly confirmed swing high/low exactly where
    swing_high_level/swing_low_level *changes value* versus the previous bar
    (including the initial NaN -> value transition). compute_structure only
    ever overwrites cur_swing_high/cur_swing_low on a bar where a new pivot
    just passed _distance_ok, and simply repeats the same value on every bar
    in between, so a value change is an exact, cheap proxy for "a new swing
    was confirmed here" that needs no access to confirm_idx or _distance_ok
    itself.

    For each such bar i (structure has already collapsed the underlying
    pivot bar and its right_bars confirmation delay into a single
    confirmation-bar row, so "this bar" IS the confirmation bar throughout):
      swing_distance_atr: |this swing's price - the most recently confirmed
        swing of the OPPOSITE type's price| / atr_series[i]. NaN if no
        opposite-type swing has been confirmed yet.
      swing_duration_bars: i - (the bar index that prior opposite-type swing
        was itself confirmed on).
      preceding_impulse_pct / preceding_impulse_atr: close[i] vs
        close[max(0, i - (cfg.pivot_left_bars + cfg.pivot_right_bars))] -
        the raw close-to-close price change over the pivot window
        immediately preceding this confirmation bar, as a fraction of the
        starting close and in atr_series[i] units. Clamped to bar 0 if the
        full lookback isn't available yet (a pivot near the start of the
        slice).

    If a bar happens to confirm BOTH a fresh high and a fresh low at once (a
    perfectly flat pivot window - degenerate, not expected with real price
    data), the low's values are what end up on that row; each branch still
    correctly uses the other type's state as it stood before this bar, so
    the collision only affects which one's values "win" that single row.

    Returns a frame aligned to `structure`'s index with columns
    swing_distance_atr, swing_duration_bars, preceding_impulse_pct,
    preceding_impulse_atr - NaN on every bar without a freshly confirmed
    swing.
    """
    n = len(structure)
    high_level = structure["swing_high_level"].to_numpy(dtype=float)
    low_level = structure["swing_low_level"].to_numpy(dtype=float)
    close = df["close"].to_numpy()
    atr_arr = atr_series.to_numpy()
    lookback = cfg.pivot_left_bars + cfg.pivot_right_bars

    swing_distance_atr = np.full(n, np.nan)
    swing_duration_bars = np.full(n, np.nan)
    preceding_impulse_pct = np.full(n, np.nan)
    preceding_impulse_atr = np.full(n, np.nan)

    def _is_fresh(level_arr: np.ndarray, i: int) -> bool:
        if np.isnan(level_arr[i]):
            return False
        return i == 0 or np.isnan(level_arr[i - 1]) or level_arr[i] != level_arr[i - 1]

    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator and not np.isnan(denominator) else np.nan

    last_high: tuple[float, int] | None = None  # (price, bar_idx) most recently confirmed
    last_low: tuple[float, int] | None = None

    for i in range(n):
        fresh_high = _is_fresh(high_level, i)
        fresh_low = _is_fresh(low_level, i)
        prior_high, prior_low = last_high, last_low  # snapshot before either branch below updates state

        if fresh_high:
            price = high_level[i]
            if prior_low is not None:
                prior_price, prior_idx = prior_low
                swing_distance_atr[i] = _safe_div(abs(price - prior_price), atr_arr[i])
                swing_duration_bars[i] = i - prior_idx
            start_idx = max(0, i - lookback)
            preceding_impulse_pct[i] = _safe_div(close[i] - close[start_idx], close[start_idx])
            preceding_impulse_atr[i] = _safe_div(close[i] - close[start_idx], atr_arr[i])
            last_high = (price, i)

        if fresh_low:
            price = low_level[i]
            if prior_high is not None:
                prior_price, prior_idx = prior_high
                swing_distance_atr[i] = _safe_div(abs(price - prior_price), atr_arr[i])
                swing_duration_bars[i] = i - prior_idx
            start_idx = max(0, i - lookback)
            preceding_impulse_pct[i] = _safe_div(close[i] - close[start_idx], close[start_idx])
            preceding_impulse_atr[i] = _safe_div(close[i] - close[start_idx], atr_arr[i])
            last_low = (price, i)

    return pd.DataFrame(
        {
            "swing_distance_atr": swing_distance_atr,
            "swing_duration_bars": swing_duration_bars,
            "preceding_impulse_pct": preceding_impulse_pct,
            "preceding_impulse_atr": preceding_impulse_atr,
        },
        index=structure.index,
    )


def post_break_signals(df: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    """Two structural facts entry-mode gating needs beyond the exhaustion-score
    flags (spec section 14): lower_high_formed (a genuine LH swing has been
    confirmed since the break - stronger evidence than the break itself) and
    new_local_low (this bar's low is the lowest since the break, i.e. price
    is continuing down rather than having only poked below the HL once).
    """
    n = len(df)
    low = df["low"].to_numpy()
    bars_since_break = structure["bars_since_structure_break"].to_numpy()
    swing_high_label = structure["swing_high_label"].to_numpy()

    lower_high_formed = np.full(n, False)
    new_local_low = np.full(n, False)
    running_min_low = np.inf
    seen_lh_since_break = False
    prev_bsb = np.nan
    for i in range(n):
        bsb = bars_since_break[i]
        if np.isnan(bsb):
            running_min_low = np.inf
            seen_lh_since_break = False
        else:
            if np.isnan(prev_bsb) or bsb < prev_bsb:  # a fresh break just occurred (counter reset to 0)
                running_min_low = np.inf
                seen_lh_since_break = False
            if swing_high_label[i] == "LH":
                seen_lh_since_break = True
            lower_high_formed[i] = seen_lh_since_break
            new_local_low[i] = low[i] < running_min_low
            running_min_low = min(running_min_low, low[i])
        prev_bsb = bsb

    return pd.DataFrame(
        {"lower_high_formed": lower_high_formed, "new_local_low": new_local_low}, index=df.index
    )


def detect_sweep_rejection(df: pd.DataFrame, structure: pd.DataFrame, cfg: SweepConfig) -> pd.Series:
    """At bar j: was the CURRENT swing-high level swept (new high beyond it by
    >= min_sweep_bps, with a rejection-shaped wick) within the last
    max_rejection_bars bars, and has price now (bar j) closed back below it?
    Only ever looks backward from j - safe to evaluate live, bar by bar.
    """
    n = len(df)
    high = df["high"].to_numpy()
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    prev_high = structure["swing_high_level"].to_numpy()

    is_sweep_bar = np.full(n, False)
    for i in range(n):
        ph = prev_high[i]
        if np.isnan(ph):
            continue
        sweep_bps = (high[i] - ph) / ph * 10_000
        if sweep_bps < cfg.min_sweep_bps:
            continue
        body = abs(close[i] - open_[i])
        upper_wick = high[i] - max(close[i], open_[i])
        wick_ok = (upper_wick >= cfg.upper_wick_body_ratio_min * body) if body > 0 else upper_wick > 0
        is_sweep_bar[i] = wick_ok

    result = np.full(n, False)
    if not cfg.close_back_below_previous_high:
        return pd.Series(result, index=df.index)
    for j in range(n):
        ph = prev_high[j]
        if np.isnan(ph) or close[j] >= ph:
            continue
        lookback_start = max(0, j - cfg.max_rejection_bars)
        for i in range(j, lookback_start - 1, -1):
            if prev_high[i] != ph:
                break  # the swing-high reference itself changed further back
            if is_sweep_bar[i]:
                result[j] = True
                break
    return pd.Series(result, index=df.index)


def detect_failed_reclaim(df: pd.DataFrame, structure: pd.DataFrame, cfg: FailedReclaimConfig) -> pd.Series:
    """At bar j (within max_bars_after_break of the most recent structure
    break): has price poked back into the reclaim zone above the broken
    level since the break, formed a lower high (didn't exceed the pre-break
    swing high) if required, and closed back below the broken level by bar j?
    """
    if not cfg.enabled:
        return pd.Series(False, index=df.index)

    n = len(df)
    high = df["high"].to_numpy()
    close = df["close"].to_numpy()
    broken_level = structure["broken_level"].to_numpy()
    pre_break_high = structure["pre_break_swing_high"].to_numpy()
    bars_since_break = structure["bars_since_structure_break"].to_numpy()

    result = np.full(n, False)
    for j in range(n):
        lvl = broken_level[j]
        bsb = bars_since_break[j]
        if np.isnan(lvl) or np.isnan(bsb) or bsb > cfg.max_bars_after_break:
            continue
        break_idx = int(j - bsb)
        window_high = high[break_idx : j + 1]
        if window_high.size == 0:
            continue
        zone_top = lvl * (1 + cfg.reclaim_zone_bps / 10_000)
        entered_zone = bool(((window_high >= lvl) & (window_high <= zone_top)).any())
        if not entered_zone:
            continue  # never poked back up into the reclaim zone - no attempt yet
        closed_below_ok = (not cfg.require_close_below_reclaim_zone) or (close[j] < lvl)
        lower_high_ok = (not cfg.require_lower_high) or (
            not np.isnan(pre_break_high[j]) and window_high.max() < pre_break_high[j]
        )
        if closed_below_ok and lower_high_ok:
            result[j] = True
    return pd.Series(result, index=df.index)
