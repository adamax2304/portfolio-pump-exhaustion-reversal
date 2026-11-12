"""Phase 3A point 2: event snapshot dataset. Replaces v1's "1 pump event = 1
row" with "1 active pump event -> a causal snapshot every 5m, from
PUMP_DETECTED through min(max_lifecycle_horizon, data availability)".

Deliberately NOT tied to v1's invalidation/state-machine/attempt-retry logic
- v2 is asking a different, prior question (does current pump STATE carry
forward-outcome information) and must not inherit v1's trading assumptions
about when an "event" ends. A snapshot's own features only ever describe
"what has this pump event's price/volume/OI/structure done so far" -
whether v1's state machine would have called it CLOSED, INVALIDATED, or kept
retrying is irrelevant to that description.

Reuses backtest.simulator's climax-anchored post-pump structure/CVD/OI
computation (_build_post_pump_inputs) unchanged - same causal-safety
guarantee already validated across Phase 2A-2D, not re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord, _PostPumpInputs, _build_post_pump_inputs
from ..config import AppConfig
from ..data.normalizer import parse_timeframe_ms

DEFAULT_MAX_LIFECYCLE_HOURS = 10.0  # user's example range (8-12h) - a frozen research parameter, never tuned per-result


@dataclass
class SnapshotWindow:
    """Everything a features.py/pump_dynamics.py function needs to compute
    ONE event's snapshot columns, vectorized over the whole post-pump window
    at once (never a per-bar Python loop across feature categories - each
    category function returns a full per-bar Series/DataFrame here, and
    snapshot_builder slices out just the snapshot rows at the end).

    Coordinate systems (deliberately kept distinct, like the rest of this
    project's causal-safety-critical code):
      - absolute: position into `ohlcv`/`feature_frame` directly (.iloc-safe).
      - wide: position into `df_slice`/`cvd_slice` (== absolute - pump_start_idx).
      - post_pump: position into `post_pump.post_pump_df` (== wide - post_pump.structure_start_idx).
    """

    event: EventRecord
    ohlcv: pd.DataFrame  # full symbol OHLCV, absolute-indexed
    feature_frame: pd.DataFrame  # aligned to ohlcv
    btc_ohlcv: pd.DataFrame
    df_slice: pd.DataFrame  # ohlcv.iloc[pump_start_idx : end_idx+1], reset_index(drop=True) - WIDE coordinates
    cvd_slice: pd.DataFrame  # feature_frame, same slice
    post_pump: _PostPumpInputs  # climax-anchored structure/cvd/oi - POST_PUMP coordinates
    pump_start_idx: int  # absolute
    detect_idx: int  # absolute
    climax_idx_abs: int  # absolute
    end_idx: int  # absolute - last bar snapshots are built for (inclusive)
    bar_interval_ms: int

    def wide(self, abs_idx: int) -> int:
        return abs_idx - self.pump_start_idx

    def post_pump_local(self, abs_idx: int) -> int:
        return self.wide(abs_idx) - self.post_pump.structure_start_idx


def _detect_idx_for(event: EventRecord, ohlcv: pd.DataFrame) -> int | None:
    idx = int(np.searchsorted(ohlcv["timestamp"].to_numpy(), event.pump_detected_ts, side="left"))
    return idx if idx < len(ohlcv) else None


def build_snapshot_window(
    event: EventRecord, ohlcv: pd.DataFrame, feature_frame: pd.DataFrame, btc_ohlcv: pd.DataFrame, timeframe: str, cfg: AppConfig,
    max_lifecycle_hours: float = DEFAULT_MAX_LIFECYCLE_HOURS,
) -> SnapshotWindow | None:
    """Resolves one event's window bounds and climax-anchored post-pump
    structure. Returns None if the event's detection timestamp can't be
    located or there's no room for even one snapshot bar."""
    bar_interval_ms = parse_timeframe_ms(timeframe)
    detect_idx = _detect_idx_for(event, ohlcv)
    if detect_idx is None or detect_idx >= len(ohlcv) - 1:
        return None

    max_bars = max(round(max_lifecycle_hours * 3_600_000 / bar_interval_ms), 1)
    end_idx = min(len(ohlcv) - 1, detect_idx + max_bars)

    # Same pump_start_idx convention as backtest.simulator._build_event_inputs
    # (max window across cfg.pump_detector.windows) - kept consistent so
    # climax_local_idx (already saved on EventRecord, wide/pump_start_idx-
    # relative) translates to the SAME absolute bar here as it did in v1.
    from ..features.returns import bars_for_window  # local import: avoids a module-level cycle with backtest.simulator

    windows_bars = [bars_for_window(parse_timeframe_ms(w), bar_interval_ms) for w in cfg.pump_detector.windows]
    lookback_bars = max(windows_bars) if windows_bars else 20
    pump_start_idx = max(0, detect_idx - lookback_bars)

    if pd.isna(event.climax_local_idx):
        return None
    climax_idx_abs = pump_start_idx + int(event.climax_local_idx)
    if climax_idx_abs < 0 or climax_idx_abs >= len(ohlcv):
        return None
    if climax_idx_abs > end_idx:
        # EventRecord.climax_local_idx is the FINAL reversal attempt's climax
        # (see backtest.simulator's own docstring) - on a multi-attempt event
        # a late retry can re-anchor to a new climax that occurs AFTER this
        # window's max_lifecycle_hours truncation point. Clamping to end_idx
        # keeps df_slice/climax_idx_wide consistent (climax_idx_wide must
        # never point past the end of df_slice, or _build_post_pump_inputs's
        # structure/CVD/OI recompute operates outside df_slice's own bounds)
        # - the snapshots themselves still only ever describe attempt-1-style
        # state up to end_idx, which is what actually happened within the
        # window regardless.
        climax_idx_abs = end_idx

    df_slice = ohlcv.iloc[pump_start_idx : end_idx + 1].reset_index(drop=True)
    cvd_slice = feature_frame.iloc[pump_start_idx : end_idx + 1].reset_index(drop=True)
    climax_idx_wide = climax_idx_abs - pump_start_idx
    post_pump = _build_post_pump_inputs(df_slice, cvd_slice, climax_idx_wide, bar_interval_ms, cfg)

    return SnapshotWindow(
        event=event, ohlcv=ohlcv, feature_frame=feature_frame, btc_ohlcv=btc_ohlcv, df_slice=df_slice, cvd_slice=cvd_slice,
        post_pump=post_pump, pump_start_idx=pump_start_idx, detect_idx=detect_idx, climax_idx_abs=climax_idx_abs,
        end_idx=end_idx, bar_interval_ms=bar_interval_ms,
    )


def assemble_snapshots(window: SnapshotWindow, feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenates every feature-category DataFrame (each already aligned to
    `window.df_slice`'s WIDE index, one column set per category), slices to
    just the snapshot rows (detect_idx..end_idx), and prepends the
    identifying columns (spec section 2's example schema)."""
    event = window.event
    wide_detect = window.wide(window.detect_idx)
    wide_end = window.wide(window.end_idx)

    identity = pd.DataFrame(index=range(len(window.df_slice)))
    identity["event_id"] = event.event_id
    identity["exchange"] = event.exchange
    identity["symbol"] = event.symbol
    identity["timestamp"] = window.df_slice["timestamp"].to_numpy()
    identity["pump_start_timestamp"] = int(window.ohlcv["timestamp"].iloc[window.pump_start_idx])
    identity["pump_detected_timestamp"] = event.pump_detected_ts
    identity["pump_age_minutes"] = (identity["timestamp"] - event.pump_detected_ts) / 60_000.0
    identity["current_price"] = window.df_slice["close"].to_numpy()
    identity["event_high_so_far"] = window.df_slice["high"].cummax().to_numpy()
    identity["event_low_so_far"] = window.df_slice["low"].cummin().to_numpy()
    # Phase 3A.1 point 13: a liquidity proxy for the symbol-cluster/liquidity-
    # bucket stability dimension - the PRE-detection baseline (pump_start_idx
    # to detect_idx), not the pump's own inflated volume, so a low-liquidity
    # symbol having one huge pump doesn't get miscounted as "high liquidity".
    # Constant across every snapshot row of the same event by construction.
    quote_volume = window.df_slice["quote_volume"] if "quote_volume" in window.df_slice.columns else window.df_slice["volume"] * window.df_slice["close"]
    baseline_quote_volume = float(quote_volume.iloc[: wide_detect + 1].mean()) if wide_detect >= 0 else np.nan
    identity["quote_volume_at_pump_start"] = baseline_quote_volume
    identity["climax_price"] = float(window.df_slice["high"].iloc[window.wide(window.climax_idx_abs)])
    identity["snapshot_abs_idx"] = window.pump_start_idx + np.arange(len(window.df_slice))

    combined = pd.concat([identity] + feature_frames, axis=1)
    snapshots = combined.iloc[wide_detect : wide_end + 1].reset_index(drop=True)
    return snapshots
