"""Backtest simulator (spec sections 22.1-22.4): the SAME detector/classifier/
exhaustion/entry/risk/exit code the state machine's pseudocode (section 26)
describes, replayed bar-by-bar over history. Never uses a bar's information
before that bar's close is known; entries fill on the NEXT bar's open.

One call to simulate_symbol() walks a single symbol's full history and
produces both event-level records (spec 22.3's event study - every detected
pump, whether or not it ever triggered a trade) and trade-level records
(spec 22.4 - the subset that reached SHORT_TRIGGERED), from one shared
lifecycle per pump so the two views can never disagree with each other about
what actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.feature_store import build_feature_frame
from ..features import avwap as avwap_feat
from ..features import cvd as cvd_feat
from ..features import funding as funding_feat
from ..features import liquidations as liq_feat
from ..features import oi as oi_feat
from ..features import returns as returns_feat
from ..features import rsi as rsi_feat
from ..features import structure as struct_feat
from ..features import volume as volume_feat
from ..data.normalizer import parse_timeframe_ms
from . import execution_model as em
from ..strategy import catalyst_filter, entry_engine, exit_engine, pump_classifier, pump_detector, risk_engine
from ..strategy.exhaustion_detector import ExhaustionFlags, ExhaustionInputs, compute_exhaustion_flags_at, compute_exhaustion_score_at, classify_exhaustion_state
from ..strategy.state_machine import PumpReversalEvent, PumpReversalState, StepContext, step_event

DEFAULT_MAX_EVENT_HORIZON_BARS = 2000


@dataclass
class TradeRecord:
    event_id: str
    exchange: str
    symbol: str
    entry_mode: str
    pump_class: str
    entry_ts: int
    entry_price: float
    stop_price_initial: float
    exit_ts: int
    exit_price: float
    exit_reason: str  # "TP1"|"TP2"|"TP3"|"STOP"|"TIME_STOP"|"FORCED_HORIZON_END"
    quantity: float
    notional: float
    final_r: float
    mae_r: float
    mfe_r: float
    fees_paid: float
    slippage_cost: float
    funding_cost: float
    net_pnl_usd: float
    hold_minutes: float


@dataclass
class EventRecord:
    event_id: str
    exchange: str
    symbol: str
    pump_start_ts: int
    pump_detected_ts: int
    pump_peak_ts: int
    pump_score: float
    pump_class: str
    pump_class_confidence: str
    data_confidence: str
    volume_zscore: float
    oi_zscore: float
    funding_percentile: float
    short_liq_zscore: float
    btc_return_concurrent: float
    news_flag: bool
    final_state: str
    exhaustion_score_peak: float
    subsequent_return_15m: float
    subsequent_return_1h: float
    subsequent_return_4h: float
    subsequent_return_24h: float
    max_adverse_after_detection: float  # positive = price rose further against a short thesis
    max_favorable_after_detection: float  # positive = price fell in the short's favor
    invalidation_reason: str | None = None  # only set when final_state == "INVALIDATED" - the state machine's own reason string
    # Research/diagnostic-phase fields (Stage 1.5) - never read by entry_engine/risk/exit,
    # only surfaced for the classifier/exhaustion-feature audits.
    pump_class_reasons: list[str] = field(default_factory=list)
    pump_class_data_confidence_notes: list[str] = field(default_factory=list)
    # Pre-seeded with all 10 component names (values None/NaN) rather than a
    # bare {} default: pyarrow refuses to write a struct-typed parquet column
    # with zero child fields ("Cannot write struct type ... with no child
    # field"), which a plain {} would produce for any EventRecord built
    # without explicitly setting these two fields (e.g. via a test helper, or
    # an event whose score never left its initial 0.0 peak).
    exhaustion_component_first_true_bar: dict[str, int | None] = field(
        default_factory=lambda: dict.fromkeys(ExhaustionFlags().as_dict())
    )
    exhaustion_component_contribution_at_peak: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(ExhaustionFlags().as_dict(), np.nan)
    )
    oi_drop_pct_from_peak_at_peak: float = np.nan
    volume_decay_ratio_at_peak: float = np.nan
    cvd_divergence_magnitude_at_peak: float = np.nan
    # Bar indices (local to this event's own df_slice - not directly comparable
    # across events without pairing with pump_start_ts/timestamp) of the first
    # structure break and of whichever dynamic-invalidation check fired first,
    # if any. Both are within-horizon, already-known-at-the-time bars, not a
    # forward/future lookup - safe to capture in the live loop like the fields
    # above, unlike research/forward_study.py's eventual-peak fields.
    structure_break_bar_local_idx: int | None = None
    dynamic_invalidation_bar_local_idx: int | None = None
    # Stage 2A: the FINAL reversal attempt's climax bar (wide/df_slice-local
    # coordinates) - deliberately distinct from pump_peak_ts above, which
    # always stays attempt-1's climax even when a retry re-anchors to a new,
    # later high. attempt_count==1 means they refer to the same bar.
    climax_local_idx: int | None = None
    attempt_count: int = 1
    # Phase 2B: one entry per attempt (index 0 = attempt 1, etc.), always
    # length == attempt_count - lets research code compare attempt #1 vs #2
    # (spec section 8) instead of only ever seeing the final attempt's state.
    attempt_climax_local_idxs: list[int] = field(default_factory=list)
    attempt_structure_break_local_idxs: list[int | None] = field(default_factory=list)
    attempt_exhaustion_score_peaks: list[float] = field(default_factory=list)
    attempt_invalidation_reason_codes: list[str | None] = field(default_factory=list)
    trade: TradeRecord | None = None


@dataclass
class _PostPumpInputs:
    """Structure/reversal-confirmation signals recomputed fresh starting at
    structure_start_idx = max(0, climax_idx - pivot_left_bars) - the pump's
    own price top, minus a small pivot-confirmation buffer - never
    contaminated by whatever happened earlier in the pre-detection lookback
    window (Stage 2A fix: see project plan "Temporal/Event Architecture
    Fix"). climax_idx/structure_start_idx are in the WIDE (df_slice-local)
    coordinate system; every other field here is reindexed to start at 0 -
    translate any bar index derived from these back with `+
    structure_start_idx` before storing it on EventRecord.

    One of these is built per reversal attempt (see _run_single_event) -
    attempt 1 from _build_event_inputs, each retry from a fresh
    _build_post_pump_inputs() call anchored at the new climax that
    invalidated the prior attempt.
    """

    climax_idx: int
    structure_start_idx: int
    post_pump_df: pd.DataFrame
    post_pump_atr: pd.Series
    structure: pd.DataFrame
    failed_reclaim: pd.Series
    sweep_rejection: pd.Series
    post_break: pd.DataFrame
    cvd_divergence: pd.Series | None
    cvd_rollover: pd.Series | None
    cvd_divergence_magnitude: pd.Series | None
    oi_rollover: pd.Series | None
    oi_drop_pct_from_peak: pd.Series | None
    rsi_divergence: pd.Series | None


@dataclass
class _EventInputs:
    df_slice: pd.DataFrame
    cvd_slice: pd.DataFrame
    post_pump: _PostPumpInputs  # attempt 1's - _run_single_event tracks the current attempt separately
    exhaustion_inputs: ExhaustionInputs  # attempt 1's - rebuilt fresh per retry, see _build_exhaustion_inputs_for_attempt
    avwap_frame: pd.DataFrame
    local_detect_idx: int
    pump_start_idx: int
    oi_danger_series: pd.Series | None = None
    # Continuous ("strength") companion to the boolean exhaustion flags above -
    # research/diagnostic-phase only, not read by entry_engine/risk/exit.
    volume_decay_ratio: pd.Series | None = None
    # Wide (never re-scoped to a specific attempt's post-pump window - see
    # _build_post_pump_inputs's docstring for why) series needed to reslice a
    # fresh ExhaustionInputs on each reversal-attempt retry.
    volume_decayed: pd.Series | None = None
    liq_depletion_ratio: pd.Series | None = None
    avwap_loss_columns: list[pd.Series] = field(default_factory=list)
    funding_overheated: pd.Series | None = None


def _wick_body_ratio(row: pd.Series) -> float:
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["close"], row["open"])
    if body > 0:
        return float(upper_wick / body)
    return float(upper_wick) if upper_wick > 0 else 0.0


def _build_post_pump_inputs(
    df_slice: pd.DataFrame, cvd_slice: pd.DataFrame, climax_idx: int, bar_interval_ms: int, cfg: AppConfig,
) -> _PostPumpInputs:
    """Recomputes every "did the pump's mechanism break" signal on a slice
    starting at structure_start_idx = max(0, climax_idx - pivot_left_bars) -
    never on the wide pre-detection lookback (see _PostPumpInputs's
    docstring). Called once for attempt 1 (from _build_event_inputs) and
    again on each reversal-attempt retry with a new climax_idx (the bar that
    invalidated the prior attempt) - see _run_single_event.

    The `pivot_left_bars` buffer (not climax_idx itself) is the actual start:
    features.structure.find_pivots needs that many bars of left-context to
    ever confirm the climax bar itself as a pivot high, otherwise the first
    reference high the reslice can see would be some arbitrary, lower, later
    high - silently weakening the "new pump high" check this whole fix is
    meant to sharpen.
    """
    structure_start_idx = max(0, climax_idx - cfg.structure.pivot_left_bars)
    post_pump_df = df_slice.iloc[structure_start_idx:].reset_index(drop=True)
    post_pump_cvd = cvd_slice.iloc[structure_start_idx:].reset_index(drop=True)

    post_pump_atr = (
        struct_feat.atr(post_pump_df, cfg.stops.atr_length).bfill().fillna(post_pump_df["high"] - post_pump_df["low"])
    )
    structure = struct_feat.compute_structure(post_pump_df, post_pump_atr, cfg.structure)
    failed_reclaim = struct_feat.detect_failed_reclaim(post_pump_df, structure, cfg.failed_reclaim)
    sweep_rejection = struct_feat.detect_sweep_rejection(post_pump_df, structure, cfg.sweep)
    post_break = struct_feat.post_break_signals(post_pump_df, structure)

    cvd_divergence = cvd_feat.cvd_bearish_divergence(
        post_pump_df["high"], post_pump_cvd["cvd_rolling"], cfg.cvd.min_cvd_lookback_bars, cfg.cvd.divergence_tolerance
    ) if cfg.cvd.enabled else None
    cvd_rollover = cvd_feat.cvd_rollover(post_pump_cvd["cvd_slope"]) if cfg.cvd.enabled else None
    cvd_divergence_magnitude = cvd_feat.cvd_divergence_magnitude(
        post_pump_df["high"], post_pump_cvd["cvd_rolling"], cfg.cvd.min_cvd_lookback_bars
    ) if cfg.cvd.enabled else None

    oi_rollover_series = None
    oi_drop_pct_from_peak = None
    if post_pump_cvd["open_interest"].notna().any():
        oi_lookback_bars = returns_feat.bars_for_window(parse_timeframe_ms(cfg.oi.rollover_lookback), bar_interval_ms)
        oi_frame = oi_feat.detect_oi_rollover(
            post_pump_df["close"], post_pump_cvd["open_interest"], max(oi_lookback_bars, 2),
            cfg.oi.min_oi_drop_pct_after_peak, cfg.oi.danger_oi_rise_pct_on_drop,
        )
        oi_rollover_series = oi_frame["oi_rollover"]
        oi_drop_pct_from_peak = oi_frame["oi_drop_pct_from_peak"]

    rsi_divergence = None
    if cfg.rsi.enabled and post_pump_cvd["rsi"].notna().any():
        rsi_divergence = rsi_feat.rsi_bearish_divergence(post_pump_df["high"], post_pump_cvd["rsi"], cfg.cvd.min_cvd_lookback_bars)

    return _PostPumpInputs(
        climax_idx=climax_idx, structure_start_idx=structure_start_idx,
        post_pump_df=post_pump_df, post_pump_atr=post_pump_atr,
        structure=structure, failed_reclaim=failed_reclaim, sweep_rejection=sweep_rejection, post_break=post_break,
        cvd_divergence=cvd_divergence, cvd_rollover=cvd_rollover, cvd_divergence_magnitude=cvd_divergence_magnitude,
        oi_rollover=oi_rollover_series, oi_drop_pct_from_peak=oi_drop_pct_from_peak, rsi_divergence=rsi_divergence,
    )


def _build_exhaustion_inputs_for_attempt(
    post_pump: _PostPumpInputs, volume_decayed: pd.Series, liq_depletion_ratio: pd.Series | None,
    avwap_loss_columns_wide: list[pd.Series], funding_overheated: pd.Series | None,
) -> ExhaustionInputs:
    """Combines one attempt's post-pump structure/CVD/OI/RSI signals with the
    wide (never re-scoped - see _EventInputs's docstring) volume-decay/
    liquidation/AVWAP/funding series, resliced to the SAME structure_start_idx
    so every field in the returned bundle shares one coordinate system -
    compute_exhaustion_flags_at reads all of them at a single index i."""
    s = post_pump.structure_start_idx
    return ExhaustionInputs(
        structure=post_pump.structure, failed_reclaim=post_pump.failed_reclaim, sweep_rejection=post_pump.sweep_rejection,
        cvd_divergence=post_pump.cvd_divergence, cvd_rollover=post_pump.cvd_rollover, oi_rollover=post_pump.oi_rollover,
        volume_decayed=volume_decayed.iloc[s:].reset_index(drop=True),
        liq_depletion_ratio=(liq_depletion_ratio.iloc[s:].reset_index(drop=True) if liq_depletion_ratio is not None else None),
        avwap_loss_columns=[c.iloc[s:].reset_index(drop=True) for c in avwap_loss_columns_wide],
        funding_overheated=(funding_overheated.iloc[s:].reset_index(drop=True) if funding_overheated is not None else None),
        rsi_divergence=post_pump.rsi_divergence,
    )


def _build_event_inputs(
    ohlcv: pd.DataFrame, feature_frame: pd.DataFrame, detect_idx: int, timeframe: str, cfg: AppConfig,
    horizon_bars: int,
) -> _EventInputs:
    bar_interval_ms = parse_timeframe_ms(timeframe)
    windows_bars = [returns_feat.bars_for_window(parse_timeframe_ms(w), bar_interval_ms) for w in cfg.pump_detector.windows]
    lookback_bars = max(windows_bars) if windows_bars else 20
    pump_start_idx = max(0, detect_idx - lookback_bars)
    # horizon_bars is room AFTER detection for the exhaustion-watch/trade
    # lifecycle to play out - anchored on detect_idx, not pump_start_idx, so
    # it can never be shorter than the lookback used to find pump_start_idx
    # (which would otherwise put detect_idx itself outside the slice).
    slice_end_idx = min(len(ohlcv) - 1, detect_idx + horizon_bars)

    df_slice = ohlcv.iloc[pump_start_idx : slice_end_idx + 1].reset_index(drop=True)
    cvd_slice = feature_frame.iloc[pump_start_idx : slice_end_idx + 1].reset_index(drop=True)
    local_detect_idx = detect_idx - pump_start_idx

    # The pump's own price top up to (and including) detection - shared by
    # the post-pump reslicing below, EventRecord.pump_peak_ts, and
    # _classify_at_detection's climax-candle wick check. Previously computed
    # independently in all three places; unified here so they can never drift.
    climax_idx = int(df_slice["high"].iloc[: local_detect_idx + 1].to_numpy().argmax())
    post_pump = _build_post_pump_inputs(df_slice, cvd_slice, climax_idx, bar_interval_ms, cfg)

    peak_volume = float(df_slice["volume"].iloc[: local_detect_idx + 1].max())
    decay_ratio = volume_feat.volume_decay_ratio(df_slice["volume"], peak_volume)
    volume_decayed = volume_feat.volume_decay_flags(decay_ratio, cfg.volume_decay.threshold, cfg.volume_decay.strong_threshold)["volume_decayed"]

    liq_depletion_ratio = None
    if cvd_slice["liq_proxy_intensity"].notna().any():
        peak_liq = float(cvd_slice["liq_proxy_intensity"].iloc[: local_detect_idx + 1].max())
        liq_depletion_ratio = liq_feat.liquidation_depletion_ratio(cvd_slice["liq_proxy_intensity"], peak_liq)

    # oi_danger_series stays wide deliberately: it's a live risk-management
    # check consumed only by _simulate_trade's time-stop during an ALREADY
    # open trade, not a "did the reversal get confirmed" signal - a different
    # semantic from oi_rollover/oi_drop_pct_from_peak above (which do move to
    # post-pump, inside _build_post_pump_inputs).
    oi_danger_series = None
    if cvd_slice["open_interest"].notna().any():
        oi_lookback_bars = returns_feat.bars_for_window(parse_timeframe_ms(cfg.oi.rollover_lookback), bar_interval_ms)
        oi_frame_wide = oi_feat.detect_oi_rollover(
            df_slice["close"], cvd_slice["open_interest"], max(oi_lookback_bars, 2),
            cfg.oi.min_oi_drop_pct_after_peak, cfg.oi.danger_oi_rise_pct_on_drop,
        )
        oi_danger_series = oi_frame_wide["oi_danger_new_shorts"]

    avwap_frame = pd.DataFrame(index=df_slice.index)
    avwap_loss_cols: list[pd.Series] = []
    if cfg.avwap.enabled:
        breakout_idx = 0
        if local_detect_idx > 0:
            rets = df_slice["close"].iloc[: local_detect_idx + 1].pct_change().abs()
            breakout_idx = int(rets.fillna(0).to_numpy().argmax())
        anchors = {"pump_start": 0, "breakout_candle": breakout_idx, "climax_candle": climax_idx}
        avwap_frame = avwap_feat.compute_avwap_features(df_slice, anchors)
        avwap_loss_cols = [avwap_frame[c] for c in avwap_frame.columns if c.startswith("avwap_loss_")]

    funding_overheated = None
    if cvd_slice["funding_percentile"].notna().any():
        funding_overheated = funding_feat.funding_heat_flags(
            cvd_slice["funding_percentile"], cfg.funding.overheated_percentile, cfg.funding.extreme_percentile
        )["funding_overheated"]

    exhaustion_inputs = _build_exhaustion_inputs_for_attempt(
        post_pump, volume_decayed, liq_depletion_ratio, avwap_loss_cols, funding_overheated
    )
    return _EventInputs(
        df_slice=df_slice, cvd_slice=cvd_slice, post_pump=post_pump, exhaustion_inputs=exhaustion_inputs,
        avwap_frame=avwap_frame, local_detect_idx=local_detect_idx, pump_start_idx=pump_start_idx,
        oi_danger_series=oi_danger_series, volume_decay_ratio=decay_ratio,
        volume_decayed=volume_decayed, liq_depletion_ratio=liq_depletion_ratio,
        avwap_loss_columns=avwap_loss_cols, funding_overheated=funding_overheated,
    )


def _classify_at_detection(
    inputs: _EventInputs, cfg: AppConfig, catalyst_status: catalyst_filter.CatalystStatus, liquidity_score: float
) -> pump_classifier.PumpClassification:
    row = inputs.cvd_slice.iloc[inputs.local_detect_idx]
    oi_series = inputs.cvd_slice["open_interest"]
    oi_pct_change = np.nan
    if oi_series.notna().iloc[0] and oi_series.notna().iloc[inputs.local_detect_idx] and oi_series.iloc[0] != 0:
        oi_pct_change = (oi_series.iloc[inputs.local_detect_idx] / oi_series.iloc[0] - 1) * 100

    funding_series = inputs.cvd_slice["funding_rate"]
    funding_pct_series = inputs.cvd_slice["funding_percentile"]
    funding_pre_pump_percentile = funding_pct_series.iloc[0] if funding_pct_series.notna().any() else np.nan
    funding_rising = False
    if funding_series.notna().iloc[0] and funding_series.notna().iloc[inputs.local_detect_idx]:
        funding_rising = bool(funding_series.iloc[inputs.local_detect_idx] > funding_series.iloc[0])

    cvd_conf = inputs.cvd_slice["cvd_confidence"].iloc[0] if "cvd_confidence" in inputs.cvd_slice else "low"
    cvd_slope_val = inputs.cvd_slice["cvd_slope"].iloc[inputs.local_detect_idx]
    cvd_slope_positive = bool(cvd_slope_val > 0) if not pd.isna(cvd_slope_val) else False

    wick_ratio = _wick_body_ratio(inputs.df_slice.iloc[inputs.post_pump.climax_idx])

    return pump_classifier.classify_pump(
        has_recent_positive_catalyst=catalyst_status.major_positive,
        short_liq_zscore=row.get("short_liq_zscore", np.nan),
        short_liq_depletion_ratio=np.nan,  # not knowable yet at the moment of detection
        oi_pct_change_during_pump=oi_pct_change,
        oi_zscore=row.get("oi_zscore", np.nan),
        funding_pre_pump_percentile=funding_pre_pump_percentile,
        funding_direction_rising=funding_rising,
        cvd_confidence=cvd_conf,
        cvd_slope_positive=cvd_slope_positive,
        liquidity_confidence=("HIGH" if liquidity_score >= 70 else "LOW"),
        upper_wick_body_ratio_climax=wick_ratio,
        pump_duration_bars=inputs.local_detect_idx,
        typical_pump_duration_bars=max(inputs.local_detect_idx, 1),
        catalyst_requires_manual_review=catalyst_status.requires_manual_review,
    )


def _simulate_trade(
    event: PumpReversalEvent, event_id: str, exchange: str, symbol: str, inputs: _EventInputs, active: _PostPumpInputs,
    fill_local_i: int, entry_mode: str, cfg: AppConfig, equity: float, bar_interval_ms: int,
) -> tuple[TradeRecord, int]:
    """Fills at fill_local_i's OPEN (the bar after the trigger bar closed -
    lookahead-safe), then walks forward resolving TP1/TP2/TP3/stop/time-stop
    against each subsequent bar's high/low. A bar where both the stop and a
    TP level could have been touched is resolved conservatively - stop first
    (matches the proven same-candle-ambiguity pattern from crypto-ai-desk).

    Stop reference/ATR come from `active` (the current reversal attempt's
    climax-anchored post-pump structure, not the wide pre-detection lookback)
    - Stage 2A fix: a stop placed against a swing from hours before the
    current pump leg would be economically meaningless, not just a research
    metric error (confirmed with the user before extending the fix here)."""
    df_slice = inputs.df_slice
    n_slice = len(df_slice)
    entry_open = float(df_slice["open"].iloc[fill_local_i])

    post_pump_fill_i = fill_local_i - active.structure_start_idx
    ref_row = active.structure.iloc[post_pump_fill_i]
    stop_reference = (
        float(ref_row["swing_high_level"])
        if not pd.isna(ref_row["swing_high_level"])
        else float(active.post_pump_df["high"].iloc[: post_pump_fill_i + 1].max())
    )
    atr_value = active.post_pump_atr.iloc[post_pump_fill_i]
    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = float((active.post_pump_df["high"] - active.post_pump_df["low"]).iloc[: post_pump_fill_i + 1].mean())
    stop_result = risk_engine.compute_structural_stop(entry_open, stop_reference, atr_value, cfg.stops, is_short=True)

    vol_ratio_at_entry = inputs.cvd_slice["realized_vol_ratio"].iloc[fill_local_i]
    high_vol = bool(vol_ratio_at_entry > 2.0) if not pd.isna(vol_ratio_at_entry) else False
    fill = em.simulate_entry_fill(entry_open, cfg.execution_model, high_vol)
    entry_price = fill["fill_price"]

    risk_mult = getattr(cfg.entry_modes, entry_mode).risk_multiplier
    size = risk_engine.compute_position_size(
        equity, entry_price, stop_result.stop_price, cfg.risk.initial_risk_per_trade_pct * risk_mult, cfg.leverage
    )
    quantity_total, notional_total = size.quantity, size.notional
    initial_risk_per_unit = abs(entry_price - stop_result.stop_price)

    include_fees = cfg.backtest.include_fees
    total_fees = em.compute_fee(notional_total, cfg.execution_model, is_maker=False) if include_fees else 0.0
    total_slippage_cost = abs(entry_open - entry_price) * quantity_total
    total_funding = 0.0

    tp_plan = risk_engine.compute_take_profit_plan(entry_price, stop_result.stop_price, cfg.take_profit, is_short=True)
    pump_high = float(df_slice["high"].iloc[: inputs.local_detect_idx + 1].max())
    pump_low = float(df_slice["low"].iloc[: inputs.local_detect_idx + 1].min())
    avwap_cols = [c for c in inputs.avwap_frame.columns if c.startswith("avwap_") and not c.startswith("avwap_loss_")]
    pump_avwap = None
    if avwap_cols and not pd.isna(inputs.avwap_frame[avwap_cols[0]].iloc[fill_local_i]):
        pump_avwap = float(inputs.avwap_frame[avwap_cols[0]].iloc[fill_local_i])
    context = {"pump_avwap": pump_avwap, "breakout_level": stop_reference, "pump_high": pump_high, "pump_low": pump_low}
    tp2_price = risk_engine.resolve_symbolic_tp_target(tp_plan.tp2_target_desc, context, entry_price, is_short=True)
    tp3_price = risk_engine.resolve_symbolic_tp_target(tp_plan.tp3_target_desc, context, entry_price, is_short=True)

    trade_state = exit_engine.TradeState(
        entry_price=entry_price, entry_ts=int(df_slice["timestamp"].iloc[fill_local_i]),
        stop_price=stop_result.stop_price, initial_risk_per_unit=initial_risk_per_unit, mode=entry_mode, is_short=True,
    )

    remaining_qty_frac = 1.0
    realized_r_weighted = 0.0
    mae_r = mfe_r = 0.0
    exit_reason = exit_price = None
    exit_local_i = fill_local_i
    tp1_done = tp2_done = False
    current_stop = stop_result.stop_price
    oi_danger_series = inputs.oi_danger_series
    funding_rate_series = inputs.cvd_slice["funding_rate"]

    for local_j in range(fill_local_i, n_slice):
        bar = df_slice.iloc[local_j]
        ts_j = int(bar["timestamp"])
        high_j, low_j, close_j = float(bar["high"]), float(bar["low"]), float(bar["close"])

        mae_r = max(mae_r, (high_j - entry_price) / initial_risk_per_unit) if initial_risk_per_unit else mae_r
        mfe_r = max(mfe_r, (entry_price - low_j) / initial_risk_per_unit) if initial_risk_per_unit else mfe_r

        if cfg.execution_model.funding_enabled and cfg.backtest.include_funding:
            funding_rate = funding_rate_series.iloc[local_j]
            if not pd.isna(funding_rate):
                proration = bar_interval_ms / (8 * 3_600_000)
                total_funding += em.compute_funding_cost(
                    notional_total * remaining_qty_frac, float(funding_rate) * proration, is_short=True
                )

        stop_hit = high_j >= current_stop
        tp_price_pending = tp_plan.tp1_price if not tp1_done else (tp2_price if not tp2_done else tp3_price)
        tp_hit = low_j <= tp_price_pending
        if stop_hit and tp_hit:
            tp_hit = False  # same-candle ambiguity resolved conservatively: stop wins

        if stop_hit:
            stop_fill = em.simulate_stop_fill(current_stop, cfg.execution_model)
            exit_price = stop_fill["fill_price"]
            exit_reason = "STOP"
            realized_r_weighted += ((entry_price - exit_price) / initial_risk_per_unit) * remaining_qty_frac
            leg_notional = notional_total * remaining_qty_frac
            total_fees += em.compute_fee(leg_notional, cfg.execution_model, is_maker=False) if include_fees else 0.0
            total_slippage_cost += abs(current_stop - exit_price) * (quantity_total * remaining_qty_frac)
            exit_local_i = local_j
            break

        if tp_hit:
            tp_fill = em.simulate_tp_fill(tp_price_pending, cfg.execution_model, high_vol)
            fill_price_tp = tp_fill["fill_price"]
            r_this_leg = (entry_price - fill_price_tp) / initial_risk_per_unit if initial_risk_per_unit else 0.0
            if not tp1_done:
                close_frac = min(tp_plan.tp1_close_pct / 100.0, remaining_qty_frac)
                tp1_done = True
                current_stop = risk_engine.stop_after_tp1(entry_price, current_stop, tp_plan.move_stop_after_tp1)
                label = "TP1"
            elif not tp2_done:
                close_frac = min(tp_plan.tp2_close_pct / 100.0, remaining_qty_frac)
                tp2_done = True
                current_stop = risk_engine.trail_stop(current_stop, tp2_price, is_short=True)
                label = "TP2"
            else:
                close_frac = remaining_qty_frac
                label = "TP3"

            realized_r_weighted += r_this_leg * close_frac
            leg_notional = notional_total * close_frac
            total_fees += em.compute_fee(leg_notional, cfg.execution_model, is_maker=False) if include_fees else 0.0
            total_slippage_cost += abs(tp_price_pending - fill_price_tp) * (quantity_total * close_frac)
            remaining_qty_frac -= close_frac
            exit_price, exit_reason, exit_local_i = fill_price_tp, label, local_j
            trade_state.tp1_hit = tp1_done

            if remaining_qty_frac <= 1e-9 or label == "TP3":
                break
            continue

        oi_reaccelerated = bool(oi_danger_series.iloc[local_j]) if oi_danger_series is not None and local_j < len(oi_danger_series) else False
        trade_state.stop_price = current_stop
        fired, reason = exit_engine.check_time_stop(trade_state, ts_j, close_j, oi_reaccelerated, cfg.time_stop)
        if fired:
            time_fill = (
                em.simulate_tp_fill(close_j, cfg.execution_model, high_vol)
                if close_j < entry_price
                else em.simulate_stop_fill(close_j, cfg.execution_model)
            )
            exit_price = time_fill["fill_price"]
            realized_r_weighted += ((entry_price - exit_price) / initial_risk_per_unit) * remaining_qty_frac
            leg_notional = notional_total * remaining_qty_frac
            total_fees += em.compute_fee(leg_notional, cfg.execution_model, is_maker=False) if include_fees else 0.0
            total_slippage_cost += abs(close_j - exit_price) * (quantity_total * remaining_qty_frac)
            exit_reason, exit_local_i = f"TIME_STOP ({reason})", local_j
            break

    if exit_reason is None:
        last_bar = df_slice.iloc[-1]
        exit_price = float(last_bar["close"])
        realized_r_weighted += ((entry_price - exit_price) / initial_risk_per_unit if initial_risk_per_unit else 0.0) * remaining_qty_frac
        leg_notional = notional_total * remaining_qty_frac
        total_fees += em.compute_fee(leg_notional, cfg.execution_model, is_maker=False) if include_fees else 0.0
        exit_reason, exit_local_i = "FORCED_HORIZON_END", n_slice - 1

    exit_ts = int(df_slice["timestamp"].iloc[exit_local_i])
    hold_minutes = (exit_ts - trade_state.entry_ts) / 60_000
    event.transition_to(PumpReversalState.CLOSED, exit_ts, f"exit: {exit_reason}")

    trade = TradeRecord(
        event_id=event_id, exchange=exchange, symbol=symbol, entry_mode=entry_mode, pump_class=event.pump_class,
        entry_ts=trade_state.entry_ts, entry_price=entry_price, stop_price_initial=stop_result.stop_price,
        exit_ts=exit_ts, exit_price=exit_price, exit_reason=exit_reason, quantity=quantity_total, notional=notional_total,
        final_r=realized_r_weighted, mae_r=mae_r, mfe_r=mfe_r, fees_paid=total_fees, slippage_cost=total_slippage_cost,
        funding_cost=total_funding, net_pnl_usd=realized_r_weighted * size.risk_dollars - total_fees - total_funding,
        hold_minutes=hold_minutes,
    )
    return trade, exit_local_i


@dataclass
class MarketBaseline:
    """Stage 1 has no live orderbook/spread feed, so these are simple fixed
    per-symbol inputs (typically derived once from the universe filter's
    liquidity_confidence) rather than a per-bar signal."""

    spread_ok: bool = True
    liquidity_ok: bool = True
    btc_regime_ok: bool = True


def _post_break_invalidation_reason(
    local_i: int, structure: pd.DataFrame, df_slice: pd.DataFrame, cfg: AppConfig
) -> tuple[str, str] | None:
    """Spec section 20: "any active state -> INVALIDATED if price continues
    up and invalidates setup", made concrete for the post-break waiting
    period specifically. Exists so that widening failed_reclaim's time
    window (max_bars_after_break) doesn't just let a dead setup sit around
    passively outlasting a timer - a setup that's genuinely still exhausting
    must keep looking weak on every one of these checks, every bar, or it's
    cut loose immediately rather than at the timeout. Checked only after a
    structure break has actually happened; pre-break behaviour (still just
    watching a pump run up) is unchanged.

    `local_i`/`structure`/`df_slice` must all share ONE coordinate system -
    the caller passes the current attempt's post-pump-scoped structure/frame
    (see _PostPumpInputs), never the wide pre-detection lookback, so a break
    from an unrelated earlier pump leg can never be read here (Stage 2A fix).

    Returns (reason_code, reason_text): reason_code drives Stage 2A's
    attempt-retry eligibility lookup (cfg.reversal_attempts); reason_text is
    byte-identical to the three strings this function has always returned,
    so existing substring-matching (e.g. research/structure_audit.py,
    research/negative_analysis.py) keeps working unchanged.

    Three checks, each a direct reading of already-computed structure state
    (no new signal invented for this):
      - new pump high: price has now exceeded the pre-break swing high -
        the original pump resumed/extended, it did not exhaust.
      - bullish reformation: a fresh swing low has confirmed ABOVE the
        broken level - the downtrend itself is failing structurally.
      - strong reclaim: price closed back above the reclaim zone for
        several consecutive bars (spec's own close_confirmation_bars) -
        a genuine reclaim, the opposite of a failed one.
    A fourth spec-listed case ("a new independent pump leg") is not checked
    separately - in practice it shows up as a new pump high (the first
    check) almost every time, and treating it separately would mostly
    duplicate that check rather than add real coverage.
    """
    if not bool(structure["structure_break_bar"].iloc[: local_i + 1].any()):
        return None

    broken_level = structure["broken_level"].iloc[local_i]
    if pd.isna(broken_level):
        return None
    pre_break_high = structure["pre_break_swing_high"].iloc[local_i]
    high = float(df_slice["high"].iloc[local_i])

    if not pd.isna(pre_break_high) and high > pre_break_high:
        return (
            "new_pump_high",
            "price made a new pump high beyond the pre-break swing high - the pump resumed, it did not exhaust",
        )

    swing_low_level = structure["swing_low_level"].iloc[local_i]
    swing_low_label = structure["swing_low_label"].iloc[local_i]
    if swing_low_label == "HL" and not pd.isna(swing_low_level) and swing_low_level > broken_level:
        return ("bullish_hl_reformation", "a new higher low confirmed above the broken level - bullish structure reforming")

    ccb = cfg.structure.close_confirmation_bars
    bars_since_break = structure["bars_since_structure_break"].iloc[local_i]
    if not pd.isna(bars_since_break) and bars_since_break >= ccb and local_i - ccb + 1 >= 0:
        zone_top = broken_level * (1 + cfg.failed_reclaim.reclaim_zone_bps / 10_000)
        recent_closes = df_slice["close"].iloc[local_i - ccb + 1 : local_i + 1]
        if len(recent_closes) == ccb and (recent_closes > zone_top).all():
            return ("strong_reclaim", "strong reclaim: closed back above the reclaim zone for several consecutive bars")

    return None


def _is_retryable_invalidation(reason_code: str, cfg: AppConfig) -> bool:
    """Whether a dynamic post-break invalidation should re-arm the event
    (retry as a new attempt) rather than terminate it - see
    config.py:ReversalAttemptsConfig for why only new_pump_high defaults to
    retryable (the other two reasons have no new price action to re-anchor
    a fresh attempt to)."""
    ra = cfg.reversal_attempts
    if not ra.enabled:
        return False
    return {
        "new_pump_high": ra.retry_on_new_pump_high,
        "bullish_hl_reformation": ra.retry_on_bullish_hl_reformation,
        "strong_reclaim": ra.retry_on_strong_reclaim,
    }.get(reason_code, False)


def _run_single_event(
    event_id: str, exchange: str, symbol: str, detect_idx: int, ohlcv: pd.DataFrame, feature_frame: pd.DataFrame,
    pump_frame: pd.DataFrame, timeframe: str, cfg: AppConfig, catalyst_df: pd.DataFrame | None,
    liquidity_score: float, equity: float, market_baseline: MarketBaseline, horizon_bars: int,
) -> tuple[EventRecord, int]:
    """Runs one pump event's full lifecycle. Returns (event_record,
    last_pump_start_relative_index_consumed) - the caller advances its outer
    scan index past this to avoid re-detecting the same pump."""
    bar_interval_ms = parse_timeframe_ms(timeframe)
    inputs = _build_event_inputs(ohlcv, feature_frame, detect_idx, timeframe, cfg, horizon_bars)
    df_slice, cvd_slice = inputs.df_slice, inputs.cvd_slice
    n_slice = len(df_slice)
    local_detect_idx = inputs.local_detect_idx
    active = inputs.post_pump  # current reversal attempt's post-pump-scoped inputs - reassigned on retry
    active_exhaustion_inputs = inputs.exhaustion_inputs

    detect_ts = int(df_slice["timestamp"].iloc[local_detect_idx])
    pump_score_at_detect = float(pump_frame["pump_score"].iloc[detect_idx])
    data_confidence = str(pump_frame["data_confidence"].iloc[detect_idx])

    catalyst_status = catalyst_filter.check_catalyst(catalyst_df, symbol, detect_ts, cfg.catalyst_filter)
    classification = _classify_at_detection(inputs, cfg, catalyst_status, liquidity_score)

    event = PumpReversalEvent(symbol=symbol, exchange=exchange)
    event.pump_class = classification.pump_class.value
    step_event(event, StepContext(ts=detect_ts, pump_detected=True, pump_class=event.pump_class))

    exhaustion_score_peak = 0.0
    last_local_idx = local_detect_idx
    trade_record: TradeRecord | None = None
    # Research/diagnostic-phase tracking (Stage 1.5) - read-only observation of
    # the same per-bar values already computed below, never fed back into
    # entry_engine/risk/exit.
    component_first_true_bar: dict[str, int | None] = dict.fromkeys(ExhaustionFlags().as_dict())
    # Pre-seeded with all 10 keys (NaN, not omitted) rather than starting {}
    # empty: an event whose score never rises above its initial 0.0 peak (all
    # components missing/inapplicable the whole time) would otherwise leave
    # this dict permanently {} - and pyarrow cannot write a struct column
    # with zero child fields to parquet ("Cannot write struct type ... with
    # no child field"), which would crash the save step after a full run.
    component_contribution_at_peak: dict[str, float] = dict.fromkeys(ExhaustionFlags().as_dict(), np.nan)
    oi_drop_pct_from_peak_at_peak = np.nan
    volume_decay_ratio_at_peak = np.nan
    cvd_divergence_magnitude_at_peak = np.nan
    structure_break_bar_local_idx: int | None = None
    dynamic_invalidation_bar_local_idx: int | None = None
    # Phase 2B: per-attempt history (one entry per CONCLUDED-or-final attempt,
    # appended either on retry - see the attempt_invalidated block below - or
    # once after the loop ends for whichever attempt is active when the event
    # finally resolves). exhaustion_score_peak above stays the whole-saga max
    # (unchanged, existing semantics); current_attempt_score_peak tracks just
    # THIS attempt's own peak, resetting on each retry.
    current_attempt_score_peak = 0.0
    attempt_climax_local_idxs: list[int] = [inputs.post_pump.climax_idx]
    attempt_structure_break_local_idxs: list[int | None] = []
    attempt_exhaustion_score_peaks: list[float] = []
    attempt_invalidation_reason_codes: list[str | None] = []
    dynamic_invalidation_result: tuple[str, str] | None = None  # safety default if the loop body never executes at all

    local_i = local_detect_idx
    while local_i < n_slice and event.state in (
        PumpReversalState.WATCHING, PumpReversalState.EXHAUSTION_FORMING, PumpReversalState.ARMED_SHORT,
    ):
        # Translates the wide (df_slice-local) loop index into the current
        # attempt's post-pump-scoped coordinate system (see _PostPumpInputs) -
        # always >= 0 since structure_start_idx <= local_detect_idx <= local_i
        # by construction (climax is found within [0, local_detect_idx]).
        post_pump_local_i = local_i - active.structure_start_idx
        ts = int(df_slice["timestamp"].iloc[local_i])
        flags = compute_exhaustion_flags_at(post_pump_local_i, active_exhaustion_inputs, cfg)
        score_result = compute_exhaustion_score_at(flags, cfg)
        for _component_name, _component_val in flags.as_dict().items():
            if _component_val and component_first_true_bar[_component_name] is None:
                component_first_true_bar[_component_name] = local_i
        if score_result["score"] > exhaustion_score_peak:
            component_contribution_at_peak = dict(score_result["components"])
            oi_drop_pct_from_peak_at_peak = (
                float(active.oi_drop_pct_from_peak.iloc[post_pump_local_i])
                if active.oi_drop_pct_from_peak is not None and not pd.isna(active.oi_drop_pct_from_peak.iloc[post_pump_local_i])
                else np.nan
            )
            volume_decay_ratio_at_peak = (
                float(inputs.volume_decay_ratio.iloc[local_i])
                if inputs.volume_decay_ratio is not None and not pd.isna(inputs.volume_decay_ratio.iloc[local_i])
                else np.nan
            )
            cvd_divergence_magnitude_at_peak = (
                float(active.cvd_divergence_magnitude.iloc[post_pump_local_i])
                if active.cvd_divergence_magnitude is not None and not pd.isna(active.cvd_divergence_magnitude.iloc[post_pump_local_i])
                else np.nan
            )
        exhaustion_score_peak = max(exhaustion_score_peak, score_result["score"])
        current_attempt_score_peak = max(current_attempt_score_peak, score_result["score"])
        exhaustion_state = classify_exhaustion_state(score_result["score"], cfg)

        signals = {
            "structure_break": flags.structure_break,
            "failed_reclaim": flags.failed_reclaim_lower_high,
            "lower_high": bool(active.post_break["lower_high_formed"].iloc[post_pump_local_i]),
            "cvd_divergence": flags.cvd_bearish_divergence,
            "sweep_rejection": flags.sweep_rejection,
            "new_local_low": bool(active.post_break["new_local_low"].iloc[post_pump_local_i]),
            "avwap_loss": flags.avwap_loss,
        }
        structure_intact = bool(active.structure["structure_intact"].iloc[post_pump_local_i])
        market = entry_engine.MarketConditionInputs(
            spread_ok=market_baseline.spread_ok,
            liquidity_ok=market_baseline.liquidity_ok,
            symbol_locked=False,
            btc_regime_ok=market_baseline.btc_regime_ok,
            price_still_hh_hl=structure_intact,
            volume_expanding_on_new_highs=(flags.volume_decay is False and structure_intact),
            oi_expanding_with_price_and_no_rollover=(flags.oi_rollover is False),
            spot_cvd_confirming_highs=False,  # Stage 1: no spot data - never vetoes (see README limitations)
            fresh_listing_without_structure_break=False,
            data_confidence=data_confidence,
        )
        entry_signal = entry_engine.evaluate_entry(
            pump_score_at_detect, score_result["score"], signals, catalyst_status, market, cfg
        )

        # Bounded by post_pump_local_i (not the whole precomputed frame, which
        # extends to slice_end_idx - far beyond the live bar currently being
        # walked) so this never peeks at a break the live walk hasn't reached
        # yet. Re-checked every bar (not a one-shot pre-loop lookup) so a
        # retry's fresh `active` naturally starts this search over.
        if structure_break_bar_local_idx is None:
            hits = active.structure.index[active.structure["structure_break_bar"]].to_numpy()
            hits = hits[hits <= post_pump_local_i]
            if len(hits):
                structure_break_bar_local_idx = int(hits[0]) + active.structure_start_idx

        is_last_bar = local_i == n_slice - 1
        dynamic_invalidation_result = (
            None if entry_signal.valid else _post_break_invalidation_reason(post_pump_local_i, active.structure, active.post_pump_df, cfg)
        )
        # Stage 2A: a retryable reason (new_pump_high by default - see
        # _is_retryable_invalidation) re-arms the event as a new attempt
        # instead of terminating it, as long as the attempt budget isn't
        # already spent. The timeout case (is_last_bar) is never retryable -
        # no new price action happened, there's nothing to re-anchor to.
        retry_eligible = (
            dynamic_invalidation_result is not None
            and _is_retryable_invalidation(dynamic_invalidation_result[0], cfg)
            and event.attempt_count < cfg.reversal_attempts.max_reversal_attempts
        )
        attempt_invalidated = bool(dynamic_invalidation_result) and retry_eligible
        invalidated = (bool(dynamic_invalidation_result) and not retry_eligible) or (is_last_bar and not entry_signal.valid)
        invalidation_reason = (
            dynamic_invalidation_result[1] if dynamic_invalidation_result
            else ("reached event horizon without a valid trigger" if invalidated else None)
        )
        if dynamic_invalidation_result and dynamic_invalidation_bar_local_idx is None:
            dynamic_invalidation_bar_local_idx = local_i

        step_event(
            event,
            StepContext(
                ts=ts, exhaustion_state=exhaustion_state,
                no_trade_reasons=entry_engine.evaluate_no_trade_filters(market, cfg),
                entry_signal_valid=entry_signal.valid, entry_mode=entry_signal.mode,
                invalidated=invalidated, attempt_invalidated=attempt_invalidated,
                invalidation_reason=invalidation_reason,
                feature_snapshot={"exhaustion_score": score_result["score"], "pump_score": pump_score_at_detect},
            ),
        )
        if attempt_invalidated:
            # Re-anchor to the new climax that triggered this retry (the
            # current bar - the only retryable reason by default,
            # new_pump_high, means THIS bar's high is the new pump top) and
            # reset per-attempt tracking so structure_break_bar_local_idx/
            # dynamic_invalidation_bar_local_idx end up reflecting only the
            # FINAL attempt on EventRecord, not attempt 1.
            # exhaustion_component_first_true_bar deliberately does NOT
            # reset here - "did this component ever fire, anywhere in this
            # event's saga" is a different, still-useful question from
            # "the final attempt's own timeline" (see project plan).
            attempt_structure_break_local_idxs.append(structure_break_bar_local_idx)
            attempt_exhaustion_score_peaks.append(current_attempt_score_peak)
            attempt_invalidation_reason_codes.append(dynamic_invalidation_result[0] if dynamic_invalidation_result else None)

            active = _build_post_pump_inputs(df_slice, cvd_slice, local_i, bar_interval_ms, cfg)
            active_exhaustion_inputs = _build_exhaustion_inputs_for_attempt(
                active, inputs.volume_decayed, inputs.liq_depletion_ratio, inputs.avwap_loss_columns, inputs.funding_overheated
            )
            attempt_climax_local_idxs.append(active.climax_idx)
            structure_break_bar_local_idx = None
            dynamic_invalidation_bar_local_idx = None
            current_attempt_score_peak = 0.0
        last_local_idx = local_i
        if event.state in (PumpReversalState.MANAGING, PumpReversalState.INVALIDATED):
            break
        local_i += 1

    # The final (whichever was active when the loop ended, whatever the
    # reason) attempt's own record - appended once here since it never went
    # through the retry branch above.
    attempt_structure_break_local_idxs.append(structure_break_bar_local_idx)
    attempt_exhaustion_score_peaks.append(current_attempt_score_peak)
    attempt_invalidation_reason_codes.append(dynamic_invalidation_result[0] if dynamic_invalidation_result else None)

    if event.state == PumpReversalState.MANAGING:
        fill_local_i = last_local_idx + 1
        if fill_local_i >= n_slice:
            event.transition_to(
                PumpReversalState.INVALIDATED, int(df_slice["timestamp"].iloc[last_local_idx]),
                "no next bar available within horizon to fill the entry",
            )
        else:
            trade_record, exit_local_i = _simulate_trade(
                event, event_id, exchange, symbol, inputs, active, fill_local_i, event.entry_mode, cfg, equity, bar_interval_ms
            )
            last_local_idx = exit_local_i

    subsequent = _subsequent_returns(ohlcv, detect_idx, bar_interval_ms)
    btc_return_concurrent = float(cvd_slice["btc_close"].iloc[local_detect_idx] / cvd_slice["btc_close"].iloc[0] - 1) if not pd.isna(cvd_slice["btc_close"].iloc[0]) and cvd_slice["btc_close"].iloc[0] else np.nan

    horizon_end_local = min(n_slice - 1, last_local_idx)
    post_detect_high = float(df_slice["high"].iloc[local_detect_idx : horizon_end_local + 1].max())
    post_detect_low = float(df_slice["low"].iloc[local_detect_idx : horizon_end_local + 1].min())
    detect_close = float(df_slice["close"].iloc[local_detect_idx])
    max_adverse = (post_detect_high - detect_close) / detect_close if detect_close else np.nan
    max_favorable = (detect_close - post_detect_low) / detect_close if detect_close else np.nan

    event_record = EventRecord(
        event_id=event_id, exchange=exchange, symbol=symbol,
        pump_start_ts=int(df_slice["timestamp"].iloc[0]), pump_detected_ts=detect_ts,
        pump_peak_ts=int(df_slice["timestamp"].iloc[inputs.post_pump.climax_idx]),
        pump_score=pump_score_at_detect, pump_class=classification.pump_class.value,
        pump_class_confidence=classification.confidence, data_confidence=data_confidence,
        volume_zscore=float(cvd_slice["volume_zscore"].iloc[local_detect_idx]) if not pd.isna(cvd_slice["volume_zscore"].iloc[local_detect_idx]) else np.nan,
        oi_zscore=float(cvd_slice["oi_zscore"].iloc[local_detect_idx]) if not pd.isna(cvd_slice["oi_zscore"].iloc[local_detect_idx]) else np.nan,
        funding_percentile=float(cvd_slice["funding_percentile"].iloc[local_detect_idx]) if not pd.isna(cvd_slice["funding_percentile"].iloc[local_detect_idx]) else np.nan,
        short_liq_zscore=float(cvd_slice["short_liq_zscore"].iloc[local_detect_idx]) if not pd.isna(cvd_slice["short_liq_zscore"].iloc[local_detect_idx]) else np.nan,
        btc_return_concurrent=btc_return_concurrent, news_flag=catalyst_status.major_positive,
        final_state=event.state.value, exhaustion_score_peak=exhaustion_score_peak,
        subsequent_return_15m=subsequent.get("15m", np.nan), subsequent_return_1h=subsequent.get("1h", np.nan),
        subsequent_return_4h=subsequent.get("4h", np.nan), subsequent_return_24h=subsequent.get("24h", np.nan),
        max_adverse_after_detection=max_adverse, max_favorable_after_detection=max_favorable,
        invalidation_reason=(event.history[-1].reason if event.state == PumpReversalState.INVALIDATED and event.history else None),
        pump_class_reasons=classification.reasons, pump_class_data_confidence_notes=classification.data_confidence_notes,
        exhaustion_component_first_true_bar=component_first_true_bar,
        exhaustion_component_contribution_at_peak=component_contribution_at_peak,
        oi_drop_pct_from_peak_at_peak=oi_drop_pct_from_peak_at_peak,
        volume_decay_ratio_at_peak=volume_decay_ratio_at_peak,
        cvd_divergence_magnitude_at_peak=cvd_divergence_magnitude_at_peak,
        structure_break_bar_local_idx=structure_break_bar_local_idx,
        dynamic_invalidation_bar_local_idx=dynamic_invalidation_bar_local_idx,
        climax_local_idx=active.climax_idx, attempt_count=event.attempt_count,
        attempt_climax_local_idxs=attempt_climax_local_idxs,
        attempt_structure_break_local_idxs=attempt_structure_break_local_idxs,
        attempt_exhaustion_score_peaks=attempt_exhaustion_score_peaks,
        attempt_invalidation_reason_codes=attempt_invalidation_reason_codes,
        trade=trade_record,
    )
    return event_record, inputs.pump_start_idx + last_local_idx


def _subsequent_returns(ohlcv: pd.DataFrame, detect_idx: int, bar_interval_ms: int) -> dict[str, float]:
    """Fixed-horizon forward returns from the detection bar - retrospective
    event-study analytics only (spec 22.3), never fed into a live decision."""
    n = len(ohlcv)
    detect_close = float(ohlcv["close"].iloc[detect_idx])
    out: dict[str, float] = {}
    for label, ms in (("15m", 15 * 60_000), ("1h", 3_600_000), ("4h", 14_400_000), ("24h", 86_400_000)):
        bars_ahead = max(round(ms / bar_interval_ms), 1)
        target_idx = detect_idx + bars_ahead
        if target_idx < n and detect_close:
            out[label] = float(ohlcv["close"].iloc[target_idx] / detect_close - 1)
        else:
            out[label] = np.nan
    return out


def simulate_symbol(
    exchange: str, symbol: str, ohlcv: pd.DataFrame, btc_ohlcv: pd.DataFrame, timeframe: str, cfg: AppConfig,
    oi_df: pd.DataFrame | None = None, funding_df: pd.DataFrame | None = None, catalyst_df: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None, liquidity_score: float = 50.0, equity: float = 5000.0,
    market_baseline: MarketBaseline | None = None, horizon_bars: int = DEFAULT_MAX_EVENT_HORIZON_BARS,
) -> list[EventRecord]:
    """Walks one symbol's full history once, detecting every pump (spec
    section 7-8, vectorized) and running each one's complete lifecycle (spec
    sections 11-20) through to CLOSED/INVALIDATED/LOCKED. Symbol-lock (spec
    section 16: max_symbol_attempts stop-outs -> lock for
    symbol_lock_hours_after_two_stops) is tracked across events here, since
    it's a cross-event, symbol-level concept.
    """
    market_baseline = market_baseline or MarketBaseline()
    feature_frame = build_feature_frame(ohlcv, btc_ohlcv, timeframe, cfg, oi_df=oi_df, funding_df=funding_df, trades=trades)
    liquidity_confidence_score = liquidity_score
    pump_frame = pump_detector.compute_pump_score_frame(
        feature_frame, liquidity_confidence_score, cfg.pump_detector, cfg.pump_score_weights
    )

    n = len(ohlcv)
    events: list[EventRecord] = []
    recent_stop_timestamps: list[int] = []
    locked_until_ts: int | None = None
    i = 0
    event_seq = 0

    while i < n:
        ts_i = int(ohlcv["timestamp"].iloc[i])
        if locked_until_ts is not None and ts_i < locked_until_ts:
            i += 1
            continue
        if not bool(pump_frame["pump_detected"].iloc[i]):
            i += 1
            continue

        event_seq += 1
        event_id = f"{exchange}:{symbol}:{event_seq}"
        event_record, last_abs_idx = _run_single_event(
            event_id, exchange, symbol, i, ohlcv, feature_frame, pump_frame, timeframe, cfg,
            catalyst_df, liquidity_score, equity, market_baseline, horizon_bars,
        )
        events.append(event_record)

        if event_record.trade is not None and event_record.trade.exit_reason == "STOP":
            recent_stop_timestamps.append(event_record.trade.exit_ts)
            recent_stop_timestamps = [
                t for t in recent_stop_timestamps if event_record.trade.exit_ts - t <= 24 * 3_600_000
            ]
            if len(recent_stop_timestamps) >= cfg.risk.max_symbol_attempts:
                locked_until_ts = event_record.trade.exit_ts + int(cfg.risk.symbol_lock_hours_after_two_stops * 3_600_000)
                recent_stop_timestamps = []

        i = max(last_abs_idx + 1, i + 1)

    return events
