"""Pydantic configuration schema for Pump Exhaustion Reversal.

Mirrors the YAML layout in config/default.yaml section-by-section. Every
field has a spec-accurate default, so a YAML file may be partial - missing
keys fall back to these defaults via the deep-merge in load_config().

Stage 1 note: this schema includes sections (e.g. entry_modes, risk,
leverage) that are only meaningful once a live/paper entry engine exists.
They are defined now because the backtest simulator evaluates the exact
same entry/risk/exit logic that Stage 2's live scanner will use - see
strategy/entry_engine.py, strategy/risk_engine.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _weights_sum_to_100(values: dict[str, float], label: str) -> None:
    total = sum(values.values())
    if abs(total - 100) > 1e-6:
        raise ValueError(f"{label} must sum to 100, got {total}")


class StrategyConfig(StrictModel):
    name: str = "Pump Exhaustion Reversal"
    version: str = "1.0"
    live_trading_enabled: bool = False
    paper_trading_enabled: bool = True

    @model_validator(mode="after")
    def _no_live_trading_stage1(self) -> "StrategyConfig":
        if self.live_trading_enabled:
            raise ValueError(
                "live_trading_enabled=True is not allowed in Stage 1 "
                "(research/backtest only, see project plan)."
            )
        return self


class UniverseConfig(StrictModel):
    exchanges: list[str] = Field(default_factory=lambda: ["bybit", "binance"])
    market_type: str = "linear_perp"
    quote_asset: str = "USDT"
    min_24h_quote_volume_usd: float = 2_000_000
    preferred_min_24h_quote_volume_usd: float = 15_000_000
    max_spread_bps: float = 20
    min_symbol_age_days: int = 14
    exclude_stablecoins: bool = True
    exclude_leveraged_tokens: bool = True
    exclude_symbols_with_trading_halts: bool = True


class PumpDetectorConfig(StrictModel):
    windows: list[str] = Field(default_factory=lambda: ["5m", "15m", "1h", "4h", "24h"])
    min_pump_score: float = 70
    min_return_zscore: float = 3.0
    min_volume_zscore: float = 3.0
    min_abnormal_return_15m: float = 0.08
    min_abnormal_return_1h: float = 0.15
    min_abnormal_return_4h: float = 0.25
    min_abnormal_return_24h: float = 0.40
    use_btc_adjusted_return: bool = True
    beta_lookback_days: int = 30


class PumpScoreWeights(StrictModel):
    abnormal_return: float = 20
    volume_anomaly: float = 15
    volatility_expansion: float = 10
    liquidation_spike: float = 20
    oi_anomaly: float = 15
    perp_spot_imbalance: float = 10
    liquidity_marketcap_context: float = 10

    @model_validator(mode="after")
    def _check_sum(self) -> "PumpScoreWeights":
        _weights_sum_to_100(self.model_dump(), "pump_score_weights")
        return self


class CatalystFilterConfig(StrictModel):
    enabled: bool = True
    block_aggressive_if_positive_news_within_hours: float = 12
    block_standard_if_positive_news_within_hours: float = 6
    allow_conservative_after_hours: float = 3
    require_manual_review_for_major_listing: bool = True


class StructureConfig(StrictModel):
    pivot_left_bars: int = 2
    pivot_right_bars: int = 2
    min_swing_distance_atr: float = 0.35
    min_swing_distance_bps: float = 30
    require_close_below_higher_low: bool = True
    close_confirmation_bars: int = 1
    allow_wick_break: bool = False


class SweepConfig(StrictModel):
    min_sweep_bps: float = 15
    max_rejection_bars: int = 5
    close_back_below_previous_high: bool = True
    upper_wick_body_ratio_min: float = 2.0


class FailedReclaimConfig(StrictModel):
    enabled: bool = True
    reclaim_zone_bps: float = 35
    max_bars_after_break: int = 24
    require_lower_high: bool = True
    require_close_below_reclaim_zone: bool = True


class VolumeDecayConfig(StrictModel):
    threshold: float = 0.50
    strong_threshold: float = 0.35


class CVDConfig(StrictModel):
    enabled: bool = True
    divergence_tolerance: float = 0.02
    min_cvd_lookback_bars: int = 20
    require_cvd_rollover_for_aggressive: bool = False
    require_cvd_rollover_for_standard: bool = True


class OIConfig(StrictModel):
    rollover_lookback: str = "30m"
    min_oi_drop_pct_after_peak: float = 5
    danger_oi_rise_pct_on_drop: float = 8


class LiquidationsConfig(StrictModel):
    enabled: bool = True
    allow_proxy_mode: bool = True
    short_liq_spike_zscore_min: float = 5
    depletion_ratio_threshold: float = 0.35
    long_liq_confirmation_after_break: bool = True


class AVWAPConfig(StrictModel):
    enabled: bool = True
    anchors: list[str] = Field(
        default_factory=lambda: ["pump_start", "breakout_candle", "climax_candle"]
    )
    require_loss_for_standard: bool = False
    add_score_if_lost: bool = True


class FundingConfig(StrictModel):
    overheated_percentile: float = 90
    extreme_percentile: float = 97
    max_score_weight: float = 3


class RSIConfig(StrictModel):
    enabled: bool = True
    length: int = 14
    divergence_weight_low: bool = True
    max_score_weight: float = 2


class ExhaustionScoreWeights(StrictModel):
    structure_break: float = 20
    failed_reclaim_lower_high: float = 15
    cvd_bearish_divergence: float = 15
    oi_rollover: float = 12
    volume_decay: float = 10
    short_liquidation_exhaustion: float = 10
    sweep_rejection: float = 8
    avwap_loss: float = 5
    funding_long_crowding: float = 3
    rsi_divergence: float = 2

    @model_validator(mode="after")
    def _check_sum(self) -> "ExhaustionScoreWeights":
        _weights_sum_to_100(self.model_dump(), "exhaustion_score_weights")
        return self


class ExhaustionThresholds(StrictModel):
    watch: float = 50
    armed: float = 65
    trade: float = 75
    high_confidence: float = 85


class ReversalAttemptsConfig(StrictModel):
    """Stage 2A: within-event retry of the exhaustion-watch cycle after a
    dynamic post-break invalidation, before any trade ever triggers - NOT
    the same concept as RiskConfig.max_symbol_attempts below (that governs
    cross-event symbol-lock after losing TRADES/stop-outs). Only
    new_pump_high has a well-defined new anchor to retry from (the bar that
    made the new high); bullish_hl_reformation/strong_reclaim have no new
    price action to re-anchor to, so a naive retry would just re-fire the
    same check next bar and burn the attempt budget with no real progress -
    left as independently configurable, not blanket-enabled, for future A/B.
    """

    enabled: bool = True
    max_reversal_attempts: int = 2
    retry_on_new_pump_high: bool = True
    retry_on_bullish_hl_reformation: bool = False
    retry_on_strong_reclaim: bool = False


class HardGatesConfig(StrictModel):
    require_pump_score_min: float = 70
    # Lowered from the spec's literal 75 to 70 (the minimum across the three
    # entry modes' own min_exhaustion_score - see EntryModesConfig below). At
    # 75, conservative mode's own stated threshold of 70 (spec section 14.3)
    # was unreachable, since this hard gate is checked before any
    # mode-specific check and rejects everything below its own value
    # regardless of a mode's more lenient threshold. A spec-internal-
    # consistency fix (project plan / README), not a loosening of any single
    # mode's own requirements - aggressive (80) and standard (75) are
    # unaffected; only conservative (70) actually becomes reachable now.
    require_exhaustion_score_min: float = 70
    require_structure_confirmation: bool = True
    require_no_major_positive_news: bool = True
    require_liquidity_ok: bool = True
    require_spread_ok: bool = True
    require_no_symbol_lock: bool = True
    require_btc_regime_check: bool = True


class EntryModeSettings(StrictModel):
    enabled: bool
    min_exhaustion_score: float
    risk_multiplier: float
    require_structure_break: bool = False
    require_failed_reclaim: bool = False
    require_lower_high: bool = False
    require_cvd_confirmation: bool = False
    require_sweep_rejection: bool = False
    require_cvd_divergence: bool = False
    require_break_of_new_local_low: bool = False
    require_avwap_loss: bool = False


class EntryModesConfig(StrictModel):
    aggressive: EntryModeSettings = Field(
        default_factory=lambda: EntryModeSettings(
            enabled=False,
            min_exhaustion_score=80,
            risk_multiplier=0.50,
            require_sweep_rejection=True,
            require_cvd_divergence=True,
            require_structure_break=False,
        )
    )
    standard: EntryModeSettings = Field(
        default_factory=lambda: EntryModeSettings(
            enabled=True,
            min_exhaustion_score=75,
            risk_multiplier=1.00,
            require_structure_break=True,
            require_failed_reclaim=True,
            require_lower_high=True,
            require_cvd_confirmation=True,
        )
    )
    conservative: EntryModeSettings = Field(
        default_factory=lambda: EntryModeSettings(
            enabled=True,
            min_exhaustion_score=70,
            risk_multiplier=0.75,
            require_structure_break=True,
            require_failed_reclaim=True,
            require_break_of_new_local_low=True,
            require_avwap_loss=False,
        )
    )


class NoTradeConfig(StrictModel):
    positive_major_news: bool = True
    fresh_listing_without_structure_break: bool = True
    spot_cvd_confirming_highs: bool = True
    price_still_hh_hl: bool = True
    volume_expanding_on_new_highs: bool = True
    oi_expanding_with_price_and_no_rollover: bool = True
    spread_too_wide: bool = True
    liquidity_too_thin: bool = True
    btc_extreme_trend_against_short: bool = True
    symbol_locked_after_failed_attempts: bool = True
    data_confidence_too_low: bool = True


class StopsConfig(StrictModel):
    atr_timeframe: str = "5m"
    atr_length: int = 14
    atr_buffer_multiplier: float = 0.25
    min_stop_bps: float = 40
    max_stop_bps: float = 1200


class RiskConfig(StrictModel):
    initial_risk_per_trade_pct: float = 0.20
    min_risk_per_trade_pct: float = 0.10
    max_risk_per_trade_pct: float = 0.25
    proven_edge_max_risk_pct: float = 0.50
    max_concurrent_positions: int = 3
    max_same_cluster_positions: int = 1
    max_daily_loss_pct: float = 1.00
    max_weekly_loss_pct: float = 3.00
    max_symbol_attempts: int = 2
    symbol_lock_hours_after_two_stops: float = 6
    no_martingale: bool = True
    no_average_up_losers: bool = True


class LeverageConfig(StrictModel):
    default_max_leverage: float = 3
    absolute_max_leverage: float = 5
    allow_higher_only_manual: bool = True


class TakeProfitConfig(StrictModel):
    tp1_r_multiple: float = 1.0
    tp1_close_pct: float = 30
    move_stop_after_tp1: str = "breakeven_or_reduce_risk"
    tp2_target: str = "pump_avwap_or_breakout_area"
    tp2_close_pct: float = 30
    tp3_target: str = "0.5_pump_retracement_or_trailing_structure"
    tp3_close_pct: float = 40


class TimeStopConfig(StrictModel):
    enabled: bool = True
    standard_minutes: float = 60
    aggressive_minutes: float = 30
    conservative_minutes: float = 90
    exit_if_no_progress_r: float = 0.25
    exit_if_oi_reaccelerates: bool = True


class ExecutionModelConfig(StrictModel):
    entry_type: str = "next_bar_open_or_limit_retest"
    default_slippage_bps: float = 10
    high_vol_slippage_multiplier: float = 2.5
    stop_slippage_bps: float = 20
    taker_fee_bps: float = 5
    maker_fee_bps: float = 2
    funding_enabled: bool = True
    max_volume_participation_pct: float = 2


class BacktestConfig(StrictModel):
    start_date: str = "2025-01-01"
    end_date: str = "2026-08-25"
    timeframe_base: str = "1m"
    train_pct: float = 0.60
    validation_pct: float = 0.20
    oos_pct: float = 0.20
    include_fees: bool = True
    include_slippage: bool = True
    include_funding: bool = True
    prevent_lookahead: bool = True

    @model_validator(mode="after")
    def _splits_sum_to_one(self) -> "BacktestConfig":
        total = self.train_pct + self.validation_pct + self.oos_pct
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/validation/oos pct must sum to 1.0, got {total}")
        return self


class AcceptanceConfig(StrictModel):
    min_oos_trades: int = 100
    preferred_oos_trades: int = 300
    oos_profit_factor_min: float = 1.25
    preferred_profit_factor: float = 1.40
    oos_expectancy_r_min: float = 0.10
    max_drawdown_r_reasonable: bool = True
    no_single_trade_dominates_profit: bool = True
    slippage_stress_still_positive: bool = True
    news_filter_improves_tail_risk: bool = True
    paper_trading_days_min: int = 30
    paper_trading_preferred_days: int = 60


class AppConfig(StrictModel):
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    pump_detector: PumpDetectorConfig = Field(default_factory=PumpDetectorConfig)
    pump_score_weights: PumpScoreWeights = Field(default_factory=PumpScoreWeights)
    catalyst_filter: CatalystFilterConfig = Field(default_factory=CatalystFilterConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    failed_reclaim: FailedReclaimConfig = Field(default_factory=FailedReclaimConfig)
    volume_decay: VolumeDecayConfig = Field(default_factory=VolumeDecayConfig)
    cvd: CVDConfig = Field(default_factory=CVDConfig)
    oi: OIConfig = Field(default_factory=OIConfig)
    liquidations: LiquidationsConfig = Field(default_factory=LiquidationsConfig)
    avwap: AVWAPConfig = Field(default_factory=AVWAPConfig)
    funding: FundingConfig = Field(default_factory=FundingConfig)
    rsi: RSIConfig = Field(default_factory=RSIConfig)
    exhaustion_score_weights: ExhaustionScoreWeights = Field(
        default_factory=ExhaustionScoreWeights
    )
    exhaustion_thresholds: ExhaustionThresholds = Field(default_factory=ExhaustionThresholds)
    reversal_attempts: ReversalAttemptsConfig = Field(default_factory=ReversalAttemptsConfig)
    hard_gates: HardGatesConfig = Field(default_factory=HardGatesConfig)
    entry_modes: EntryModesConfig = Field(default_factory=EntryModesConfig)
    no_trade: NoTradeConfig = Field(default_factory=NoTradeConfig)
    stops: StopsConfig = Field(default_factory=StopsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    leverage: LeverageConfig = Field(default_factory=LeverageConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    time_stop: TimeStopConfig = Field(default_factory=TimeStopConfig)
    execution_model: ExecutionModelConfig = Field(default_factory=ExecutionModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(*paths: str | Path | None) -> AppConfig:
    """Load and deep-merge YAML file(s) on top of full schema defaults.

    Each path is optional (None entries are skipped, so callers can pass an
    optional overlay conditionally). Later paths override earlier ones.
    Merging happens at the raw-dict level *before* validation, so a YAML
    file may specify just a handful of leaf keys (e.g. only
    entry_modes.aggressive.min_exhaustion_score) without needing to repeat
    sibling keys - unlike pydantic's own default_factory, which only
    applies when an entire nested block is absent.
    """
    merged: dict[str, Any] = AppConfig().model_dump(mode="python")
    for p in paths:
        if p is None:
            continue
        text = Path(p).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        merged = _deep_merge(merged, data)
    return AppConfig.model_validate(merged)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
BACKTEST_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "backtest.yaml"
