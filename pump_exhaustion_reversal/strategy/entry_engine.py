"""Hard gates (spec section 13) + no-trade filters (spec section 19) + entry-
mode evaluation (spec section 14).

evaluate_entry() is a pure function of already-computed scores/flags/market
conditions - it is the SAME function a Stage-2 live scanner would call, and
the backtest simulator calls it identically bar-by-bar. Score alone can
never open a trade (spec section 12): every mode also requires its own
structural confirmations no matter how high the exhaustion score is.

`signals` is a flat dict[str, bool | None] with these keys (None = unknown/
unavailable, never treated as satisfied):
  structure_break, failed_reclaim, lower_high, cvd_divergence,
  sweep_rejection, new_local_low, avwap_loss
- structure_break/failed_reclaim/sweep_rejection/cvd_divergence/avwap_loss
  come from exhaustion_detector.ExhaustionFlags (as_dict(), renamed);
  lower_high/new_local_low come from features.structure.post_break_signals().
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import AppConfig
from .catalyst_filter import CatalystStatus

ENTRY_MODE_REQUIREMENT_TO_SIGNAL = {
    "require_structure_break": "structure_break",
    "require_failed_reclaim": "failed_reclaim",
    "require_lower_high": "lower_high",
    "require_cvd_confirmation": "cvd_divergence",
    "require_sweep_rejection": "sweep_rejection",
    "require_cvd_divergence": "cvd_divergence",
    "require_break_of_new_local_low": "new_local_low",
    "require_avwap_loss": "avwap_loss",
}


@dataclass
class MarketConditionInputs:
    """Everything hard gates + no-trade filters need beyond scores/flags."""

    spread_ok: bool = True
    liquidity_ok: bool = True
    symbol_locked: bool = False
    btc_regime_ok: bool = True
    price_still_hh_hl: bool = False
    volume_expanding_on_new_highs: bool = False
    oi_expanding_with_price_and_no_rollover: bool = False
    spot_cvd_confirming_highs: bool = False
    fresh_listing_without_structure_break: bool = False
    data_confidence: str = "HIGH"  # "HIGH" | "LOW"


@dataclass
class HardGateResult:
    passed: bool
    failed_checks: list[str] = field(default_factory=list)


@dataclass
class EntrySignal:
    valid: bool
    mode: str | None = None
    reasons: list[str] = field(default_factory=list)
    exhaustion_score: float = 0.0


def evaluate_hard_gates(
    pump_score: float,
    exhaustion_score: float,
    structure_confirmed: bool,
    catalyst_status: CatalystStatus,
    market: MarketConditionInputs,
    cfg: AppConfig,
) -> HardGateResult:
    g = cfg.hard_gates
    failed = []
    if g.require_pump_score_min and pump_score < g.require_pump_score_min:
        failed.append(f"pump_score {pump_score:.1f} < {g.require_pump_score_min}")
    if g.require_exhaustion_score_min and exhaustion_score < g.require_exhaustion_score_min:
        failed.append(f"exhaustion_score {exhaustion_score:.1f} < {g.require_exhaustion_score_min}")
    if g.require_structure_confirmation and not structure_confirmed:
        failed.append("structure not confirmed broken")
    if g.require_no_major_positive_news and catalyst_status.major_positive:
        failed.append("major positive news present")
    if g.require_liquidity_ok and not market.liquidity_ok:
        failed.append("liquidity not ok")
    if g.require_spread_ok and not market.spread_ok:
        failed.append("spread too wide")
    if g.require_no_symbol_lock and market.symbol_locked:
        failed.append("symbol locked after failed attempts")
    if g.require_btc_regime_check and not market.btc_regime_ok:
        failed.append("BTC regime check failed")
    return HardGateResult(passed=not failed, failed_checks=failed)


def evaluate_no_trade_filters(market: MarketConditionInputs, cfg: AppConfig) -> list[str]:
    """spec section 19: "overextended != exhausted" - any one of these vetoes
    a trade regardless of how attractive the scores look."""
    n = cfg.no_trade
    reasons = []
    if n.price_still_hh_hl and market.price_still_hh_hl:
        reasons.append("price_still_hh_hl")
    if n.volume_expanding_on_new_highs and market.volume_expanding_on_new_highs:
        reasons.append("volume_expanding_on_new_highs")
    if n.oi_expanding_with_price_and_no_rollover and market.oi_expanding_with_price_and_no_rollover:
        reasons.append("oi_expanding_with_price_and_no_rollover")
    if n.spot_cvd_confirming_highs and market.spot_cvd_confirming_highs:
        reasons.append("spot_cvd_confirming_highs")
    if n.spread_too_wide and not market.spread_ok:
        reasons.append("spread_too_wide")
    if n.liquidity_too_thin and not market.liquidity_ok:
        reasons.append("liquidity_too_thin")
    if n.btc_extreme_trend_against_short and not market.btc_regime_ok:
        reasons.append("btc_extreme_trend_against_short")
    if n.symbol_locked_after_failed_attempts and market.symbol_locked:
        reasons.append("symbol_locked_after_failed_attempts")
    if n.data_confidence_too_low and market.data_confidence == "LOW":
        reasons.append("data_confidence_too_low")
    if n.fresh_listing_without_structure_break and market.fresh_listing_without_structure_break:
        reasons.append("fresh_listing_without_structure_break")
    return reasons


def _mode_requirements_met(mode_cfg, signals: dict[str, bool | None], exhaustion_score: float) -> tuple[bool, list[str]]:
    missing = []
    if exhaustion_score < mode_cfg.min_exhaustion_score:
        missing.append(f"exhaustion_score {exhaustion_score:.1f} < {mode_cfg.min_exhaustion_score}")
    for req_field, signal_key in ENTRY_MODE_REQUIREMENT_TO_SIGNAL.items():
        if getattr(mode_cfg, req_field, False) and signals.get(signal_key) is not True:
            missing.append(f"{req_field} not satisfied (signal {signal_key}={signals.get(signal_key)!r})")
    return (len(missing) == 0), missing


def evaluate_entry(
    pump_score: float,
    exhaustion_score: float,
    signals: dict[str, bool | None],
    catalyst_status: CatalystStatus,
    market: MarketConditionInputs,
    cfg: AppConfig,
) -> EntrySignal:
    """Tries aggressive -> standard -> conservative (best-R-first, spec
    section 14) and returns the first mode whose hard gates + catalyst gate
    + mode-specific structural requirements all pass."""
    hard = evaluate_hard_gates(
        pump_score, exhaustion_score, signals.get("structure_break") is True, catalyst_status, market, cfg
    )
    if not hard.passed:
        return EntrySignal(False, None, hard.failed_checks, exhaustion_score)

    no_trade_reasons = evaluate_no_trade_filters(market, cfg)
    if no_trade_reasons:
        return EntrySignal(False, None, no_trade_reasons, exhaustion_score)

    catalyst_ok_by_mode = {
        "aggressive": catalyst_status.aggressive_allowed,
        "standard": catalyst_status.standard_allowed,
        "conservative": catalyst_status.conservative_allowed in ("yes", "only_if_strong_confirmation"),
    }
    all_missing: list[str] = []
    for mode_name, mode_cfg in (
        ("aggressive", cfg.entry_modes.aggressive),
        ("standard", cfg.entry_modes.standard),
        ("conservative", cfg.entry_modes.conservative),
    ):
        if not mode_cfg.enabled:
            continue
        if not catalyst_ok_by_mode[mode_name]:
            all_missing.append(f"{mode_name}: blocked by catalyst filter")
            continue
        ok, missing = _mode_requirements_met(mode_cfg, signals, exhaustion_score)
        if ok:
            return EntrySignal(True, mode_name, [], exhaustion_score)
        all_missing.extend(f"{mode_name}: {m}" for m in missing)

    return EntrySignal(False, None, all_missing or ["no entry mode enabled"], exhaustion_score)
