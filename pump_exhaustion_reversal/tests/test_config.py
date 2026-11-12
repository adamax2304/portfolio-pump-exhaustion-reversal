from __future__ import annotations

import pytest
from pydantic import ValidationError

from pump_exhaustion_reversal.config import (
    AppConfig,
    ExhaustionScoreWeights,
    PumpScoreWeights,
    load_config,
)


def test_defaults_construct_and_validate():
    cfg = AppConfig()
    assert cfg.strategy.name == "Pump Exhaustion Reversal"
    assert cfg.strategy.live_trading_enabled is False
    assert cfg.pump_detector.min_pump_score == 70
    assert cfg.exhaustion_thresholds.trade == 75


def test_load_default_yaml_matches_schema_defaults(default_config_path):
    cfg = load_config(default_config_path)
    assert cfg == AppConfig()


def test_backtest_overlay_merges_partially(default_config_path, backtest_config_path):
    cfg = load_config(default_config_path, backtest_config_path)
    # overridden by backtest.yaml
    assert cfg.backtest.timeframe_base == "5m"
    assert cfg.backtest.start_date == "2024-09-01"
    # untouched sections still carry spec defaults
    assert cfg.risk.initial_risk_per_trade_pct == 0.20
    assert cfg.entry_modes.standard.min_exhaustion_score == 75


def test_partial_nested_override_does_not_drop_siblings(tmp_path, default_config_path):
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("entry_modes:\n  aggressive:\n    min_exhaustion_score: 82\n", encoding="utf-8")
    cfg = load_config(default_config_path, overlay)
    assert cfg.entry_modes.aggressive.min_exhaustion_score == 82
    # siblings not mentioned in overlay must survive the merge
    assert cfg.entry_modes.aggressive.enabled is False
    assert cfg.entry_modes.aggressive.risk_multiplier == 0.50
    assert cfg.entry_modes.aggressive.require_sweep_rejection is True


def test_pump_score_weights_must_sum_to_100():
    with pytest.raises(ValidationError):
        PumpScoreWeights(
            abnormal_return=50,
            volume_anomaly=15,
            volatility_expansion=10,
            liquidation_spike=20,
            oi_anomaly=15,
            perp_spot_imbalance=10,
            liquidity_marketcap_context=10,
        )


def test_exhaustion_score_weights_must_sum_to_100():
    with pytest.raises(ValidationError):
        ExhaustionScoreWeights(structure_break=99)


def test_unknown_key_is_rejected(tmp_path, default_config_path):
    overlay = tmp_path / "typo.yaml"
    overlay.write_text("universe:\n  min_24h_quote_volum_usd: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(default_config_path, overlay)


def test_live_trading_cannot_be_enabled_in_stage1(tmp_path, default_config_path):
    overlay = tmp_path / "live.yaml"
    overlay.write_text("strategy:\n  live_trading_enabled: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(default_config_path, overlay)


def test_backtest_splits_must_sum_to_one(tmp_path, default_config_path):
    overlay = tmp_path / "splits.yaml"
    overlay.write_text("backtest:\n  train_pct: 0.5\n  validation_pct: 0.3\n  oos_pct: 0.3\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(default_config_path, overlay)
