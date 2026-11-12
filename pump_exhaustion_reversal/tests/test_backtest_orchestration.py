from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest import event_study, reports, walk_forward
from pump_exhaustion_reversal.backtest.simulator import EventRecord
from pump_exhaustion_reversal.config import AcceptanceConfig
from pump_exhaustion_reversal.tests.test_metrics import _event, _trade
from pump_exhaustion_reversal.tests.test_simulator import _FAST_WARMUP_CFG, _flat_btc_ohlcv, _pump_and_reversal_ohlcv


# ---- event_study.py --------------------------------------------------


def test_events_to_frames_splits_trade_rows_out():
    t = _trade(1.0)
    events = [_event(trade=t), _event(trade=None)]
    events_df, trades_df = event_study.events_to_frames(events)
    assert len(events_df) == 2
    assert len(trades_df) == 1
    assert "trade" not in events_df.columns
    assert trades_df.iloc[0]["final_r"] == pytest.approx(1.0)


def test_events_to_frames_empty_list():
    events_df, trades_df = event_study.events_to_frames([])
    assert events_df.empty
    assert trades_df.empty


def test_run_event_study_over_small_universe_runs_end_to_end():
    n = 1200
    ohlcv_a = _pump_and_reversal_ohlcv(n, seed=21, first_cycle_start=700)
    ohlcv_b = _pump_and_reversal_ohlcv(n, seed=22, first_cycle_start=700)
    btc = _flat_btc_ohlcv(n)
    universe = [
        {"exchange": "bybit", "symbol": "AUSDT", "ohlcv": ohlcv_a, "btc_ohlcv": btc, "liquidity_score": 80.0},
        {"exchange": "bybit", "symbol": "BUSDT", "ohlcv": ohlcv_b, "btc_ohlcv": btc, "liquidity_score": 80.0},
    ]
    events_df, trades_df = event_study.run_event_study(universe, "5m", _FAST_WARMUP_CFG, horizon_bars=200)
    assert isinstance(events_df, pd.DataFrame)
    assert isinstance(trades_df, pd.DataFrame)
    if not events_df.empty:
        assert set(events_df["symbol"].unique()) <= {"AUSDT", "BUSDT"}


# ---- walk_forward.py --------------------------------------------------


def test_split_chronological_is_time_ordered_not_shuffled():
    events_df = pd.DataFrame({"pump_detected_ts": [50, 10, 40, 20, 30], "event_id": ["e5", "e1", "e4", "e2", "e3"]})
    splits = walk_forward.split_chronological(events_df, train_pct=0.6, validation_pct=0.2, oos_pct=0.2)
    assert list(splits["train"]["pump_detected_ts"]) == [10, 20, 30]
    assert list(splits["validation"]["pump_detected_ts"]) == [40]
    assert list(splits["oos"]["pump_detected_ts"]) == [50]
    assert splits["train"]["pump_detected_ts"].max() <= splits["validation"]["pump_detected_ts"].min()
    assert splits["validation"]["pump_detected_ts"].max() <= splits["oos"]["pump_detected_ts"].min()


def test_split_chronological_rejects_bad_percentages():
    events_df = pd.DataFrame({"pump_detected_ts": [1, 2, 3]})
    with pytest.raises(ValueError):
        walk_forward.split_chronological(events_df, train_pct=0.5, validation_pct=0.3, oos_pct=0.3)


def test_label_market_regime_classifies_trend_and_calendar():
    ts_weekday = int(pd.Timestamp("2026-08-24T10:00:00Z").timestamp() * 1000)  # Monday, EU-ish hour
    ts_weekend = int(pd.Timestamp("2026-08-22T02:00:00Z").timestamp() * 1000)  # Saturday, Asia-ish hour
    events_df = pd.DataFrame(
        {
            "pump_detected_ts": [ts_weekday, ts_weekday, ts_weekend],
            "btc_return_concurrent": [0.05, -0.05, 0.0],
        }
    )
    labeled = walk_forward.label_market_regime(events_df)
    assert labeled.loc[0, "regime_trend"] == "bull"
    assert labeled.loc[1, "regime_trend"] == "bear"
    assert labeled.loc[2, "regime_trend"] == "sideways"
    assert labeled.loc[0, "regime_weekday"] == "weekday"
    assert labeled.loc[2, "regime_weekday"] == "weekend"
    assert labeled.loc[0, "regime_session"] == "eu"
    assert labeled.loc[2, "regime_session"] == "asia"


def test_label_market_regime_empty_df():
    out = walk_forward.label_market_regime(pd.DataFrame(columns=["pump_detected_ts", "btc_return_concurrent"]))
    assert out.empty


def test_parameter_sweep_smoke():
    n = 1200
    ohlcv = _pump_and_reversal_ohlcv(n, seed=31, first_cycle_start=700)
    btc = _flat_btc_ohlcv(n)
    universe = [{"exchange": "bybit", "symbol": "CUSDT", "ohlcv": ohlcv, "btc_ohlcv": btc, "liquidity_score": 80.0}]
    grid = {"pump_detector.min_pump_score": [60, 90]}
    result = walk_forward.parameter_sweep(universe, "5m", _FAST_WARMUP_CFG, grid, horizon_bars=200)
    assert len(result) == 2
    assert set(result["pump_detector.min_pump_score"]) == {60, 90}
    assert "n_trades" in result.columns


# ---- reports.py --------------------------------------------------


def _passing_metrics():
    return {
        "n_trades": 150, "profit_factor": 1.5, "expectancy_r": 0.15,
        "no_single_trade_dominates_profit": True, "win_rate": 0.4,
    }


def test_evaluate_acceptance_all_pass_is_go():
    result = reports.evaluate_acceptance(_passing_metrics(), None, None, AcceptanceConfig())
    assert result["go"] is True
    assert all(c["passed"] for c in result["checks"].values())


def test_evaluate_acceptance_too_few_trades_is_no_go():
    m = _passing_metrics()
    m["n_trades"] = 5
    result = reports.evaluate_acceptance(m, None, None, AcceptanceConfig())
    assert result["go"] is False
    assert result["checks"]["min_oos_trades"]["passed"] is False


def test_evaluate_acceptance_dominant_trade_blocks_go_even_if_pf_high():
    m = _passing_metrics()
    m["no_single_trade_dominates_profit"] = False
    result = reports.evaluate_acceptance(m, None, None, AcceptanceConfig())
    assert result["go"] is False


def test_evaluate_acceptance_slippage_stress_can_flip_a_pass_to_fail():
    passing = _passing_metrics()
    stress_negative = {"expectancy_r": -0.05}
    result = reports.evaluate_acceptance(passing, None, stress_negative, AcceptanceConfig())
    assert result["go"] is False
    assert result["checks"]["slippage_stress_still_positive"]["passed"] is False


def test_render_markdown_report_does_not_crash_on_nan_and_contains_verdict():
    empty_metrics = {
        "n_trades": 0, "win_rate": float("nan"), "avg_r": float("nan"), "median_r": float("nan"),
        "profit_factor": float("nan"), "max_drawdown_r": 0.0, "max_consecutive_losses": 0,
        "avg_hold_minutes": float("nan"), "avg_mae_r": float("nan"), "avg_mfe_r": float("nan"),
        "avg_slippage_bps_of_notional": float("nan"), "stop_overrun_bps": float("nan"),
        "tail_loss_r_p05": float("nan"), "best_trade_r": float("nan"), "worst_trade_r": float("nan"),
        "total_fees": 0.0, "total_funding": 0.0, "net_pnl_usd": 0.0, "no_single_trade_dominates_profit": True,
    }
    event_metrics = {"n_pump_events": 0, "n_events_triggered_trade": 0, "trigger_rate": float("nan"), "events_by_class": {}}
    acceptance = reports.evaluate_acceptance(empty_metrics, None, None, AcceptanceConfig())
    md = reports.render_markdown_report(
        run_id="test123", event_metrics=event_metrics, overall_trade_metrics=empty_metrics,
        oos_trade_metrics=empty_metrics, by_pump_class={}, by_entry_mode={}, acceptance=acceptance,
        limitations=["no liquidation feed - proxy only", "sparse news coverage"],
    )
    assert "GO / NO-GO" in md
    assert "NO-GO" in md  # 0 trades must fail acceptance
    assert "no liquidation feed" in md


def test_save_report_writes_file(tmp_path, monkeypatch):
    import pump_exhaustion_reversal.data.storage as storage_mod

    monkeypatch.setattr(storage_mod, "REPORTS_ROOT", tmp_path)
    path = reports.save_report("# hello", "runX")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "# hello"
