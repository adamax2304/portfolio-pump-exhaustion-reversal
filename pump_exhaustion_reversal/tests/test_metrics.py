from __future__ import annotations

import numpy as np
import pytest

from pump_exhaustion_reversal.backtest import metrics
from pump_exhaustion_reversal.backtest.simulator import EventRecord, TradeRecord


def _trade(final_r, exit_ts=0, entry_ts=0, exit_reason="TP1", pump_class="SHORT_SQUEEZE", entry_mode="standard", **kw):
    defaults = dict(
        event_id="e1", exchange="bybit", symbol="BTCUSDT", entry_mode=entry_mode, pump_class=pump_class,
        entry_ts=entry_ts, entry_price=100.0, stop_price_initial=105.0, exit_ts=exit_ts, exit_price=95.0,
        exit_reason=exit_reason, quantity=10.0, notional=1000.0, final_r=final_r, mae_r=0.5, mfe_r=abs(final_r) + 0.1,
        fees_paid=1.0, slippage_cost=0.5, funding_cost=0.1, net_pnl_usd=final_r * 20, hold_minutes=30.0,
    )
    defaults.update(kw)
    return TradeRecord(**defaults)


def _event(max_adverse=0.0, pump_class="SHORT_SQUEEZE", trade=None, **kw):
    defaults = dict(
        event_id="e1", exchange="bybit", symbol="BTCUSDT", pump_start_ts=0, pump_detected_ts=0, pump_peak_ts=0,
        pump_score=80.0, pump_class=pump_class, pump_class_confidence="high", data_confidence="HIGH",
        volume_zscore=5.0, oi_zscore=np.nan, funding_percentile=np.nan, short_liq_zscore=np.nan,
        btc_return_concurrent=0.0, news_flag=False, final_state="CLOSED", exhaustion_score_peak=80.0,
        subsequent_return_15m=0.0, subsequent_return_1h=0.0, subsequent_return_4h=0.0, subsequent_return_24h=0.0,
        max_adverse_after_detection=max_adverse, max_favorable_after_detection=0.1, trade=trade,
    )
    defaults.update(kw)
    return EventRecord(**defaults)


def test_compute_trade_metrics_empty_is_well_formed():
    result = metrics.compute_trade_metrics([])
    assert result["n_trades"] == 0
    assert result["max_drawdown_r"] == 0.0
    assert result["no_single_trade_dominates_profit"] is True


def test_compute_trade_metrics_basic_stats():
    trades = [_trade(1.0, exit_ts=100), _trade(-1.0, exit_ts=200), _trade(2.0, exit_ts=300), _trade(-0.5, exit_ts=400)]
    result = metrics.compute_trade_metrics(trades)
    assert result["n_trades"] == 4
    assert result["win_rate"] == pytest.approx(0.5)
    assert result["avg_r"] == pytest.approx(0.375)
    assert result["median_r"] == pytest.approx(0.25)  # sorted [-1,-0.5,1,2] -> mean of two middle values
    # profit_factor = gains/losses = (1.0+2.0)/(1.0+0.5) = 3/1.5 = 2.0
    assert result["profit_factor"] == pytest.approx(2.0)
    assert result["best_trade_r"] == pytest.approx(2.0)
    assert result["worst_trade_r"] == pytest.approx(-1.0)


def test_compute_trade_metrics_profit_factor_no_losses_is_inf():
    trades = [_trade(1.0), _trade(2.0)]
    result = metrics.compute_trade_metrics(trades)
    assert result["profit_factor"] == float("inf")


def test_max_drawdown_r_on_a_known_equity_curve():
    # R sequence in time order: +2, +1, -3, +1 -> equity: 2,3,0,1 -> peak 3, trough 0 -> DD=3
    trades = [_trade(2.0, exit_ts=1), _trade(1.0, exit_ts=2), _trade(-3.0, exit_ts=3), _trade(1.0, exit_ts=4)]
    assert metrics._max_drawdown_r(trades) == pytest.approx(3.0)


def test_max_consecutive_losses():
    trades = [_trade(1.0, exit_ts=1), _trade(-1.0, exit_ts=2), _trade(-1.0, exit_ts=3), _trade(-1.0, exit_ts=4), _trade(1.0, exit_ts=5), _trade(-1.0, exit_ts=6)]
    assert metrics._max_consecutive_losses(trades) == 3


def test_no_single_trade_dominates_profit_flags_a_dominant_winner():
    trades = [_trade(10.0), _trade(0.1), _trade(0.1)]
    result = metrics.compute_trade_metrics(trades)
    assert result["no_single_trade_dominates_profit"] is False

    balanced = [_trade(1.0), _trade(1.1), _trade(0.9)]
    result2 = metrics.compute_trade_metrics(balanced)
    assert result2["no_single_trade_dominates_profit"] is True


def test_stop_overrun_only_considers_stop_exits():
    trades = [_trade(1.0, exit_reason="TP1"), _trade(-1.0, exit_reason="STOP")]
    trades[1].stop_price_initial = 100.0
    trades[1].exit_price = 100.5  # 50bps overrun past the theoretical stop
    result = metrics.compute_trade_metrics(trades)
    assert result["stop_overrun_bps"] == pytest.approx(50.0, abs=0.1)


def test_compute_continuation_probabilities():
    events = [_event(max_adverse=0.05), _event(max_adverse=0.25), _event(max_adverse=0.55), _event(max_adverse=1.20)]
    probs = metrics.compute_continuation_probabilities(events)
    assert probs["prob_continuation_20pct"] == pytest.approx(0.75)  # 3/4 reached +20%
    assert probs["prob_continuation_50pct"] == pytest.approx(0.50)  # 2/4 reached +50%
    assert probs["prob_continuation_100pct"] == pytest.approx(0.25)  # 1/4 reached +100%


def test_compute_event_metrics_trigger_rate_and_class_breakdown():
    t = _trade(1.0)
    events = [_event(pump_class="SHORT_SQUEEZE", trade=t), _event(pump_class="SHORT_SQUEEZE", trade=None), _event(pump_class="LEVERAGED_LONG", trade=None)]
    result = metrics.compute_event_metrics(events)
    assert result["n_pump_events"] == 3
    assert result["n_events_triggered_trade"] == 1
    assert result["trigger_rate"] == pytest.approx(1 / 3)
    assert result["events_by_class"] == {"SHORT_SQUEEZE": 2, "LEVERAGED_LONG": 1}


def test_breakdown_by_pump_class_and_entry_mode():
    trades = [
        _trade(1.0, pump_class="SHORT_SQUEEZE", entry_mode="standard"),
        _trade(-1.0, pump_class="SHORT_SQUEEZE", entry_mode="conservative"),
        _trade(2.0, pump_class="LEVERAGED_LONG", entry_mode="standard"),
    ]
    by_class = metrics.breakdown_by_pump_class(trades)
    assert set(by_class.keys()) == {"SHORT_SQUEEZE", "LEVERAGED_LONG"}
    assert by_class["SHORT_SQUEEZE"]["n_trades"] == 2
    by_mode = metrics.breakdown_by_entry_mode(trades)
    assert by_mode["standard"]["n_trades"] == 2
    assert by_mode["conservative"]["n_trades"] == 1


def test_breakdown_by_exit_reason_strips_time_stop_detail():
    trades = [_trade(1.0, exit_reason="TIME_STOP (no progress: 0.1R < 0.25R after 60min)"), _trade(-1.0, exit_reason="STOP")]
    by_reason = metrics.breakdown_by_exit_reason(trades)
    assert set(by_reason.keys()) == {"TIME_STOP", "STOP"}


def test_compute_full_report_shape():
    trades = [_trade(1.0)]
    events = [_event(trade=trades[0])]
    report = metrics.compute_full_report(trades, events)
    assert set(report.keys()) == {"overall", "events", "by_pump_class", "by_entry_mode", "by_exit_reason"}
