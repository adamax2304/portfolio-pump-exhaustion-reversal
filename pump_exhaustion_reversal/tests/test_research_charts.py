from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pump_exhaustion_reversal.research import charts
from pump_exhaustion_reversal.tests.test_metrics import _event


def _make_ohlcv(n: int = 40, start_ts: int = 0, step_ms: int = 5 * 60 * 1000, start_price: float = 100.0) -> pd.DataFrame:
    timestamps = [start_ts + i * step_ms for i in range(n)]
    prices = [start_price + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [10.0 + i for i in range(n)],
            "quote_volume": [1000.0 for _ in range(n)],
            "taker_buy_base_volume": [5.0 for _ in range(n)],
        }
    )


# --- select_cohorts ---


def test_select_cohorts_top_reversals_and_continuations_ranked_descending_excluding_nan():
    events = [_event(event_id=f"e{i}") for i in range(5)]
    forward_paths_df = pd.DataFrame(
        {
            "event_id": ["e0", "e1", "e2", "e3", "e4"],
            "drawdown_from_eventual_peak_24h": [0.5, np.nan, 0.9, 0.1, 0.3],
            "maximum_additional_upside": [0.2, 0.8, np.nan, 0.05, 0.4],
        }
    )

    cohorts = charts.select_cohorts(events, forward_paths_df, top_n=3)

    assert cohorts["top_reversals"] == ["e2", "e0", "e4"]  # 0.9, 0.5, 0.3 - e1 (NaN) excluded entirely
    assert cohorts["top_continuations"] == ["e1", "e4", "e0"]  # 0.8, 0.4, 0.2 - e2 (NaN) excluded entirely


def test_select_cohorts_high_exhaustion_and_pump_class_cohorts_and_empty_omitted():
    events = [
        _event(event_id="a", exhaustion_score_peak=70.0, pump_class="LEVERAGED_LONG"),
        _event(event_id="b", exhaustion_score_peak=60.0, pump_class="LIQUIDITY_MANIPULATION"),
        _event(event_id="c", exhaustion_score_peak=80.0, pump_class="LEVERAGED_LONG"),
    ]
    empty_forward_paths_df = pd.DataFrame(columns=["event_id", "drawdown_from_eventual_peak_24h", "maximum_additional_upside"])

    cohorts = charts.select_cohorts(events, empty_forward_paths_df)

    assert cohorts["high_exhaustion_score"] == ["a", "c"]  # b is exactly 60.0, not > 65
    assert cohorts["leveraged_long"] == ["a", "c"]
    assert cohorts["liquidity_manipulation"] == ["b"]
    assert "short_squeeze" not in cohorts  # zero SHORT_SQUEEZE events - key omitted, not an empty list
    assert "top_reversals" not in cohorts  # nothing to rank - key omitted
    assert "top_continuations" not in cohorts


def test_select_cohorts_event_can_appear_in_multiple_cohorts_without_dedup():
    events = [_event(event_id="dual", exhaustion_score_peak=99.0, pump_class="SHORT_SQUEEZE")]
    forward_paths_df = pd.DataFrame(
        {"event_id": ["dual"], "drawdown_from_eventual_peak_24h": [0.9], "maximum_additional_upside": [0.9]}
    )

    cohorts = charts.select_cohorts(events, forward_paths_df)

    assert cohorts["top_reversals"] == ["dual"]
    assert cohorts["top_continuations"] == ["dual"]
    assert cohorts["high_exhaustion_score"] == ["dual"]
    assert cohorts["short_squeeze"] == ["dual"]


# --- plot_event_chart ---


def test_plot_event_chart_creates_nonempty_png_with_annotations(tmp_path):
    long_reason = "new pump high broke structure: a very long invalidation reason that should get truncated"
    event = _event(
        event_id="bybit:BTCUSDT:1", exchange="bybit", symbol="BTCUSDT",
        pump_start_ts=0, pump_detected_ts=20 * 5 * 60 * 1000, pump_peak_ts=25 * 5 * 60 * 1000,
        final_state="INVALIDATED", invalidation_reason=long_reason,
    )
    ohlcv = _make_ohlcv(n=60)
    out_path = tmp_path / "chart.png"

    charts.plot_event_chart(event, ohlcv, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_event_chart_handles_very_short_window_without_raising(tmp_path):
    event = _event(event_id="e1", pump_start_ts=0, pump_detected_ts=0)
    ohlcv = _make_ohlcv(n=3, start_ts=0)
    out_path = tmp_path / "short.png"

    charts.plot_event_chart(event, ohlcv, out_path, window_bars_before=1, window_bars_after=1)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_event_chart_handles_empty_ohlcv_without_raising(tmp_path):
    event = _event(event_id="e1")
    empty_ohlcv = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out_path = tmp_path / "empty.png"

    charts.plot_event_chart(event, empty_ohlcv, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_event_chart_creates_parent_directories(tmp_path):
    event = _event(event_id="e1")
    ohlcv = _make_ohlcv(n=10)
    out_path = tmp_path / "nested" / "dir" / "chart.png"

    charts.plot_event_chart(event, ohlcv, out_path)

    assert out_path.exists()


def test_plot_event_chart_closes_figure_after_saving(tmp_path):
    initial_open_figures = len(plt.get_fignums())
    event = _event(event_id="e1")
    ohlcv = _make_ohlcv(n=20)

    for i in range(3):
        charts.plot_event_chart(event, ohlcv, tmp_path / f"chart_{i}.png")

    assert len(plt.get_fignums()) == initial_open_figures


# --- render_cohort_charts ---


def test_render_cohort_charts_creates_expected_structure_and_paths(tmp_path):
    events = [
        _event(event_id="bybit:BTCUSDT:1", exchange="bybit", symbol="BTCUSDT"),
        _event(event_id="bybit:ETHUSDT:1", exchange="bybit", symbol="ETHUSDT"),
    ]
    ohlcv = _make_ohlcv(n=20)
    ohlcv_by_symbol = {("bybit", "BTCUSDT"): ohlcv, ("bybit", "ETHUSDT"): ohlcv}
    cohorts = {"top_reversals": ["bybit:BTCUSDT:1", "bybit:ETHUSDT:1"]}
    out_dir = tmp_path / "charts"

    result = charts.render_cohort_charts(cohorts, events, ohlcv_by_symbol, out_dir)

    expected_btc = out_dir / "top_reversals" / "bybit_BTCUSDT_1.png"
    expected_eth = out_dir / "top_reversals" / "bybit_ETHUSDT_1.png"
    assert result == {"top_reversals": [str(expected_btc), str(expected_eth)]}
    assert expected_btc.exists() and expected_btc.stat().st_size > 0
    assert expected_eth.exists() and expected_eth.stat().st_size > 0


def test_render_cohort_charts_skips_missing_ohlcv_without_crashing(tmp_path):
    events = [
        _event(event_id="bybit:BTCUSDT:1", exchange="bybit", symbol="BTCUSDT"),
        _event(event_id="bybit:MISSING:1", exchange="bybit", symbol="MISSING"),
    ]
    ohlcv_by_symbol = {("bybit", "BTCUSDT"): _make_ohlcv(n=20)}  # "MISSING" has no cached OHLCV
    cohorts = {"top_reversals": ["bybit:BTCUSDT:1", "bybit:MISSING:1"]}
    out_dir = tmp_path / "charts"

    result = charts.render_cohort_charts(cohorts, events, ohlcv_by_symbol, out_dir)

    assert len(result["top_reversals"]) == 1
    assert "MISSING" not in result["top_reversals"][0]
    assert Path(result["top_reversals"][0]).exists()


def test_render_cohort_charts_fully_skipped_cohort_keeps_key_with_empty_list(tmp_path):
    events = [_event(event_id="bybit:MISSING:1", exchange="bybit", symbol="MISSING")]
    cohorts = {"top_reversals": ["bybit:MISSING:1"]}
    out_dir = tmp_path / "charts"

    result = charts.render_cohort_charts(cohorts, events, {}, out_dir)

    assert result == {"top_reversals": []}
    assert not (out_dir / "top_reversals").exists()  # nothing was ever rendered, so no directory was created
