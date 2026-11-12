from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import cross_exchange
from pump_exhaustion_reversal.tests.test_metrics import _event


def _linked_row(
    binance_present=True,
    bybit_present=True,
    lead_exchange=None,
    delta=0.0,
    binance_class="SHORT_SQUEEZE",
    bybit_class="SHORT_SQUEEZE",
    symbol="BTCUSDT",
    **kw,
):
    """Small builder for hand-constructed linked_df rows, mirroring
    test_metrics._event()'s style - used by the confusion_matrix/
    lead_lag_summary tests below so they exercise those functions directly
    against known DataFrames rather than indirectly through
    link_cross_exchange_events."""
    both = binance_present and bybit_present
    row = dict(
        underlying_event_id=f"underlying_{symbol}_0",
        symbol=symbol,
        binance_event_id="b1" if binance_present else None,
        bybit_event_id="y1" if bybit_present else None,
        binance_present=binance_present,
        bybit_present=bybit_present,
        lead_exchange=lead_exchange if both else None,
        detection_time_delta_minutes=delta if both else np.nan,
        binance_return=0.0 if binance_present else np.nan,
        bybit_return=0.0 if bybit_present else np.nan,
        binance_pump_class=binance_class if binance_present else None,
        bybit_pump_class=bybit_class if bybit_present else None,
    )
    row.update(kw)
    return row


def _linked_df(rows):
    return pd.DataFrame(rows, columns=cross_exchange.LINKED_COLUMNS)


def test_link_close_match_within_window_links_correctly():
    be = _event(exchange="binance", symbol="BTCUSDT", event_id="b1", pump_detected_ts=1_000_000,
                subsequent_return_1h=0.05, pump_class="SHORT_SQUEEZE")
    ye = _event(exchange="bybit", symbol="BTCUSDT", event_id="y1", pump_detected_ts=1_000_000 + 10 * 60_000,
                subsequent_return_1h=-0.02, pump_class="LEVERAGED_LONG")
    linked = cross_exchange.link_cross_exchange_events([be], [ye], window_minutes=60)

    assert len(linked) == 1
    row = linked.iloc[0]
    assert row["binance_present"] and row["bybit_present"]
    assert row["binance_event_id"] == "b1"
    assert row["bybit_event_id"] == "y1"
    assert row["lead_exchange"] == "binance"  # binance's ts is earlier
    assert row["detection_time_delta_minutes"] == pytest.approx(10.0)  # (bybit - binance) / 60000, positive
    assert row["binance_return"] == pytest.approx(0.05)
    assert row["bybit_return"] == pytest.approx(-0.02)
    assert row["binance_pump_class"] == "SHORT_SQUEEZE"
    assert row["bybit_pump_class"] == "LEVERAGED_LONG"
    assert row["underlying_event_id"] == f"underlying_BTCUSDT_{1_000_000}"


def test_link_pair_outside_window_does_not_link():
    be = _event(exchange="binance", symbol="BTCUSDT", event_id="b1", pump_detected_ts=0)
    ye = _event(exchange="bybit", symbol="BTCUSDT", event_id="y1", pump_detected_ts=61 * 60_000)  # 61 min > 60 min window
    linked = cross_exchange.link_cross_exchange_events([be], [ye], window_minutes=60)

    assert len(linked) == 2
    solo_binance = linked[linked["binance_event_id"] == "b1"].iloc[0]
    solo_bybit = linked[linked["bybit_event_id"] == "y1"].iloc[0]
    assert solo_binance["binance_present"] and not solo_binance["bybit_present"]
    assert solo_bybit["bybit_present"] and not solo_bybit["binance_present"]
    assert not (linked["binance_present"] & linked["bybit_present"]).any()


def test_link_different_symbols_never_link_even_if_timestamps_are_close():
    be = _event(exchange="binance", symbol="BTCUSDT", event_id="b1", pump_detected_ts=0)
    ye = _event(exchange="bybit", symbol="ETHUSDT", event_id="y1", pump_detected_ts=0)
    linked = cross_exchange.link_cross_exchange_events([be], [ye], window_minutes=60)

    assert len(linked) == 2
    assert not (linked["binance_present"] & linked["bybit_present"]).any()
    assert set(linked["symbol"]) == {"BTCUSDT", "ETHUSDT"}


def test_link_greedy_nearest_match_prefers_closer_candidate():
    be = _event(exchange="binance", symbol="BTCUSDT", event_id="b1", pump_detected_ts=1_000_000)
    ye_close = _event(exchange="bybit", symbol="BTCUSDT", event_id="y_close", pump_detected_ts=1_000_000 + 5 * 60_000)
    ye_far = _event(exchange="bybit", symbol="BTCUSDT", event_id="y_far", pump_detected_ts=1_000_000 + 30 * 60_000)
    # pass the farther candidate first to prove the algorithm sorts by delta globally,
    # not "first candidate encountered in list order wins"
    linked = cross_exchange.link_cross_exchange_events([be], [ye_far, ye_close], window_minutes=60)

    matched = linked[linked["binance_present"] & linked["bybit_present"]]
    assert len(matched) == 1
    assert matched.iloc[0]["bybit_event_id"] == "y_close"
    assert matched.iloc[0]["detection_time_delta_minutes"] == pytest.approx(5.0)

    solo = linked[linked["bybit_event_id"] == "y_far"]
    assert len(solo) == 1
    assert not solo.iloc[0]["binance_present"]
    assert solo.iloc[0]["bybit_present"]


def test_confusion_matrix_known_cell_counts():
    rows = [
        _linked_row(binance_class="SHORT_SQUEEZE", bybit_class="SHORT_SQUEEZE"),
        _linked_row(binance_class="SHORT_SQUEEZE", bybit_class="SHORT_SQUEEZE"),
        _linked_row(binance_class="SHORT_SQUEEZE", bybit_class="LEVERAGED_LONG"),
        _linked_row(binance_class="LEVERAGED_LONG", bybit_class="LEVERAGED_LONG"),
        _linked_row(binance_present=True, bybit_present=False, binance_class="SHORT_SQUEEZE"),  # solo - excluded
    ]
    cm = cross_exchange.confusion_matrix(_linked_df(rows))

    assert cm.loc["SHORT_SQUEEZE", "SHORT_SQUEEZE"] == 2
    assert cm.loc["SHORT_SQUEEZE", "LEVERAGED_LONG"] == 1
    assert cm.loc["LEVERAGED_LONG", "LEVERAGED_LONG"] == 1
    assert cm.values.sum() == 4  # the solo row must not be counted


def test_confusion_matrix_empty_when_zero_both_present_rows():
    rows = [
        _linked_row(binance_present=True, bybit_present=False),
        _linked_row(binance_present=False, bybit_present=True),
    ]
    cm = cross_exchange.confusion_matrix(_linked_df(rows))
    assert isinstance(cm, pd.DataFrame)
    assert cm.empty


def test_confusion_matrix_empty_on_empty_linked_df():
    cm = cross_exchange.confusion_matrix(_linked_df([]))
    assert isinstance(cm, pd.DataFrame)
    assert cm.empty


def test_lead_lag_summary_known_counts_and_percentages():
    rows = [
        _linked_row(lead_exchange="binance", delta=10.0),
        _linked_row(lead_exchange="binance", delta=20.0),
        _linked_row(lead_exchange="bybit", delta=-5.0),
        _linked_row(lead_exchange=None, delta=0.0),
        _linked_row(binance_present=True, bybit_present=False),  # solo - excluded from n_both_present
    ]
    summary = cross_exchange.lead_lag_summary(_linked_df(rows))

    assert summary["n_both_present"] == 4
    assert summary["n_binance_led"] == 2
    assert summary["n_bybit_led"] == 1
    assert summary["n_simultaneous"] == 1
    assert summary["pct_binance_led"] == pytest.approx(0.5)
    assert summary["pct_bybit_led"] == pytest.approx(0.25)
    # abs deltas [10, 20, 5, 0] -> sorted [0, 5, 10, 20]; pandas linear-interpolation quantiles:
    # median (pos 1.5) = 5 + 0.5*(10-5) = 7.5; p25 (pos 0.75) = 0 + 0.75*(5-0) = 3.75; p75 (pos 2.25) = 10 + 0.25*(20-10) = 12.5
    assert summary["median_abs_delta_minutes"] == pytest.approx(7.5)
    assert summary["p25_abs_delta_minutes"] == pytest.approx(3.75)
    assert summary["p75_abs_delta_minutes"] == pytest.approx(12.5)


def test_lead_lag_summary_zero_when_zero_both_present_rows():
    rows = [
        _linked_row(binance_present=True, bybit_present=False),
        _linked_row(binance_present=False, bybit_present=True),
    ]
    summary = cross_exchange.lead_lag_summary(_linked_df(rows))

    assert summary["n_both_present"] == 0
    assert summary["n_binance_led"] == 0
    assert summary["n_bybit_led"] == 0
    assert summary["n_simultaneous"] == 0
    assert np.isnan(summary["pct_binance_led"])
    assert np.isnan(summary["pct_bybit_led"])
    assert np.isnan(summary["median_abs_delta_minutes"])
    assert np.isnan(summary["p25_abs_delta_minutes"])
    assert np.isnan(summary["p75_abs_delta_minutes"])


def test_lead_lag_summary_zero_on_empty_linked_df():
    summary = cross_exchange.lead_lag_summary(_linked_df([]))
    assert summary["n_both_present"] == 0
    assert np.isnan(summary["pct_binance_led"])
    assert np.isnan(summary["median_abs_delta_minutes"])


def test_render_cross_exchange_report_smoke():
    be1 = _event(exchange="binance", symbol="BTCUSDT", event_id="b1", pump_detected_ts=0, pump_class="SHORT_SQUEEZE")
    ye1 = _event(exchange="bybit", symbol="BTCUSDT", event_id="y1", pump_detected_ts=5 * 60_000, pump_class="SHORT_SQUEEZE")
    be2 = _event(exchange="binance", symbol="ETHUSDT", event_id="b2", pump_detected_ts=0, pump_class="LEVERAGED_LONG")
    linked = cross_exchange.link_cross_exchange_events([be1, be2], [ye1], window_minutes=60)
    cm = cross_exchange.confusion_matrix(linked)
    lead_lag = cross_exchange.lead_lag_summary(linked)

    report = cross_exchange.render_cross_exchange_report(linked, cm, lead_lag)
    assert "# Cross-Exchange Event Linkage Report" in report
    assert "Total underlying events: 2" in report
    assert "Both exchanges present" in report
    assert "SHORT_SQUEEZE" in report


def test_render_cross_exchange_report_handles_empty_confusion_matrix():
    be1 = _event(exchange="binance", symbol="BTCUSDT", event_id="b1", pump_detected_ts=0)
    linked = cross_exchange.link_cross_exchange_events([be1], [], window_minutes=60)
    cm = cross_exchange.confusion_matrix(linked)
    lead_lag = cross_exchange.lead_lag_summary(linked)

    report = cross_exchange.render_cross_exchange_report(linked, cm, lead_lag)
    assert "Not enough overlapping" in report
    assert "No both-present events" in report


def test_render_cross_exchange_report_handles_zero_events():
    linked = cross_exchange.link_cross_exchange_events([], [])
    cm = cross_exchange.confusion_matrix(linked)
    lead_lag = cross_exchange.lead_lag_summary(linked)

    report = cross_exchange.render_cross_exchange_report(linked, cm, lead_lag)
    assert "Total underlying events: 0" in report
    assert "nothing to link" in report.lower()
