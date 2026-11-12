"""Hand-computed-fixture tests for research/forward_study.py (spec sections
14/15/16). Every OHLCV fixture below is a fully deterministic, formula-driven
price path (not random) so expected values are derived analytically, not just
sanity-checked for finiteness - matching this repo's precision-testing
convention (see test_post_break_invalidation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import forward_study
from pump_exhaustion_reversal.tests.test_metrics import _event

BAR_MS = 300_000  # 5m


def _flat_prefix_rows(n_pre: int, price: float = 100.0) -> list[dict]:
    return [
        {"timestamp": i * BAR_MS, "open": price, "high": price, "low": price, "close": price}
        for i in range(n_pre)
    ]


def _clean_forward_ohlcv() -> pd.DataFrame:
    """detect bar at idx=50 (close=100.0, ts=15_000_000). Rising phase idx
    51..150 (o=1..100): high=100+o, low=100+o-1, close=100+o-0.5 - peaks
    exactly at idx150 (o=100) with high=200. Falling phase idx151..350
    (d=1..200): high=200-0.5d, low=199-0.5d, close=199.5-0.5d - strictly
    below the idx150 peak throughout, and low decreases by a clean 0.5/bar so
    every percent-off-peak threshold below is crossed on an exact bar, no
    rounding ambiguity. 351 rows total (idx 0..350)."""
    rows = _flat_prefix_rows(51)  # idx 0..50
    for o in range(1, 101):  # idx 51..150
        idx = 50 + o
        high, low, close = 100.0 + o, 100.0 + o - 1.0, 100.0 + o - 0.5
        rows.append({"timestamp": idx * BAR_MS, "open": close, "high": high, "low": low, "close": close})
    for d in range(1, 201):  # idx 151..350
        idx = 150 + d
        high, low, close = 200.0 - 0.5 * d, 199.0 - 0.5 * d, 199.5 - 0.5 * d
        rows.append({"timestamp": idx * BAR_MS, "open": close, "high": high, "low": low, "close": close})
    df = pd.DataFrame(rows)
    assert len(df) == 351
    return df


def _censored_forward_ohlcv() -> pd.DataFrame:
    """detect bar at idx=50 (close=100.0, ts=15_000_000), then only 4 more
    bars (idx 51..54) before the data just stops - the eventual "peak" within
    this short series is forced to be the very last bar (monotonically
    rising highs/lows), so there is zero room left to observe any
    drawdown-from-peak or time-to-drop-X% behaviour. 55 rows total."""
    rows = _flat_prefix_rows(51)  # idx 0..50
    for o in range(1, 5):  # idx 51..54
        idx = 50 + o
        high, low, close = 100.0 + o, 100.0 + o - 1.0, 100.0 + o - 0.5
        rows.append({"timestamp": idx * BAR_MS, "open": close, "high": high, "low": low, "close": close})
    df = pd.DataFrame(rows)
    assert len(df) == 55
    return df


DETECT_TS = 50 * BAR_MS  # 15_000_000


# ---------------------------------------------------------------------------
# compute_forward_path - clean case (plenty of forward data)
# ---------------------------------------------------------------------------

def test_compute_forward_path_clean_case_exact_values():
    ev = _event(event_id="eA", exchange="bybit", symbol="AAAUSDT", pump_class="SHORT_SQUEEZE", pump_detected_ts=DETECT_TS)
    ohlcv = _clean_forward_ohlcv()
    result = forward_study.compute_forward_path(ev, ohlcv, bar_interval_ms=BAR_MS, max_forward_bars=300)

    # passthrough
    assert result["event_id"] == "eA"
    assert result["symbol"] == "AAAUSDT"
    assert result["exchange"] == "bybit"
    assert result["pump_class"] == "SHORT_SQUEEZE"

    # exactly max_forward_bars (300) of forward data was available -> NOT censored
    assert result["right_censored"] is False

    # forward_return_Xm: close-to-close from detect_close=100.0
    assert result["forward_return_15m"] == pytest.approx(0.025)
    assert result["forward_return_30m"] == pytest.approx(0.055)
    assert result["forward_return_1h"] == pytest.approx(0.115)
    assert result["forward_return_2h"] == pytest.approx(0.235)
    assert result["forward_return_4h"] == pytest.approx(0.475)
    assert result["forward_return_8h"] == pytest.approx(0.955)
    assert result["forward_return_12h"] == pytest.approx(0.775)
    assert result["forward_return_24h"] == pytest.approx(0.055)

    # extremes over the whole bounded forward window (idx 51..350)
    assert result["maximum_additional_upside"] == pytest.approx(1.0)  # (200-100)/100, peak high at idx150
    assert result["maximum_downside"] == pytest.approx(-0.01)  # (99-100)/100, lowest low at idx350

    # eventual peak = idx150 (high=200), 100 bars / 500 minutes after detection
    assert result["time_to_eventual_peak_bars"] == 100
    assert result["time_to_eventual_peak_minutes"] == pytest.approx(500.0)
    assert result["eventual_peak_price"] == pytest.approx(200.0)
    assert result["eventual_peak_return_from_detection"] == pytest.approx(1.0)

    # drawdown from the idx150 peak, over growing windows into the falling phase
    assert result["drawdown_from_eventual_peak_1h"] == pytest.approx(0.035)   # (200-193)/200
    assert result["drawdown_from_eventual_peak_4h"] == pytest.approx(0.125)   # (200-175)/200
    assert result["drawdown_from_eventual_peak_8h"] == pytest.approx(0.245)   # (200-151)/200
    assert result["drawdown_from_eventual_peak_24h"] == pytest.approx(0.505)  # (200-99)/200

    # bars from the peak to the first close <= peak*(1-X%); low drops by an
    # exact 0.5/bar so every threshold lands on a whole bar, no rounding
    assert result["time_peak_to_minus5pct_bars"] == 18
    assert result["time_peak_to_minus5pct_minutes"] == pytest.approx(90.0)
    assert result["time_peak_to_minus10pct_bars"] == 38
    assert result["time_peak_to_minus10pct_minutes"] == pytest.approx(190.0)
    assert result["time_peak_to_minus20pct_bars"] == 78
    assert result["time_peak_to_minus20pct_minutes"] == pytest.approx(390.0)
    assert result["time_peak_to_minus30pct_bars"] == 118
    assert result["time_peak_to_minus30pct_minutes"] == pytest.approx(590.0)


# ---------------------------------------------------------------------------
# compute_forward_path - right-censored case (event near the end of the data)
# ---------------------------------------------------------------------------

def test_compute_forward_path_right_censored_case():
    ev = _event(event_id="eB", exchange="bybit", symbol="BBBUSDT", pump_class="LEVERAGED_LONG", pump_detected_ts=DETECT_TS)
    ohlcv = _censored_forward_ohlcv()
    # default max_forward_bars (2000) is nowhere near satisfiable - only 4 bars exist after detection
    result = forward_study.compute_forward_path(ev, ohlcv, bar_interval_ms=BAR_MS)

    assert result["right_censored"] is True

    # 15m (3 bars ahead) fits inside the 4 available forward bars -> real, not NaN/0
    assert result["forward_return_15m"] == pytest.approx(0.025)
    # everything needing more than 4 forward bars must be NaN, never silently 0
    for label in ("30m", "1h", "2h", "4h", "8h", "12h", "24h"):
        assert np.isnan(result[f"forward_return_{label}"]), label

    # the 4 available forward bars are still enough to answer "what's the peak so far"
    assert result["maximum_additional_upside"] == pytest.approx(0.04)
    assert result["maximum_downside"] == pytest.approx(0.0)
    assert result["time_to_eventual_peak_bars"] == 4
    assert result["time_to_eventual_peak_minutes"] == pytest.approx(20.0)
    assert result["eventual_peak_price"] == pytest.approx(104.0)
    assert result["eventual_peak_return_from_detection"] == pytest.approx(0.04)

    # but the "peak" IS the last available bar, so there is zero room left to
    # observe any drawdown-from-peak or time-to-drop-X% - all NaN, not 0
    for label in ("1h", "4h", "8h", "24h"):
        assert np.isnan(result[f"drawdown_from_eventual_peak_{label}"]), label
    for pct in (5, 10, 20, 30):
        assert np.isnan(result[f"time_peak_to_minus{pct}pct_bars"]), pct
        assert np.isnan(result[f"time_peak_to_minus{pct}pct_minutes"]), pct


def test_compute_forward_path_no_anchor_bar_available_is_graceful():
    """pump_detected_ts is after every bar we have on disk - there isn't even
    a detection-bar close to anchor off of. Must degrade to an all-NaN,
    right_censored=True row rather than raising/crashing or inventing 0s."""
    ev = _event(event_id="eC", exchange="bybit", symbol="CCCUSDT", pump_class="SPOT_ORGANIC", pump_detected_ts=10_000_000_000)
    ohlcv = _clean_forward_ohlcv()  # last timestamp is 350*300_000 = 105_000_000, far short of pump_detected_ts
    result = forward_study.compute_forward_path(ev, ohlcv, bar_interval_ms=BAR_MS)

    assert result["event_id"] == "eC"
    assert result["right_censored"] is True
    assert np.isnan(result["forward_return_15m"])
    assert np.isnan(result["maximum_additional_upside"])
    assert np.isnan(result["eventual_peak_price"])
    assert np.isnan(result["time_peak_to_minus5pct_bars"])


def test_compute_forward_path_empty_ohlcv_is_graceful():
    ev = _event(event_id="eD", exchange="bybit", symbol="DDDUSDT", pump_detected_ts=DETECT_TS)
    result = forward_study.compute_forward_path(ev, pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"]), bar_interval_ms=BAR_MS)
    assert result["right_censored"] is True
    assert np.isnan(result["forward_return_24h"])
    assert np.isnan(result["eventual_peak_price"])


# ---------------------------------------------------------------------------
# compute_forward_paths_for_events - batch + skip-missing-symbol
# ---------------------------------------------------------------------------

def _three_events_and_lookup():
    events = [
        _event(event_id="eA", exchange="bybit", symbol="AAAUSDT", pump_class="SHORT_SQUEEZE", pump_detected_ts=DETECT_TS),
        _event(event_id="eB", exchange="bybit", symbol="BBBUSDT", pump_class="LEVERAGED_LONG", pump_detected_ts=DETECT_TS),
        _event(event_id="eC", exchange="bybit", symbol="CCCUSDT", pump_class="SHORT_SQUEEZE", pump_detected_ts=DETECT_TS),
        _event(event_id="eD", exchange="bybit", symbol="ZZZUSDT", pump_class="NEWS_FUNDAMENTAL", pump_detected_ts=DETECT_TS),
    ]
    ohlcv_by_symbol = {
        ("bybit", "AAAUSDT"): _clean_forward_ohlcv(),
        ("bybit", "BBBUSDT"): _censored_forward_ohlcv(),
        ("bybit", "CCCUSDT"): _clean_forward_ohlcv(),
        # ZZZUSDT deliberately absent -> event eD must be skipped, not crash
    }
    return events, ohlcv_by_symbol


def test_compute_forward_paths_for_events_skips_missing_symbol():
    events, ohlcv_by_symbol = _three_events_and_lookup()
    df = forward_study.compute_forward_paths_for_events(events, ohlcv_by_symbol, bar_interval_ms=BAR_MS, max_forward_bars=300)
    assert len(df) == 3
    assert set(df["symbol"]) == {"AAAUSDT", "BBBUSDT", "CCCUSDT"}
    assert "ZZZUSDT" not in set(df["symbol"])


def test_compute_forward_paths_for_events_empty_result_has_correct_columns():
    ev = _event(event_id="eOnly", exchange="bybit", symbol="MISSINGUSDT", pump_detected_ts=DETECT_TS)
    df = forward_study.compute_forward_paths_for_events([ev], {}, bar_interval_ms=BAR_MS)
    assert len(df) == 0
    assert list(df.columns) == forward_study.FORWARD_PATH_COLUMNS


# ---------------------------------------------------------------------------
# breakdown_by_pump_class
# ---------------------------------------------------------------------------

def test_breakdown_by_pump_class_stats_and_probabilities():
    events, ohlcv_by_symbol = _three_events_and_lookup()
    forward_paths_df = forward_study.compute_forward_paths_for_events(events, ohlcv_by_symbol, bar_interval_ms=BAR_MS, max_forward_bars=300)
    by_class = forward_study.breakdown_by_pump_class(forward_paths_df)

    assert set(by_class["pump_class"]) == {"SHORT_SQUEEZE", "LEVERAGED_LONG"}

    ss = by_class[by_class["pump_class"] == "SHORT_SQUEEZE"].iloc[0]
    assert ss["n"] == 2
    # both SHORT_SQUEEZE rows are identical clean fixtures -> median==p75==p90
    assert ss["median_additional_upside"] == pytest.approx(1.0)
    assert ss["p90_additional_upside"] == pytest.approx(1.0)
    assert ss["median_drawdown_from_peak_24h"] == pytest.approx(0.505)
    assert ss["median_time_to_peak_minutes"] == pytest.approx(500.0)
    # minus5pct fires at minute 90 for both rows: not within 1h(60), within 4h(240)
    assert ss["prob_minus5pct_1h"] == pytest.approx(0.0)
    assert ss["prob_minus5pct_4h"] == pytest.approx(1.0)

    ll = by_class[by_class["pump_class"] == "LEVERAGED_LONG"].iloc[0]
    assert ll["n"] == 1
    assert ll["median_time_to_peak_minutes"] == pytest.approx(20.0)
    # the censored fixture never observes any peak-to-drop within its bounded
    # window (all NaN) -> every probability column must read 0.0, not NaN,
    # per the "never observed -> No" convention
    assert ll["prob_minus5pct_24h"] == pytest.approx(0.0)
    assert ll["prob_minus30pct_24h"] == pytest.approx(0.0)


def test_breakdown_by_pump_class_empty_input():
    df = forward_study.breakdown_by_pump_class(pd.DataFrame(columns=forward_study.FORWARD_PATH_COLUMNS))
    assert len(df) == 0
    assert list(df.columns) == forward_study.BREAKDOWN_COLUMNS


# ---------------------------------------------------------------------------
# render_forward_event_study_report
# ---------------------------------------------------------------------------

def test_render_forward_event_study_report_smoke():
    events, ohlcv_by_symbol = _three_events_and_lookup()
    forward_paths_df = forward_study.compute_forward_paths_for_events(events, ohlcv_by_symbol, bar_interval_ms=BAR_MS, max_forward_bars=300)
    by_class = forward_study.breakdown_by_pump_class(forward_paths_df)

    report = forward_study.render_forward_event_study_report(forward_paths_df, by_class)

    assert "# Forward Event Study Report" in report
    assert "Events analyzed: 3" in report
    # exactly 1 of the 3 (BBBUSDT) is right-censored -> 1/3 = 33.3%
    assert "33.3%" in report
    assert "SHORT_SQUEEZE" in report
    assert "LEVERAGED_LONG" in report


def test_render_forward_event_study_report_empty_input():
    empty_paths = pd.DataFrame(columns=forward_study.FORWARD_PATH_COLUMNS)
    empty_by_class = forward_study.breakdown_by_pump_class(empty_paths)
    report = forward_study.render_forward_event_study_report(empty_paths, empty_by_class)
    assert "Events analyzed: 0" in report
    assert "No events." in report
