from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.config import AppConfig
from pump_exhaustion_reversal.research import underlying_classification as uc
from pump_exhaustion_reversal.tests.test_metrics import _event


def test_classify_short_squeeze_state_active_when_retried_on_new_high():
    cfg = AppConfig()
    event = _event(pump_class="SHORT_SQUEEZE", attempt_count=2, attempt_invalidation_reason_codes=["new_pump_high"], oi_drop_pct_from_peak_at_peak=20.0)
    assert uc.classify_short_squeeze_state(event, cfg) == "SHORT_SQUEEZE_ACTIVE"


def test_classify_short_squeeze_state_active_when_oi_has_not_dropped():
    cfg = AppConfig()
    event = _event(pump_class="SHORT_SQUEEZE", oi_drop_pct_from_peak_at_peak=1.0)  # below cfg.oi.min_oi_drop_pct_after_peak (5)
    assert uc.classify_short_squeeze_state(event, cfg) == "SHORT_SQUEEZE_ACTIVE"


def test_classify_short_squeeze_state_depleted_when_oi_dropped_and_never_retried():
    cfg = AppConfig()
    event = _event(pump_class="SHORT_SQUEEZE", attempt_count=1, attempt_invalidation_reason_codes=[], oi_drop_pct_from_peak_at_peak=12.0)
    assert uc.classify_short_squeeze_state(event, cfg) == "SHORT_SQUEEZE_DEPLETED"


def test_classify_short_squeeze_state_active_when_oi_drop_unknown():
    cfg = AppConfig()
    event = _event(pump_class="SHORT_SQUEEZE", oi_drop_pct_from_peak_at_peak=np.nan)
    assert uc.classify_short_squeeze_state(event, cfg) == "SHORT_SQUEEZE_ACTIVE"


def test_positive_classification_wins_over_spot_organic():
    binance_ev = _event(exchange="binance", pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=np.nan, short_liq_zscore=np.nan)
    bybit_ev = _event(exchange="bybit", pump_class="LEVERAGED_LONG", pump_class_confidence="high", oi_zscore=3.0)
    result = uc.classify_underlying_event(binance_ev, bybit_ev)
    assert result["underlying_pump_class"] == "LEVERAGED_LONG"
    assert result["source_exchange"] == "bybit"


def test_both_spot_organic_with_derivatives_signal_stays_spot_organic():
    binance_ev = _event(exchange="binance", pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=1.0)
    bybit_ev = _event(exchange="bybit", pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=np.nan, short_liq_zscore=np.nan)
    result = uc.classify_underlying_event(binance_ev, bybit_ev)
    assert result["underlying_pump_class"] == "SPOT_ORGANIC"


def test_no_derivatives_signal_anywhere_is_insufficient_data_not_spot_organic():
    binance_ev = _event(exchange="binance", pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=np.nan, short_liq_zscore=np.nan)
    result = uc.classify_underlying_event(binance_ev, None)
    assert result["underlying_pump_class"] == "UNKNOWN_INSUFFICIENT_DATA"


def test_no_events_at_all_is_unknown():
    result = uc.classify_underlying_event(None, None)
    assert result["underlying_pump_class"] == "UNKNOWN"
    assert result["source_exchange"] is None


def test_two_positive_classifications_conflict_resolved_by_confidence():
    binance_ev = _event(exchange="binance", pump_class="LIQUIDITY_MANIPULATION", pump_class_confidence="medium")
    bybit_ev = _event(exchange="bybit", pump_class="SHORT_SQUEEZE", pump_class_confidence="high")
    result = uc.classify_underlying_event(binance_ev, bybit_ev)
    assert result["underlying_pump_class"] == "SHORT_SQUEEZE"
    assert result["source_exchange"] == "bybit"


def test_classify_underlying_events_joins_by_linked_df():
    binance_ev = _event(event_id="bin1", exchange="binance", pump_class="LEVERAGED_LONG", pump_class_confidence="high", oi_zscore=2.0)
    bybit_ev = _event(event_id="byb1", exchange="bybit", pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=np.nan, short_liq_zscore=np.nan)
    linked_df = pd.DataFrame([{"underlying_event_id": "u1", "binance_event_id": "bin1", "bybit_event_id": "byb1"}])
    result = uc.classify_underlying_events(linked_df, [binance_ev], [bybit_ev])
    assert result.iloc[0]["underlying_pump_class"] == "LEVERAGED_LONG"
    assert result.iloc[0]["underlying_source_exchange"] == "binance"


def test_underlying_class_distribution_counts_and_shares():
    df = pd.DataFrame({"underlying_pump_class": ["SPOT_ORGANIC", "SPOT_ORGANIC", "LEVERAGED_LONG"]})
    dist = uc.underlying_class_distribution(df)
    spot_row = dist[dist["underlying_pump_class"] == "SPOT_ORGANIC"].iloc[0]
    assert spot_row["n"] == 2
    assert spot_row["share"] == pytest.approx(2 / 3)


def test_underlying_class_distribution_empty():
    dist = uc.underlying_class_distribution(pd.DataFrame())
    assert len(dist) == 0


def test_render_underlying_classification_report_smoke():
    df = pd.DataFrame({"underlying_pump_class": ["SPOT_ORGANIC"], "underlying_source_exchange": ["binance"]})
    dist = uc.underlying_class_distribution(df)
    report = uc.render_underlying_classification_report(df, dist)
    assert "Underlying Cross-Exchange Classification" in report
    assert "SPOT_ORGANIC" in report
    assert "binance" in report
