from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import negative_analysis
from pump_exhaustion_reversal.tests.test_metrics import _event, _trade


def _fp_row(event_id, max_upside=0.0, dd_24h=0.0, dd_1h=0.0, dd_4h=0.0, dd_8h=0.0):
    return {
        "event_id": event_id, "maximum_additional_upside": max_upside,
        "drawdown_from_eventual_peak_1h": dd_1h, "drawdown_from_eventual_peak_4h": dd_4h,
        "drawdown_from_eventual_peak_8h": dd_8h, "drawdown_from_eventual_peak_24h": dd_24h,
    }


def test_find_false_negatives_requires_no_trade_and_big_drawdown():
    events = [
        _event(event_id="e1", pump_class="SPOT_ORGANIC", exhaustion_score_peak=60.0, trade=None,
               invalidation_reason="reached event horizon without a valid trigger",
               exhaustion_component_first_true_bar={"structure_break": 1}),
        _event(event_id="e2", pump_class="SPOT_ORGANIC", exhaustion_score_peak=20.0, trade=_trade(final_r=1.0), invalidation_reason=None),  # had a trade -> excluded
        _event(event_id="e3", pump_class="SPOT_ORGANIC", exhaustion_score_peak=10.0, trade=None, invalidation_reason=None),  # small drawdown -> excluded
    ]
    forward_paths = pd.DataFrame([
        _fp_row("e1", dd_24h=0.25),  # big reversal, no trade -> false negative
        _fp_row("e2", dd_24h=0.30),  # big reversal, but had a trade -> not a false negative
        _fp_row("e3", dd_24h=0.02),  # small reversal -> not a false negative
    ])
    fn = negative_analysis.find_false_negatives(events, forward_paths, drawdown_threshold=0.20, within_hours=24)
    assert list(fn["event_id"]) == ["e1"]
    assert "structure_break" in fn.iloc[0]["features_fired"]
    assert "cvd_bearish_divergence" in fn.iloc[0]["features_not_fired"]


def test_find_false_negatives_missing_column_raises():
    events = [_event(event_id="e1")]
    forward_paths = pd.DataFrame([{"event_id": "e1", "maximum_additional_upside": 0.0}])
    with pytest.raises(ValueError):
        negative_analysis.find_false_negatives(events, forward_paths, within_hours=24)


def test_find_true_negatives_requires_no_trade_and_big_continuation():
    events = [
        _event(event_id="e1", trade=None, invalidation_reason="price made a new pump high beyond the pre-break swing high - the pump resumed, it did not exhaust"),
        _event(event_id="e2", trade=_trade(final_r=-1.0), invalidation_reason=None),  # had a trade -> excluded
    ]
    forward_paths = pd.DataFrame([
        _fp_row("e1", max_upside=0.55),
        _fp_row("e2", max_upside=0.60),
    ])
    tn = negative_analysis.find_true_negatives(events, forward_paths, continuation_threshold=0.20)
    assert list(tn["event_id"]) == ["e1"]


def test_summarize_false_negatives_empty():
    assert negative_analysis.summarize_false_negatives(pd.DataFrame()) == {"n": 0}


def test_summarize_false_negatives_categorizes_reasons_and_missing_features():
    fn_df = pd.DataFrame(
        [
            {
                "pump_class": "SPOT_ORGANIC", "invalidation_reason": "reached event horizon without a valid trigger",
                "features_not_fired": ["cvd_bearish_divergence", "oi_rollover"], "exhaustion_score_peak": 60.0,
            },
            {
                "pump_class": "LEVERAGED_LONG", "invalidation_reason": None,
                "features_not_fired": ["cvd_bearish_divergence"], "exhaustion_score_peak": 40.0,
            },
        ]
    )
    summary = negative_analysis.summarize_false_negatives(fn_df)
    assert summary["n"] == 2
    assert summary["by_pump_class"] == {"SPOT_ORGANIC": 1, "LEVERAGED_LONG": 1}
    assert summary["by_invalidation_reason_category"]["timeout"] == 1
    assert summary["by_invalidation_reason_category"]["still open / no invalidation recorded"] == 1
    assert summary["most_commonly_missing_features"]["cvd_bearish_divergence"] == 2
    assert summary["median_exhaustion_score_peak"] == pytest.approx(50.0)


def test_summarize_true_negatives_empty():
    assert negative_analysis.summarize_true_negatives(pd.DataFrame()) == {"n": 0}


def test_render_negative_analysis_reports_smoke():
    fn_df = pd.DataFrame(
        [{"symbol": "AAAUSDT", "exchange": "binance", "pump_class": "SPOT_ORGANIC", "exhaustion_score_peak": 60.0,
          "drawdown_from_eventual_peak": 0.25, "features_not_fired": ["cvd_bearish_divergence"]}]
    )
    tn_df = pd.DataFrame(
        [{"symbol": "BBBUSDT", "exchange": "bybit", "maximum_additional_upside": 0.55, "invalidation_reason": "new pump high..."}]
    )
    fn_summary = negative_analysis.summarize_false_negatives(
        pd.DataFrame([{"pump_class": "SPOT_ORGANIC", "invalidation_reason": None, "features_not_fired": ["cvd_bearish_divergence"], "exhaustion_score_peak": 60.0}])
    )
    tn_summary = negative_analysis.summarize_true_negatives(
        pd.DataFrame([{"invalidation_reason": "new pump high...", "maximum_additional_upside": 0.55}])
    )
    fn_report, tn_report = negative_analysis.render_negative_analysis_reports(fn_df, fn_summary, tn_df, tn_summary)
    assert "# False Negative Analysis" in fn_report
    assert "AAAUSDT" in fn_report
    assert "# True Negative Analysis" in tn_report
    assert "BBBUSDT" in tn_report
