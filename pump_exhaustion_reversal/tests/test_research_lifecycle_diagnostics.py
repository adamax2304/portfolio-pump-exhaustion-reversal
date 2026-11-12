from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import lifecycle_diagnostics as ld
from pump_exhaustion_reversal.tests.test_metrics import _event, _trade


def test_lifecycle_funnel_counts_each_stage():
    events = [
        _event(climax_local_idx=5, exhaustion_score_peak=80.0, structure_break_bar_local_idx=6,
               exhaustion_component_first_true_bar={"failed_reclaim_lower_high": 7}, attempt_count=2, trade=_trade(final_r=1.0)),
        _event(climax_local_idx=None, exhaustion_score_peak=10.0, structure_break_bar_local_idx=None,
               exhaustion_component_first_true_bar={}, attempt_count=1, trade=None),
    ]
    funnel = ld.lifecycle_funnel(events, armed_threshold=65.0)
    assert funnel["n_pump_detected"] == 2
    assert funnel["n_climax_found"] == 1
    assert funnel["n_exhaustion_forming_or_beyond"] == 1
    assert funnel["n_structure_break"] == 1
    assert funnel["n_failed_reclaim"] == 1
    assert funnel["n_multi_attempt"] == 1
    assert funnel["n_trade"] == 1


def test_lifecycle_funnel_empty():
    assert ld.lifecycle_funnel([], armed_threshold=65.0) == {"n_pump_detected": 0}


def test_render_lifecycle_funnel_smoke():
    funnel = ld.lifecycle_funnel(
        [_event(climax_local_idx=1, exhaustion_score_peak=80.0)], armed_threshold=65.0
    )
    report = ld.render_lifecycle_funnel(funnel)
    assert "# Lifecycle Funnel" in report
    assert "Pump detected" in report


def test_render_lifecycle_funnel_no_events():
    assert "No events" in ld.render_lifecycle_funnel({"n_pump_detected": 0})


def _flat_ohlcv(n: int = 400, base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 300_000, "open": [base] * n, "high": [base] * n,
            "low": [base] * n, "close": [base] * n,
        }
    )


def test_anchor_idx_for_pump_detected():
    ohlcv = _flat_ohlcv()
    event = _event(pump_detected_ts=int(ohlcv["timestamp"].iloc[50]))
    idx = ld._anchor_idx_for(event, "pump_detected", ohlcv, pump_start_idx=0)
    assert idx == 50


def test_anchor_idx_for_climax_and_structure_break_translate_from_wide_coords():
    ohlcv = _flat_ohlcv()
    event = _event(climax_local_idx=10, structure_break_bar_local_idx=15)
    assert ld._anchor_idx_for(event, "climax", ohlcv, pump_start_idx=100) == 110
    assert ld._anchor_idx_for(event, "structure_break", ohlcv, pump_start_idx=100) == 115


def test_anchor_idx_for_missing_field_returns_none():
    ohlcv = _flat_ohlcv()
    event = _event(climax_local_idx=None)
    assert ld._anchor_idx_for(event, "climax", ohlcv, pump_start_idx=100) is None


def test_anchor_idx_for_first_exhaustion_signal_takes_earliest():
    ohlcv = _flat_ohlcv()
    event = _event(exhaustion_component_first_true_bar={"structure_break": 20, "cvd_bearish_divergence": 12, "avwap_loss": None})
    assert ld._anchor_idx_for(event, "first_exhaustion_signal", ohlcv, pump_start_idx=0) == 12


def test_compute_first_passage_by_anchor_produces_rows_with_component_flags():
    n = 400
    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame(
        {"timestamp": np.arange(n) * 300_000, "open": close, "high": close * 1.02, "low": close * 0.98, "close": close}
    )
    event = _event(
        event_id="e1", exchange="bybit", symbol="TESTUSDT", pump_detected_ts=int(ohlcv["timestamp"].iloc[50]),
        climax_local_idx=45, exhaustion_component_first_true_bar={"structure_break": 48},
    )
    result = ld.compute_first_passage_by_anchor(
        [event], {("bybit", "TESTUSDT"): ohlcv}, bar_interval_ms=300_000, lookback_bars=20
    )
    assert not result.empty
    assert set(result["anchor"]) <= set(ld.ANCHOR_NAMES)
    assert "structure_break" in result.columns  # component flag column present
    assert result["structure_break"].iloc[0] == True  # noqa: E712


def test_summarize_by_anchor_empty():
    result = ld.summarize_by_anchor(pd.DataFrame())
    assert len(result) == 0


def test_first_passage_single_feature_counterfactual():
    df = pd.DataFrame(
        {
            "anchor": ["pump_detected"] * 4,
            "outcome": ["profit", "profit", "adverse", "adverse"],
            "structure_break": [True, True, False, False],
            **{name: [False] * 4 for name in ld.COMPONENT_NAMES if name != "structure_break"},
        }
    )
    result = ld.first_passage_single_feature_counterfactual(df)
    row = result.set_index("feature").loc["structure_break"]
    assert row["p_outcome_given_condition"] == pytest.approx(1.0)
    assert row["p_outcome_given_not_condition"] == pytest.approx(0.0)


def test_first_passage_combination_counterfactual():
    base = {name: [False] * 4 for name in ld.COMPONENT_NAMES}
    base["cvd_bearish_divergence"] = [True, True, False, False]
    base["oi_rollover"] = [True, False, True, False]
    base["anchor"] = ["pump_detected"] * 4
    base["outcome"] = ["profit", "adverse", "adverse", "adverse"]
    df = pd.DataFrame(base)
    result = ld.first_passage_combination_counterfactual(df, [("cvd_bearish_divergence", "oi_rollover")])
    row = result.iloc[0]
    assert row["n_with_condition"] == 1
    assert row["p_outcome_given_condition"] == pytest.approx(1.0)


def test_render_first_passage_report_smoke():
    by_anchor = pd.DataFrame([{"anchor": "pump_detected", "n": 10, "p_profit": 0.3, "p_adverse": 0.5, "p_timeout": 0.2, "median_bars_to_profit": 4.0}])
    single = pd.DataFrame([{"feature": "structure_break", "n_with_condition": 5, "n_without_condition": 5, "p_outcome_given_condition": 0.4, "p_outcome_given_not_condition": 0.2}])
    combo = pd.DataFrame([{"combination": "a + b", "n_with_condition": 2, "n_without_condition": 8, "p_outcome_given_condition": 0.5, "p_outcome_given_not_condition": 0.25}])
    report = ld.render_first_passage_report(by_anchor, single, combo, "standard_4h")
    assert "First-Passage Diagnostics" in report
    assert "structure_break" in report
