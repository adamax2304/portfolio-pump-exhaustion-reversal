from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import predictive_value as pv
from pump_exhaustion_reversal.tests.test_metrics import _event


def _fp_row(event_id, max_upside, dd_24h):
    return {"event_id": event_id, "maximum_additional_upside": max_upside, "drawdown_from_eventual_peak_24h": dd_24h}


def test_build_feature_outcome_frame_joins_correctly():
    events = [
        _event(event_id="e1", pump_class="SPOT_ORGANIC", pump_score=70.0, exhaustion_component_first_true_bar={"structure_break": 2}),
        _event(event_id="e2", pump_class="LEVERAGED_LONG", pump_score=80.0, exhaustion_component_first_true_bar={}),
    ]
    fp = pd.DataFrame([_fp_row("e1", 0.1, 0.3), _fp_row("e2", 0.5, 0.05)])
    frame = pv.build_feature_outcome_frame(events, fp)
    assert len(frame) == 2
    row1 = frame.set_index("event_id").loc["e1"]
    assert row1["structure_break"] == True  # noqa: E712
    assert row1["cvd_bearish_divergence"] == False  # noqa: E712
    assert row1["drawdown_from_eventual_peak_24h"] == pytest.approx(0.3)
    assert row1["pump_class"] == "SPOT_ORGANIC"


def test_single_feature_counterfactual_computes_conditional_probabilities():
    frame = pd.DataFrame(
        {
            "structure_break": [True, True, False, False],
            **{name: [False, False, False, False] for name in pv.COMPONENT_NAMES if name != "structure_break"},
            "drawdown_from_eventual_peak_24h": [0.3, 0.25, 0.01, 0.02],
        }
    )
    table = pv.single_feature_counterfactual(frame, "drawdown_from_eventual_peak_24h", 0.20)
    row = table.set_index("feature").loc["structure_break"]
    assert row["p_outcome_given_condition"] == pytest.approx(1.0)
    assert row["p_outcome_given_not_condition"] == pytest.approx(0.0)


def test_combination_counterfactual_requires_all_features():
    base = {name: [False] * 4 for name in pv.COMPONENT_NAMES}
    base["cvd_bearish_divergence"] = [True, True, False, False]
    base["oi_rollover"] = [True, False, True, False]
    base["drawdown_from_eventual_peak_24h"] = [0.30, 0.01, 0.01, 0.01]
    frame = pd.DataFrame(base)
    table = pv.combination_counterfactual(frame, "drawdown_from_eventual_peak_24h", 0.20)
    row = table.set_index("combination").loc["cvd_bearish_divergence + oi_rollover"]
    assert row["n_with_condition"] == 1  # only row 0 has BOTH true
    assert row["p_outcome_given_condition"] == pytest.approx(1.0)


def test_label_reversal_continuation_not_mutually_exclusive():
    frame = pd.DataFrame({"drawdown_from_eventual_peak_24h": [0.25, 0.05], "maximum_additional_upside": [0.60, 0.60]})
    labeled = pv.label_reversal_continuation(frame)
    # row 0: both a big eventual reversal AND a big prior continuation - valid, not contradictory
    assert bool(labeled.loc[0, "REVERSAL_20"]) is True
    assert bool(labeled.loc[0, "CONTINUATION_50"]) is True
    assert bool(labeled.loc[1, "REVERSAL_20"]) is False
    assert bool(labeled.loc[1, "CONTINUATION_50"]) is True


def test_compare_reversal_vs_continuation_returns_effect_size_report():
    frame = pd.DataFrame(
        {
            "pump_score": [10, 20, 30, 80, 90, 95],
            "REVERSAL_20": [True, True, True, False, False, False],
            "CONTINUATION_20": [False, False, False, True, True, True],
        }
    )
    report = pv.compare_reversal_vs_continuation(frame, "pump_score")
    assert report["n_a"] == 3
    assert report["n_b"] == 3
    assert report["median_a"] < report["median_b"]


def test_logistic_regression_check_skips_when_n_too_small():
    frame = pd.DataFrame({**{name: [True, False, True] for name in pv.COMPONENT_NAMES}, "REVERSAL_20": [True, False, True]})
    result = pv.logistic_regression_check(frame)
    assert result["sufficient_n"] is False
    assert result["n"] == 3


def test_logistic_regression_check_runs_on_sufficient_synthetic_data():
    rng = np.random.default_rng(3)
    n = 100
    structure_break = rng.integers(0, 2, n).astype(bool)
    other_cols = {name: rng.integers(0, 2, n).astype(bool) for name in pv.COMPONENT_NAMES if name != "structure_break"}
    # target strongly (not perfectly) driven by structure_break plus noise, so the fit is well-behaved
    noise = rng.random(n)
    target = structure_break & (noise > 0.2)
    frame = pd.DataFrame({"structure_break": structure_break, **other_cols, "REVERSAL_20": target})
    result = pv.logistic_regression_check(frame, train_frac=0.7)
    assert result["sufficient_n"] is True
    if "coefficients" in result:
        assert "structure_break" in result["coefficients"]
        assert 0.0 <= result["oos_accuracy"] <= 1.0


def test_render_predictive_value_report_smoke():
    single_tables = {"24h_10pct": pd.DataFrame([{"feature": "structure_break", "n_with_condition": 5, "n_without_condition": 3, "p_outcome_given_condition": 0.6, "p_outcome_given_not_condition": 0.1}])}
    combo_tables = {"24h_10pct": pd.DataFrame([{"combination": "a + b", "n_with_condition": 2, "n_without_condition": 6, "p_outcome_given_condition": 0.5, "p_outcome_given_not_condition": 0.2}])}
    rvc = {"label_a": "REVERSAL_20", "label_b": "CONTINUATION_20", "n_a": 3, "n_b": 3, "median_a": 20.0, "median_b": 90.0, "mann_whitney": {"p_value": 0.04}, "cliffs_delta": -0.8}
    logit = {"sufficient_n": False, "n": 5, "note": "too few events"}
    report = pv.render_predictive_value_report(single_tables, combo_tables, rvc, logit)
    assert "# Feature Predictive Value" in report
    assert "structure_break" in report
    assert "too few events" in report
