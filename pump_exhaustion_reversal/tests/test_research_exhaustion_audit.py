from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.config import ExhaustionScoreWeights
from pump_exhaustion_reversal.research import exhaustion_audit
from pump_exhaustion_reversal.tests.test_metrics import _event


def _ev(first_true: dict, contribution: dict | None = None):
    return _event(
        exhaustion_component_first_true_bar={**dict.fromkeys(exhaustion_audit.COMPONENT_NAMES), **first_true},
        exhaustion_component_contribution_at_peak=contribution or {},
    )


def test_hit_rate_by_component_counts_correctly():
    events = [
        _ev({"structure_break": 3, "cvd_bearish_divergence": 5}),
        _ev({"structure_break": 2}),
        _ev({}),
    ]
    hit_rates = exhaustion_audit.hit_rate_by_component(events)
    sb = hit_rates.set_index("component").loc["structure_break"]
    assert sb["n_fired"] == 2
    assert sb["hit_rate"] == pytest.approx(2 / 3)
    cvd = hit_rates.set_index("component").loc["cvd_bearish_divergence"]
    assert cvd["n_fired"] == 1


def test_hit_rate_by_component_empty():
    result = exhaustion_audit.hit_rate_by_component([])
    assert len(result) == 0


def test_timing_by_component_percentiles():
    events = [_ev({"structure_break": b}) for b in (2, 4, 6, 8, 10)]
    timing = exhaustion_audit.timing_by_component(events)
    row = timing.set_index("component").loc["structure_break"]
    assert row["n"] == 5
    assert row["median_bar"] == pytest.approx(6.0)


def test_timing_by_component_zero_hits_is_nan_not_zero():
    events = [_ev({})]
    timing = exhaustion_audit.timing_by_component(events)
    row = timing.set_index("component").loc["structure_break"]
    assert row["n"] == 0
    assert np.isnan(row["median_bar"])


def test_co_occurrence_matrix_basic():
    events = [
        _ev({"structure_break": 1, "cvd_bearish_divergence": 2}),
        _ev({"structure_break": 1}),
        _ev({"cvd_bearish_divergence": 3}),
    ]
    matrix = exhaustion_audit.co_occurrence_matrix(events)
    # structure_break fired in 2 events, of which 1 also had cvd_bearish_divergence
    assert matrix.loc["structure_break", "cvd_bearish_divergence"] == pytest.approx(0.5)
    # cvd_bearish_divergence fired in 2 events, of which 1 also had structure_break
    assert matrix.loc["cvd_bearish_divergence", "structure_break"] == pytest.approx(0.5)
    assert pd.isna(matrix.loc["structure_break", "structure_break"])


def test_co_occurrence_matrix_empty():
    matrix = exhaustion_audit.co_occurrence_matrix([])
    assert matrix.shape == (len(exhaustion_audit.COMPONENT_NAMES), len(exhaustion_audit.COMPONENT_NAMES))
    assert matrix.isna().all().all()


def test_signal_count_distribution():
    events = [_ev({}), _ev({"structure_break": 1}), _ev({"structure_break": 1, "cvd_bearish_divergence": 2})]
    dist = exhaustion_audit.signal_count_distribution(events)
    assert dist.loc[0] == 1
    assert dist.loc[1] == 1
    assert dist.loc[2] == 1


def test_score_decomposition_uses_configured_weights_and_excludes_missing():
    weights = ExhaustionScoreWeights()  # defaults sum to 100 (structure_break=20, ...)
    events = [
        _ev({}, contribution={"structure_break": 100.0, "cvd_bearish_divergence": 0.0}),
        _ev({}, contribution={"structure_break": 100.0}),  # cvd missing here - excluded from its average, not counted as 0
    ]
    decomp = exhaustion_audit.score_decomposition(events, weights)
    sb = decomp.set_index("component").loc["structure_break"]
    assert sb["n_present"] == 2
    assert sb["avg_weighted_contribution"] == pytest.approx(weights.structure_break)  # both fired at full weight
    cvd = decomp.set_index("component").loc["cvd_bearish_divergence"]
    assert cvd["n_present"] == 1  # only the first event had a real (non-missing) value
    assert cvd["avg_weighted_contribution"] == pytest.approx(0.0)  # it was present but not firing


def test_render_exhaustion_audit_report_smoke():
    events = [_ev({"structure_break": 1}, contribution={"structure_break": 100.0})]
    weights = ExhaustionScoreWeights()
    report = exhaustion_audit.render_exhaustion_audit_report(
        exhaustion_audit.hit_rate_by_component(events),
        exhaustion_audit.timing_by_component(events),
        exhaustion_audit.co_occurrence_matrix(events),
        exhaustion_audit.signal_count_distribution(events),
        exhaustion_audit.score_decomposition(events, weights),
    )
    assert "# Exhaustion Feature Audit" in report
    assert "structure_break" in report
