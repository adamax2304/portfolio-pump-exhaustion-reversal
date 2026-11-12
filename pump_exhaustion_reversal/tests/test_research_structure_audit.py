from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.research import structure_audit
from pump_exhaustion_reversal.tests.test_metrics import _event

NEW_PUMP_HIGH_REASON = (
    "price made a new pump high beyond the pre-break swing high - the pump resumed, it did not exhaust"
)
HIGHER_LOW_REASON = "a new higher low confirmed above the broken level - bullish structure reforming"


# ---- new_pump_high_timing_distribution -------------------------------------


def test_new_pump_high_timing_lag_and_percentile_math_on_known_set():
    # lags (dynamic - structure_break) = 2, 4, 6, 8 -> hand-computable linear-interpolation percentiles
    events = [
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=10, dynamic_invalidation_bar_local_idx=12,
        ),
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=10, dynamic_invalidation_bar_local_idx=14,
        ),
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=10, dynamic_invalidation_bar_local_idx=16,
        ),
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=10, dynamic_invalidation_bar_local_idx=18,
        ),
    ]
    result = structure_audit.new_pump_high_timing_distribution(events)

    assert result["n_matched"] == 4
    assert result["n_excluded_missing_index"] == 0
    assert result["mean_bars"] == pytest.approx(5.0)
    # numpy linear-interpolation percentiles of sorted [2, 4, 6, 8]:
    assert result["p25_bars"] == pytest.approx(3.5)
    assert result["p50_bars"] == pytest.approx(5.0)
    assert result["p75_bars"] == pytest.approx(6.5)
    assert result["p90_bars"] == pytest.approx(7.4)
    assert result["p95_bars"] == pytest.approx(7.7)
    # overlapping "at most N" buckets, not mutually exclusive
    assert result["within_1_bar"] == 0
    assert result["within_2_bars"] == 1
    assert result["within_3_bars"] == 1
    assert result["within_6_bars"] == 3
    assert result["within_12_bars"] == 4
    assert result["within_24_bars"] == 4
    assert result["over_24_bars"] == 0


def test_new_pump_high_timing_excludes_missing_indices_and_other_reasons():
    events = [
        # matched: both indices present, lag=4
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=5, dynamic_invalidation_bar_local_idx=9,
        ),
        # reason matches but structure_break_bar_local_idx missing -> excluded
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=None, dynamic_invalidation_bar_local_idx=9,
        ),
        # reason matches but dynamic_invalidation_bar_local_idx missing -> excluded
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=5, dynamic_invalidation_bar_local_idx=None,
        ),
        # different invalidation reason entirely -> not matched, not excluded either
        _event(
            invalidation_reason=HIGHER_LOW_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=5, dynamic_invalidation_bar_local_idx=9,
        ),
        # no invalidation_reason at all (e.g. a CLOSED/trade event) -> not matched, not excluded
        _event(invalidation_reason=None, final_state="CLOSED", structure_break_bar_local_idx=5, dynamic_invalidation_bar_local_idx=9),
    ]
    result = structure_audit.new_pump_high_timing_distribution(events)
    assert result["n_matched"] == 1
    assert result["n_excluded_missing_index"] == 2
    assert result["mean_bars"] == pytest.approx(4.0)


def test_new_pump_high_timing_zero_matches_returns_placeholder_dict_not_raise():
    result = structure_audit.new_pump_high_timing_distribution([])
    assert result["n_matched"] == 0
    assert result["n_excluded_missing_index"] == 0
    assert np.isnan(result["mean_bars"])
    assert np.isnan(result["p25_bars"])
    assert np.isnan(result["p50_bars"])
    assert np.isnan(result["p95_bars"])
    assert result["within_1_bar"] == 0
    assert result["within_24_bars"] == 0
    assert result["over_24_bars"] == 0


def test_new_pump_high_timing_zero_matches_still_reports_real_exclusion_count():
    # every event matches the reason text but is missing an index -> n_matched=0 yet
    # n_excluded_missing_index must still reflect the real count, not be zeroed out too.
    events = [
        _event(invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED", structure_break_bar_local_idx=None, dynamic_invalidation_bar_local_idx=9),
        _event(invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED", structure_break_bar_local_idx=5, dynamic_invalidation_bar_local_idx=None),
    ]
    result = structure_audit.new_pump_high_timing_distribution(events)
    assert result["n_matched"] == 0
    assert result["n_excluded_missing_index"] == 2
    assert np.isnan(result["mean_bars"])


# ---- classify_pivot_significance / broken_hl_significance_distribution -----


def test_classify_pivot_significance_unknown_on_nan():
    assert structure_audit.classify_pivot_significance(np.nan) == "UNKNOWN"


def test_classify_pivot_significance_bucket_boundaries():
    micro_max = structure_audit.MICRO_HL_MAX_ATR
    local_max = structure_audit.LOCAL_HL_MAX_ATR
    meaningful_max = structure_audit.MEANINGFUL_HL_MAX_ATR

    assert structure_audit.classify_pivot_significance(0.0) == "MICRO_HL"
    assert structure_audit.classify_pivot_significance(micro_max - 0.01) == "MICRO_HL"
    assert structure_audit.classify_pivot_significance(micro_max) == "LOCAL_HL"  # lower bound inclusive
    assert structure_audit.classify_pivot_significance(local_max - 0.01) == "LOCAL_HL"
    assert structure_audit.classify_pivot_significance(local_max) == "MEANINGFUL_HL"  # lower bound inclusive
    assert structure_audit.classify_pivot_significance(meaningful_max - 0.01) == "MEANINGFUL_HL"
    assert structure_audit.classify_pivot_significance(meaningful_max) == "MAJOR_HL"  # lower bound inclusive
    assert structure_audit.classify_pivot_significance(meaningful_max + 100) == "MAJOR_HL"


def test_broken_hl_significance_distribution_counts_each_bucket():
    df = pd.DataFrame({"swing_distance_atr": [0.1, 0.4, 0.5, 1.0, 1.5, 3.9, 4.0, 10.0, np.nan]})
    counts = structure_audit.broken_hl_significance_distribution(df)
    assert counts["MICRO_HL"] == 2
    assert counts["LOCAL_HL"] == 2
    assert counts["MEANINGFUL_HL"] == 2
    assert counts["MAJOR_HL"] == 2
    assert counts["UNKNOWN"] == 1


# ---- render_structure_audit_report ------------------------------------------


def test_render_structure_audit_report_smoke_without_hl_distribution():
    timing = structure_audit.new_pump_high_timing_distribution([])
    report = structure_audit.render_structure_audit_report(timing, None)
    assert "# Structure Audit" in report
    assert "Not computed for this run" in report
    assert "nothing to report" in report.lower()


def test_render_structure_audit_report_smoke_with_hl_distribution():
    events = [
        _event(
            invalidation_reason=NEW_PUMP_HIGH_REASON, final_state="INVALIDATED",
            structure_break_bar_local_idx=0, dynamic_invalidation_bar_local_idx=5,
        )
    ]
    timing = structure_audit.new_pump_high_timing_distribution(events)
    hl_counts = pd.Series({"MICRO_HL": 3, "MAJOR_HL": 1})
    report = structure_audit.render_structure_audit_report(timing, hl_counts)
    assert "# Structure Audit" in report
    assert "MICRO_HL" in report
    assert "MAJOR_HL" in report
    assert "n_matched=1" in report
