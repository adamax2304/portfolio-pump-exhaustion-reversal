from __future__ import annotations

import numpy as np
import pytest

from pump_exhaustion_reversal.research import classifier_audit
from pump_exhaustion_reversal.tests.test_metrics import _event


def _events():
    return [
        _event(pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=np.nan, short_liq_zscore=np.nan,
               pump_class_reasons=["no signature matched"], pump_class_data_confidence_notes=["residual fallback"]),
        _event(pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=np.nan, short_liq_zscore=np.nan),
        _event(pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=np.nan, short_liq_zscore=np.nan),
        _event(pump_class="LEVERAGED_LONG", pump_class_confidence="high", oi_zscore=3.5, short_liq_zscore=np.nan),
        _event(pump_class="SHORT_SQUEEZE", pump_class_confidence="high", oi_zscore=np.nan, short_liq_zscore=6.0),
        _event(pump_class="SPOT_ORGANIC", pump_class_confidence="low", oi_zscore=1.0, short_liq_zscore=np.nan),  # has OI but still fell through
    ]


def test_class_distribution_counts_and_shares():
    dist = classifier_audit.class_distribution(_events())
    spot_row = dist[dist["pump_class"] == "SPOT_ORGANIC"].iloc[0]
    assert spot_row["n"] == 4
    assert spot_row["share"] == pytest.approx(4 / 6)
    assert spot_row["pct_low_confidence"] == pytest.approx(1.0)
    lev_row = dist[dist["pump_class"] == "LEVERAGED_LONG"].iloc[0]
    assert lev_row["pct_low_confidence"] == pytest.approx(0.0)


def test_class_distribution_empty():
    dist = classifier_audit.class_distribution([])
    assert len(dist) == 0


def test_oi_availability_vs_class_separates_groups():
    result = classifier_audit.oi_availability_vs_class(_events())
    available = result[result["oi_available"]]
    unavailable = result[~result["oi_available"]]
    # 2 events have oi_zscore present (LEVERAGED_LONG, and the one SPOT_ORGANIC with oi_zscore=1.0)
    assert available["n"].sum() == 2
    assert unavailable["n"].sum() == 4
    assert set(available["pump_class"]) == {"LEVERAGED_LONG", "SPOT_ORGANIC"}


def test_reasons_and_notes_sample_respects_limit_and_class_filter():
    sample = classifier_audit.reasons_and_notes_sample(_events(), "SPOT_ORGANIC", limit=2)
    assert len(sample) == 2
    assert all({"event_id", "symbol", "exchange", "confidence", "reasons", "data_confidence_notes"} <= set(row) for row in sample)
    assert sample[0]["reasons"] == ["no signature matched"]


def test_fallback_bias_check_higher_spot_share_without_derivatives_signal():
    bias = classifier_audit.fallback_bias_check(_events())
    # no-signal group: 3 events, all SPOT_ORGANIC -> share 1.0
    # has-signal group: 3 events (LEVERAGED_LONG, SHORT_SQUEEZE, 1 SPOT_ORGANIC with oi present) -> share 1/3
    assert bias["n_no_oi_or_liq_signal"] == 3
    assert bias["spot_organic_share_no_signal"] == pytest.approx(1.0)
    assert bias["n_has_oi_or_liq_signal"] == 3
    assert bias["spot_organic_share_has_signal"] == pytest.approx(1 / 3)


def test_fallback_bias_check_empty():
    assert classifier_audit.fallback_bias_check([]) == {}


def test_render_classifier_audit_report_smoke():
    report = classifier_audit.render_classifier_audit_report({"binance": _events(), "bybit": []})
    assert "# Pump Classifier Audit" in report
    assert "SPOT_ORGANIC" in report
    assert "residual fallback" in report.lower() or "residual" in report.lower()
    assert "No events." in report  # bybit's empty-list branch
