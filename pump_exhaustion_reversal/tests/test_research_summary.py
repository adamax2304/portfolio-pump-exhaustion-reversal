from __future__ import annotations

import pandas as pd

from pump_exhaustion_reversal.research import research_summary


def test_render_research_summary_smoke_with_full_data():
    class_dist = pd.DataFrame([{"pump_class": "SPOT_ORGANIC", "n": 75, "share": 0.93, "pct_low_confidence": 1.0}])
    report = research_summary.render_research_summary(
        exchange_event_counts={"binance": 81, "bybit": 67},
        class_distributions={"binance": class_dist, "bybit": pd.DataFrame()},
        oi_hypothesis={"binance": {"n_no_oi_or_liq_signal": 70, "spot_organic_share_no_signal": 0.95, "n_has_oi_or_liq_signal": 11, "spot_organic_share_has_signal": 0.3}},
        new_high_timing={"n_matched": 76, "n_excluded_missing_index": 0, "p50_bars": 1.0, "within_3_bars": 60, "within_12_bars": 74},
        exhaustion_hit_rates=pd.DataFrame([
            {"component": "structure_break", "n_fired": 80, "hit_rate": 0.99},
            {"component": "sweep_rejection", "n_fired": 5, "hit_rate": 0.06},
        ]),
        false_negative_summary={"n": 2},
        true_negative_summary={"n": 30},
        logit_result={"sufficient_n": False, "n": 20, "note": "n=20 < 40 - too few events"},
    )
    assert "# Research Summary" in report
    assert "binance: 81 events" in report
    assert "SPOT_ORGANIC" in report
    assert "93%" in report
    assert "too few events" in report
    assert "spec section 63" in report


def test_render_research_summary_handles_empty_inputs_gracefully():
    report = research_summary.render_research_summary(
        exchange_event_counts={"binance": 0},
        class_distributions={"binance": pd.DataFrame()},
        oi_hypothesis={"binance": {}},
        new_high_timing={"n_matched": 0},
        exhaustion_hit_rates=pd.DataFrame(),
        false_negative_summary={"n": 0},
        true_negative_summary={"n": 0},
        logit_result={"sufficient_n": False, "n": 0, "note": "no events"},
    )
    assert "# Research Summary" in report
    assert "no matched events" in report
