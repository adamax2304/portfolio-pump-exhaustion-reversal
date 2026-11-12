from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import model_evaluation


def test_score_decile_curve_monotonic_lift():
    n = 1000
    rng = np.random.default_rng(1)
    score = rng.uniform(0, 1, n)
    p_reversal = 0.1 + 0.6 * score
    resolved = rng.uniform(size=n) < 0.7
    direction = np.where(rng.uniform(size=n) < p_reversal, "REVERSAL", "CONTINUATION")
    outcome = np.where(resolved, direction, "NEUTRAL")
    df = pd.DataFrame(
        {
            "event_id": [f"e{i % 100}" for i in range(n)], "symbol": [f"SYM{i % 20}" for i in range(n)],
            "score": score, "outcome": outcome,
        }
    )
    result = model_evaluation.score_decile_curve(df, "score", "outcome", top_fractions=(0.5, 0.2, 0.1))
    baseline = result.loc[result["top_pct"] == 1.0].iloc[0]
    assert baseline["directional_lift"] == pytest.approx(1.0)
    lift_05 = result.loc[result["top_pct"] == 0.5, "directional_lift"].iloc[0]
    lift_02 = result.loc[result["top_pct"] == 0.2, "directional_lift"].iloc[0]
    lift_01 = result.loc[result["top_pct"] == 0.1, "directional_lift"].iloc[0]
    assert lift_01 > lift_02 > lift_05 > 1.0


def test_score_decile_curve_empty_when_no_scored_rows():
    df = pd.DataFrame({"event_id": [], "symbol": [], "score": [], "outcome": []})
    result = model_evaluation.score_decile_curve(df, "score", "outcome", top_fractions=(0.5,))
    assert pd.isna(result.loc[result["top_pct"] == 1.0, "directional_lift"].iloc[0])


def test_ranking_by_drawdown_bucket_distinguishes_early_warning_from_confirmation_only():
    n = 20000
    rng = np.random.default_rng(7)
    drawdown = rng.uniform(0, 0.10, n)
    score = rng.uniform(0, 1, n)
    is_far = drawdown >= 0.05
    # score predicts reversal ONLY in the FAR (>=5%) bucket - the "confirmation-only screener" shape
    p_reversal = np.where(is_far, 0.1 + 0.6 * score, 0.2)
    direction = np.where(rng.uniform(size=n) < p_reversal, "REVERSAL", "CONTINUATION")
    resolved = rng.uniform(size=n) < 0.7
    outcome = np.where(resolved, direction, "NEUTRAL")
    df = pd.DataFrame(
        {
            "event_id": [f"e{i % 800}" for i in range(n)], "symbol": [f"SYM{i % 30}" for i in range(n)],
            "score": score, "outcome": outcome, "drawdown": drawdown,
        }
    )
    buckets = [("NEAR_0_5", 0.0, 0.05), ("FAR_5_10", 0.05, None)]
    result = model_evaluation.ranking_by_drawdown_bucket(df, "score", "outcome", "drawdown", buckets, top_fractions=(0.1,))

    near_top10 = result[(result["bucket"] == "NEAR_0_5") & (result["top_pct"] == 0.1)]["directional_lift"].iloc[0]
    far_top10 = result[(result["bucket"] == "FAR_5_10") & (result["top_pct"] == 0.1)]["directional_lift"].iloc[0]
    assert far_top10 > 1.3  # clear lift where the signal actually exists
    assert abs(near_top10 - 1.0) < 0.25  # little to no lift where the signal was constructed to be absent


def test_precision_at_k_perfectly_separated_scores_gives_perfect_top1_precision():
    rows = []
    rng = np.random.default_rng(3)
    for t in range(200):
        ts = t * 300_000
        n_candidates = int(rng.integers(2, 5))
        reversal_idx = int(rng.integers(0, n_candidates))
        for c in range(n_candidates):
            is_rev = c == reversal_idx
            score = rng.uniform(0.7, 1.0) if is_rev else rng.uniform(0.0, 0.5)
            rows.append({"event_id": f"t{t}_c{c}", "timestamp": ts, "score": score, "outcome": "REVERSAL" if is_rev else "CONTINUATION"})
    df = pd.DataFrame(rows)
    result = model_evaluation.precision_at_k(df, "score", "outcome", ks=(1, 3))
    assert result["n_timestamps_used"] == 200
    assert result["by_k"][1]["mean_precision_at_k_per_timestamp"] == pytest.approx(1.0)


def test_precision_at_k_deduplicates_repeated_alerts_across_bars():
    rows = []
    for t in range(5):
        rows.append({"event_id": "persistent_winner", "timestamp": t * 300_000, "score": 0.9, "outcome": "REVERSAL"})
        rows.append({"event_id": f"loser_{t}", "timestamp": t * 300_000, "score": 0.1, "outcome": "CONTINUATION"})
    df = pd.DataFrame(rows)
    result = model_evaluation.precision_at_k(df, "score", "outcome", ks=(1,))
    assert result["by_k"][1]["n_unique_events_alerted"] == 1
    assert result["by_k"][1]["n_true_alerts"] == 1
    assert result["by_k"][1]["n_false_alerts"] == 0


def test_precision_at_k_skips_timestamps_below_min_concurrent():
    df = pd.DataFrame([{"event_id": "solo", "timestamp": 0, "score": 0.5, "outcome": "REVERSAL"}])
    result = model_evaluation.precision_at_k(df, "score", "outcome", ks=(1,), min_concurrent=2)
    assert result["n_timestamps_used"] == 0
    assert result["by_k"][1]["n_unique_events_alerted"] == 0


def test_classification_metrics_perfect_separation_gives_auc_one():
    df = pd.DataFrame(
        {
            "score": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            "outcome": ["REVERSAL", "REVERSAL", "REVERSAL", "CONTINUATION", "CONTINUATION", "CONTINUATION"],
        }
    )
    result = model_evaluation.classification_metrics(df, "score", "outcome")
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["n_resolved"] == 6


def test_classification_metrics_random_scores_give_auc_near_half():
    n = 4000
    rng = np.random.default_rng(9)
    score = rng.uniform(0, 1, n)  # independent of outcome
    outcome = rng.choice(["REVERSAL", "CONTINUATION"], size=n)
    df = pd.DataFrame({"score": score, "outcome": outcome})
    result = model_evaluation.classification_metrics(df, "score", "outcome")
    assert result["roc_auc"] == pytest.approx(0.5, abs=0.03)


def test_classification_metrics_brier_score_zero_for_perfect_confident_predictions():
    df = pd.DataFrame({"score": [1.0, 1.0, 0.0, 0.0], "outcome": ["REVERSAL", "REVERSAL", "CONTINUATION", "CONTINUATION"]})
    result = model_evaluation.classification_metrics(df, "score", "outcome")
    assert result["brier_score"] == pytest.approx(0.0)


def test_first_alert_per_event_keeps_only_earliest_entry_per_event():
    rows = [
        {"event_id": "e1", "timestamp": 100, "score": 0.95},
        {"event_id": "e1", "timestamp": 200, "score": 0.96},  # e1's SECOND time in the top bucket - must be dropped
        {"event_id": "e2", "timestamp": 150, "score": 0.97},
        {"event_id": "e3", "timestamp": 50, "score": 0.1},  # never crosses the threshold
    ]
    # pad with enough low-score rows (spread out, not a single repeated value) so the 90th-percentile
    # threshold naturally lands well below 0.95 without also sweeping the padding rows in
    rng = np.random.default_rng(1)
    rows += [{"event_id": f"pad{i}", "timestamp": 1000 + i, "score": float(rng.uniform(0, 0.5))} for i in range(40)]
    df = pd.DataFrame(rows)
    result = model_evaluation.first_alert_per_event(df, "score", top_quantile=0.9, min_score_rows=10)
    assert (result["event_id"] == "e1").sum() == 1  # e1 crossed the threshold twice but is counted only once
    assert result.loc[result["event_id"] == "e1", "timestamp"].iloc[0] == 100  # the EARLIER of e1's two crossings
    assert "e2" in set(result["event_id"])
    assert "e3" not in set(result["event_id"])  # e3's score (0.1) never crossed the threshold


def test_calibration_curve_well_calibrated_scores_track_actual_rate():
    n = 2000
    rng = np.random.default_rng(5)
    score = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < score).astype(int)
    outcome = np.where(y == 1, "REVERSAL", "CONTINUATION")
    df = pd.DataFrame({"score": score, "outcome": outcome})
    result = model_evaluation.calibration_curve(df, "score", "outcome", n_bins=5)
    assert len(result) == 5
    assert (result["mean_predicted_probability"] - result["actual_reversal_rate"]).abs().max() < 0.15


def test_calibration_curve_empty_when_no_resolved_rows():
    df = pd.DataFrame({"score": [0.5, 0.6], "outcome": ["NEUTRAL", "NEUTRAL"]})
    result = model_evaluation.calibration_curve(df, "score", "outcome")
    assert result.empty
