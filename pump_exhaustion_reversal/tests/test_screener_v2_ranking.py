from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import ranking


def _synthetic_snapshots(n: int = 300, seed: int = 4) -> pd.DataFrame:
    """A feature genuinely correlated with reversal (monotonic by
    construction) plus event_id grouping, for testing decile monotonicity
    and effect-size sign without relying on real market data."""
    rng = np.random.default_rng(seed)
    feature = rng.uniform(0, 1, n)
    p_reversal = 0.2 + 0.5 * feature  # monotonic in feature: higher feature -> more likely REVERSAL
    outcome = np.where(rng.uniform(size=n) < p_reversal, "REVERSAL", np.where(rng.uniform(size=n) < 0.5, "CONTINUATION", "NEUTRAL"))
    event_id = [f"e{i % 40}" for i in range(n)]  # 40 events, uneven snapshot counts per event
    return pd.DataFrame({"event_id": event_id, "my_feature": feature, "primary_outcome": outcome})


def test_decile_analysis_shows_monotonic_reversal_trend():
    df = _synthetic_snapshots()
    result = ranking.decile_analysis(df, "my_feature", "primary_outcome", n_bins=5)
    assert len(result) == 5
    # not strictly guaranteed with randomness, but with a strong monotonic construction the top bin should clearly exceed the bottom bin
    assert result.sort_values("bin_order")["p_reversal"].iloc[-1] > result.sort_values("bin_order")["p_reversal"].iloc[0]
    assert result["n_snapshots"].sum() == len(df)


def test_decile_analysis_insufficient_unique_values_returns_empty():
    df = pd.DataFrame({"event_id": ["e1"] * 10, "my_feature": [1.0] * 10, "primary_outcome": ["REVERSAL"] * 10})
    result = ranking.decile_analysis(df, "my_feature", "primary_outcome", n_bins=5)
    assert result.empty


def test_effect_size_report_detects_positive_association():
    df = _synthetic_snapshots()
    result = ranking.effect_size_report(df, "my_feature", "primary_outcome")
    assert result["valid"] is True
    # REVERSAL snapshots should have a HIGHER my_feature than CONTINUATION snapshots -> positive cliffs_delta
    assert result["cliffs_delta"] > 0
    assert result["reversal_lift_top_vs_bottom_tercile"] > 1.0
    assert result["top_tercile_reversal_rate_ci_low"] <= result["top_tercile_reversal_rate"] <= result["top_tercile_reversal_rate_ci_high"]


def test_effect_size_report_invalid_when_single_outcome_class():
    df = pd.DataFrame({"event_id": ["e1"] * 10, "my_feature": np.arange(10.0), "primary_outcome": ["NEUTRAL"] * 10})
    result = ranking.effect_size_report(df, "my_feature", "primary_outcome")
    assert result["valid"] is False


def test_feature_stability_report_splits_by_available_dimensions():
    df = _synthetic_snapshots()
    df["period"] = np.where(np.arange(len(df)) < len(df) // 2, "early", "late")
    result = ranking.feature_stability_report(df, "my_feature", "primary_outcome", split_dims=("period", "missing_dim"))
    assert set(result["dimension"]) == {"period"}
    assert set(result["value"]) == {"early", "late"}


def test_is_stable_true_when_direction_consistent():
    stability_df = pd.DataFrame({"valid": [True, True, True], "cliffs_delta": [0.2, 0.3, 0.15]})
    result = ranking.is_stable(stability_df)
    assert result["stable"] is True
    assert result["fraction_same_direction"] == pytest.approx(1.0)


def test_is_stable_false_when_direction_flips():
    # a clean 50/50 sign split (2 positive, 2 negative) - well below the 0.7 same-direction threshold either way
    stability_df = pd.DataFrame({"valid": [True, True, True, True], "cliffs_delta": [0.3, 0.25, -0.2, -0.25]})
    result = ranking.is_stable(stability_df)
    assert result["stable"] is False
    assert result["fraction_same_direction"] == pytest.approx(0.5)


def _resolution_vs_direction_snapshots(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Phase 3A.1's central failure mode, built by construction: the feature
    raises P(leaving NEUTRAL) as it rises, but among resolved snapshots the
    REVERSAL/CONTINUATION split stays a flat 50/50 regardless of feature
    value - i.e. pure resolution power, zero direction power. This is the
    exact shape of the concern the user raised about realized_vol_1h /
    volume_current_over_climax in the delivered Phase 3A ranking."""
    rng = np.random.default_rng(seed)
    feature = rng.uniform(0, 1, n)
    p_resolved = 0.3 + 0.6 * feature
    resolved = rng.uniform(size=n) < p_resolved
    direction = np.where(rng.uniform(size=n) < 0.5, "REVERSAL", "CONTINUATION")
    outcome = np.where(resolved, direction, "NEUTRAL")
    event_id = [f"e{i % 60}" for i in range(n)]
    return pd.DataFrame({"event_id": event_id, "resolution_only_feature": feature, "primary_outcome": outcome})


def test_decile_analysis_ratio_flat_when_feature_only_has_resolution_power():
    df = _resolution_vs_direction_snapshots()
    result = ranking.decile_analysis(df, "resolution_only_feature", "primary_outcome", n_bins=5)
    ordered = result.sort_values("bin_order")
    # raw p_reversal rises across bins (resolution power alone drags it up)...
    assert ordered["p_reversal"].iloc[-1] > ordered["p_reversal"].iloc[0]
    # ...but p_reversal_given_resolved stays roughly flat - no real direction power to find
    assert (ordered["p_reversal_given_resolved"].max() - ordered["p_reversal_given_resolved"].min()) < 0.15
    assert ordered["n_resolved"].sum() == (df["primary_outcome"] != "NEUTRAL").sum()


def test_resolution_effect_size_report_detects_resolution_power():
    df = _resolution_vs_direction_snapshots()
    result = ranking.resolution_effect_size_report(df, "resolution_only_feature", "primary_outcome")
    assert result["valid"] is True
    assert result["resolution_cliffs_delta"] > 0.1


def test_resolution_effect_size_report_invalid_when_no_resolved_or_no_neutral_rows():
    df = pd.DataFrame({"event_id": ["e1"] * 10, "my_feature": np.arange(10.0), "primary_outcome": ["NEUTRAL"] * 10})
    result = ranking.resolution_effect_size_report(df, "my_feature", "primary_outcome")
    assert result["valid"] is False


def test_directional_effect_size_report_flat_when_feature_only_has_resolution_power():
    df = _resolution_vs_direction_snapshots()
    result = ranking.directional_effect_size_report(df, "resolution_only_feature", "primary_outcome")
    assert result["valid"] is True
    assert abs(result["cliffs_delta"]) < 0.1
    assert result["directional_lift_top_vs_bottom_tercile"] == pytest.approx(1.0, abs=0.25)


def test_directional_effect_size_report_detects_positive_association():
    df = _synthetic_snapshots()
    result = ranking.directional_effect_size_report(df, "my_feature", "primary_outcome")
    assert result["valid"] is True
    assert result["cliffs_delta"] > 0
    assert result["directional_lift_top_vs_bottom_tercile"] > 1.0
    assert result["directional_lift_sign"] == 1
    assert result["top_tercile_reversal_share_ci_low"] <= result["top_tercile_reversal_share"] <= result["top_tercile_reversal_share_ci_high"]


def test_directional_effect_size_report_invalid_when_no_resolved_rows():
    df = pd.DataFrame({"event_id": ["e1"] * 10, "my_feature": np.arange(10.0), "primary_outcome": ["NEUTRAL"] * 10})
    result = ranking.directional_effect_size_report(df, "my_feature", "primary_outcome")
    assert result["valid"] is False


def test_directional_feature_stability_report_splits_by_available_dimensions():
    df = _synthetic_snapshots()
    df["period"] = np.where(np.arange(len(df)) < len(df) // 2, "early", "late")
    result = ranking.directional_feature_stability_report(df, "my_feature", "primary_outcome", split_dims=("period", "missing_dim"))
    assert set(result["dimension"]) == {"period"}
    assert set(result["value"]) == {"early", "late"}


def test_classify_feature_verdict_keep_when_directional_stable_and_high_state_survives():
    directional_effect = {"valid": True, "cliffs_delta": 0.25}
    directional_stability = {"stable": True}
    high_state_effect = {"valid": True, "cliffs_delta": 0.15}
    result = ranking.classify_feature_verdict(directional_effect, directional_stability, high_state_effect)
    assert result["verdict"] == "KEEP"


def test_classify_feature_verdict_reject_when_no_directional_signal():
    directional_effect = {"valid": True, "cliffs_delta": 0.01}  # resolution-power-only shape: near-zero directional delta
    directional_stability = {"stable": True}
    result = ranking.classify_feature_verdict(directional_effect, directional_stability)
    assert result["verdict"] == "REJECT"


def test_classify_feature_verdict_reject_when_directional_effect_invalid():
    directional_effect = {"valid": False}
    directional_stability = {"stable": True}
    result = ranking.classify_feature_verdict(directional_effect, directional_stability)
    assert result["verdict"] == "REJECT"


def test_classify_feature_verdict_weak_when_high_state_sign_flips():
    directional_effect = {"valid": True, "cliffs_delta": 0.25}
    directional_stability = {"stable": True}
    high_state_effect = {"valid": True, "cliffs_delta": -0.20}  # opposite sign in the high-state cohort
    result = ranking.classify_feature_verdict(directional_effect, directional_stability, high_state_effect)
    assert result["verdict"] == "WEAK"


def test_classify_feature_verdict_weak_when_not_stable_across_slices():
    directional_effect = {"valid": True, "cliffs_delta": 0.25}
    directional_stability = {"stable": False}
    result = ranking.classify_feature_verdict(directional_effect, directional_stability)  # no high-state cohort supplied -> defaults to "ok"
    assert result["verdict"] == "WEAK"


def test_classify_feature_verdict_reject_when_neither_stable_nor_high_state_survives():
    directional_effect = {"valid": True, "cliffs_delta": 0.25}
    directional_stability = {"stable": False}
    high_state_effect = {"valid": True, "cliffs_delta": -0.20}
    result = ranking.classify_feature_verdict(directional_effect, directional_stability, high_state_effect)
    assert result["verdict"] == "REJECT"


def test_is_directionally_stable_true_when_lift_sign_consistent():
    stability_df = pd.DataFrame({"valid": [True, True, True], "directional_lift_sign": [1, 1, 1]})
    result = ranking.is_directionally_stable(stability_df)
    assert result["stable"] is True
    assert result["fraction_same_direction"] == pytest.approx(1.0)


def test_interaction_grid_reports_per_cell_directional_rate():
    n = 600
    rng = np.random.default_rng(11)
    regime = rng.choice(["UP", "DOWN"], size=n)
    feature = rng.uniform(0, 1, n)
    p_reversal = np.where(regime == "UP", 0.2 + 0.5 * feature, 0.3)  # direction depends on feature ONLY within regime==UP
    direction = np.where(rng.uniform(size=n) < p_reversal, "REVERSAL", "CONTINUATION")
    resolved = rng.uniform(size=n) < 0.7
    outcome = np.where(resolved, direction, "NEUTRAL")
    event_id = [f"e{i % 100}" for i in range(n)]
    df = pd.DataFrame({"event_id": event_id, "regime": regime, "feature": feature, "primary_outcome": outcome})

    result = ranking.interaction_grid(df, [("regime", None), ("feature", 2)], "primary_outcome")
    assert set(result["regime"]) == {"UP", "DOWN"}
    assert len(result) == 4  # 2 regimes x 2 feature bins
    up_rows = result[result["regime"] == "UP"]
    down_rows = result[result["regime"] == "DOWN"]
    up_spread = up_rows["p_reversal_given_resolved"].max() - up_rows["p_reversal_given_resolved"].min()
    down_spread = down_rows["p_reversal_given_resolved"].max() - down_rows["p_reversal_given_resolved"].min()
    assert up_spread > 0.1
    assert up_spread > down_spread


def test_interaction_grid_missing_dim_returns_empty():
    df = pd.DataFrame({"event_id": ["e1"], "regime": ["UP"], "primary_outcome": ["REVERSAL"]})
    result = ranking.interaction_grid(df, [("regime", None), ("missing_col", 2)], "primary_outcome")
    assert result.empty


def test_lead_time_analysis_measures_genuine_lead_when_signal_precedes_reversal():
    # 20 events, each with 2 snapshots: the SECOND has a high feature value
    # (top decile) AND its own forward label already resolves REVERSAL with
    # a known bars-to-resolution - exactly the "signal fires, then reversal
    # happens N bars later" shape a real lead-time signal has.
    rows = []
    for i in range(20):
        rows.append({"event_id": f"e{i}", "pump_age_minutes": 0, "my_feature": 0.1, "primary_outcome": "NEUTRAL", "primary_bars": 0, "current_price": 100.0, "event_high_so_far": 100.0})
        rows.append({"event_id": f"e{i}", "pump_age_minutes": 5, "my_feature": 0.95, "primary_outcome": "REVERSAL", "primary_bars": 3, "current_price": 98.0, "event_high_so_far": 100.0})
    df = pd.DataFrame(rows)
    result = ranking.lead_time_analysis(df, "my_feature", "primary_outcome", "primary_bars", bar_interval_ms=300_000, top_quantile=0.5)
    assert result["valid"] is True
    assert result["n_events_with_signal"] == 20
    assert result["fraction_first_entries_resolved_reversal"] == pytest.approx(1.0)
    assert result["median_time_to_reversal_minutes"] == pytest.approx(3 * 300_000 / 60_000.0)
    assert result["fraction_already_deep_in_drawdown_at_signal"] == pytest.approx(0.0)  # only 2% below the high, well under the 5% "already deep" threshold


def test_lead_time_analysis_empty_after_dropna_returns_invalid():
    df = pd.DataFrame({"event_id": ["e1"], "pump_age_minutes": [0], "my_feature": [np.nan], "primary_outcome": ["NEUTRAL"], "primary_bars": [0]})
    result = ranking.lead_time_analysis(df, "my_feature", "primary_outcome", "primary_bars", bar_interval_ms=300_000)
    assert result["valid"] is False


def test_is_directionally_stable_false_when_lift_sign_flips():
    # 3-vs-2 split (60% agreement) - below the 0.7 threshold, and avoids an
    # exact 2-2 tie where the median sign is 0 and no direction can be judged at all
    stability_df = pd.DataFrame({"valid": [True, True, True, True, True], "directional_lift_sign": [1, 1, 1, -1, -1]})
    result = ranking.is_directionally_stable(stability_df)
    assert result["stable"] is False
    assert result["fraction_same_direction"] == pytest.approx(0.6)
