from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import matched_controls as mc


def test_select_matched_controls_exact_bucket_match():
    candidates = pd.DataFrame({"event_id": ["c1", "c2"], "period": ["early", "late"], "liquidity_bucket": ["LOW", "HIGH"]})
    pool = pd.DataFrame(
        {
            "event_id": ["p1", "p2", "p3", "p4", "p5"],
            "period": ["early", "early", "late", "late", "early"],
            "liquidity_bucket": ["LOW", "HIGH", "HIGH", "LOW", "LOW"],
        }
    )
    result = mc.select_matched_controls(candidates, pool, ["period", "liquidity_bucket"], n_controls_per_candidate=5, seed=1)
    c1_controls = set(result.loc[result["candidate_event_id"] == "c1", "control_event_id"])
    c2_controls = set(result.loc[result["candidate_event_id"] == "c2", "control_event_id"])
    assert c1_controls == {"p1", "p5"}  # only pool events matching period=early, liquidity_bucket=LOW
    assert c2_controls == {"p3"}  # only pool event matching period=late, liquidity_bucket=HIGH


def test_select_matched_controls_excludes_candidate_ids_from_pool():
    candidates = pd.DataFrame({"event_id": ["c1"], "period": ["early"]})
    pool = pd.DataFrame({"event_id": ["c1", "p1"], "period": ["early", "early"]})  # c1 appears in the pool too - must be excluded
    result = mc.select_matched_controls(candidates, pool, ["period"], n_controls_per_candidate=5, seed=1)
    assert set(result["control_event_id"]) == {"p1"}


def test_select_matched_controls_no_bucket_match_returns_no_rows_for_that_candidate():
    candidates = pd.DataFrame({"event_id": ["c1"], "period": ["early"]})
    pool = pd.DataFrame({"event_id": ["p1"], "period": ["late"]})
    result = mc.select_matched_controls(candidates, pool, ["period"], seed=1)
    assert result.empty


def test_select_matched_controls_caps_at_available_pool_size():
    candidates = pd.DataFrame({"event_id": ["c1"], "period": ["early"]})
    pool = pd.DataFrame({"event_id": ["p1", "p2"], "period": ["early", "early"]})
    result = mc.select_matched_controls(candidates, pool, ["period"], n_controls_per_candidate=10, seed=1)
    assert len(result) == 2  # requested 10, but only 2 are available - no crash, no duplicates
    assert set(result["control_event_id"]) == {"p1", "p2"}
