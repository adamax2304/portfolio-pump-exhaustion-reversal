"""Phase 4A point 15: for each V2 top-ranked candidate, pick control events
from the SAME categorical bucket (period/liquidity/magnitude/volatility -
whatever the caller passes as match_cols) that were NOT top-ranked, then run
the identical entry algorithm on both. If entry mechanics alone are
profitable on any falling pump of similar size/liquidity/age, V2's ranking
adds no value; the comparison is only meaningful if the control pool is
pre-filtered by the caller to exclude anything that itself cleared the
ranking threshold being tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_matched_controls(
    candidate_events: pd.DataFrame, pool_events: pd.DataFrame, match_cols: list[str],
    event_col: str = "event_id", n_controls_per_candidate: int = 1, seed: int = 7,
) -> pd.DataFrame:
    """Exact-match on every column in match_cols (already-bucketed
    categoricals - the caller quantile-buckets any continuous match
    variable before calling this). Samples WITHOUT replacement within each
    bucket where possible (a bucket with fewer available controls than
    requested returns as many as it has, never crashes or over-samples with
    duplicates). pool_events must already exclude candidate_events'
    event_ids and anything else the caller does not want eligible as a
    control (e.g. other alerted events) - this function does no ranking-
    specific filtering of its own, only bucket matching."""
    rng = np.random.default_rng(seed)
    candidate_ids = set(candidate_events[event_col])
    pool = pool_events[~pool_events[event_col].isin(candidate_ids)].dropna(subset=match_cols)
    if pool.empty:
        return pd.DataFrame(columns=["candidate_event_id", "control_event_id", *match_cols])

    grouped_pool = dict(tuple(pool.groupby(match_cols, observed=True))) if len(match_cols) > 1 else {k: g for k, g in pool.groupby(match_cols[0], observed=True)}

    rows = []
    for _, cand in candidate_events.dropna(subset=match_cols).iterrows():
        key = tuple(cand[c] for c in match_cols) if len(match_cols) > 1 else cand[match_cols[0]]
        bucket = grouped_pool.get(key)
        if bucket is None or bucket.empty:
            continue
        n_pick = min(n_controls_per_candidate, len(bucket))
        picked = bucket.sample(n=n_pick, random_state=int(rng.integers(0, 2**31 - 1)))
        for _, control_row in picked.iterrows():
            rows.append({"candidate_event_id": cand[event_col], "control_event_id": control_row[event_col], **{c: cand[c] for c in match_cols}})
    return pd.DataFrame(rows)
