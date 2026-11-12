"""Phase 3B points 7-8: fold construction only - no fitting, no scoring.
Every fold is a set of event_ids; the caller filters snapshots_df by
event_id membership. A snapshot row is never split from its own event
across folds (point 7's core rule), and every symbol's events stay together
within a single symbol-holdout side (point 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_expanding_temporal_folds(
    events_df: pd.DataFrame, event_col: str = "event_id", ts_col: str = "pump_detected_ts",
    n_folds: int = 4, min_train_fraction: float = 0.4,
) -> list[dict]:
    """Point 7: expanding-window temporal folds by EVENT, sorted by
    detection time - never a random split. Fold k trains on every event up
    to a chronological point and validates on the NEXT chronological slice
    (train always strictly precedes its own validation slice in time; later
    folds' train sets are supersets of earlier ones' - the "expanding
    window" the point asks for, as opposed to a fixed-width rolling
    window). `min_train_fraction` is the earliest fold's training size as a
    fraction of all events - fixed before looking at any fit result."""
    unique_events = events_df[[event_col, ts_col]].drop_duplicates(subset=event_col).sort_values(ts_col)
    n = len(unique_events)
    ids = unique_events[event_col].tolist()
    val_width_frac = (1.0 - min_train_fraction) / n_folds

    folds = []
    for k in range(n_folds):
        train_end = min(max(int(round(n * (min_train_fraction + k * val_width_frac))), 1), n)
        val_end = min(int(round(n * (min_train_fraction + (k + 1) * val_width_frac))), n)
        if val_end <= train_end:
            continue
        folds.append(
            {
                "fold": k, "train_event_ids": set(ids[:train_end]), "val_event_ids": set(ids[train_end:val_end]),
                "n_train_events": train_end, "n_val_events": val_end - train_end,
            }
        )
    return folds


def build_symbol_holdout_fold(
    events_df: pd.DataFrame, symbol_col: str = "symbol", event_col: str = "event_id",
    holdout_fraction: float = 0.2, seed: int = 7,
) -> dict:
    """Point 8's robustness check: withhold an entire random subset of
    SYMBOLS (not events) from training - every event belonging to a held-out
    symbol goes to validation, every event of a kept symbol goes to
    training. If the model degrades sharply here versus the temporal folds,
    it is partly learning token-specific quirks rather than general
    microstructure, even if it looked fine when the same symbols appeared
    in both train and validation (as the temporal folds allow)."""
    symbols = sorted(events_df[symbol_col].dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    n_holdout = max(1, round(len(symbols) * holdout_fraction))
    holdout_symbols = set(rng.choice(symbols, size=n_holdout, replace=False).tolist())
    train_events = set(events_df.loc[~events_df[symbol_col].isin(holdout_symbols), event_col])
    val_events = set(events_df.loc[events_df[symbol_col].isin(holdout_symbols), event_col])
    return {
        "holdout_symbols": holdout_symbols, "train_event_ids": train_events, "val_event_ids": val_events,
        "n_train_events": len(train_events), "n_val_events": len(val_events),
    }
