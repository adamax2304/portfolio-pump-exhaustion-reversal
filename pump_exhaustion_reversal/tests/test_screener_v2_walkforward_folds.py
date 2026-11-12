from __future__ import annotations

import pandas as pd
import pytest

from pump_exhaustion_reversal.screener_v2 import walkforward_folds as wf


def _events_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "pump_detected_ts": list(range(n)), "symbol": [f"SYM{i % 10}" for i in range(n)]})


def test_expanding_temporal_folds_never_overlap_and_train_precedes_val():
    events_df = _events_df()
    folds = wf.build_expanding_temporal_folds(events_df, n_folds=4, min_train_fraction=0.4)
    assert len(folds) == 4
    for f in folds:
        assert not (f["train_event_ids"] & f["val_event_ids"])
        train_max_ts = max(int(e[1:]) for e in f["train_event_ids"])
        val_min_ts = min(int(e[1:]) for e in f["val_event_ids"])
        assert train_max_ts < val_min_ts  # every train event strictly precedes every val event chronologically


def test_expanding_temporal_folds_train_set_grows_across_folds():
    events_df = _events_df()
    folds = wf.build_expanding_temporal_folds(events_df, n_folds=4, min_train_fraction=0.4)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier["train_event_ids"] <= later["train_event_ids"]  # expanding window: strictly growing (superset)
        assert len(later["train_event_ids"]) > len(earlier["train_event_ids"])


def test_expanding_temporal_folds_covers_events_by_detection_order():
    events_df = _events_df(n=10)
    folds = wf.build_expanding_temporal_folds(events_df, n_folds=1, min_train_fraction=0.5)
    assert len(folds) == 1
    assert folds[0]["train_event_ids"] == {f"e{i}" for i in range(5)}
    assert folds[0]["val_event_ids"] == {f"e{i}" for i in range(5, 10)}


def test_symbol_holdout_fold_keeps_each_symbol_entirely_on_one_side():
    events_df = _events_df()
    result = wf.build_symbol_holdout_fold(events_df, holdout_fraction=0.3, seed=1)
    train_symbols = set(events_df.loc[events_df["event_id"].isin(result["train_event_ids"]), "symbol"])
    val_symbols = set(events_df.loc[events_df["event_id"].isin(result["val_event_ids"]), "symbol"])
    assert not (train_symbols & val_symbols)  # no symbol appears on both sides
    assert val_symbols == result["holdout_symbols"]
    assert result["n_train_events"] + result["n_val_events"] == len(events_df)
