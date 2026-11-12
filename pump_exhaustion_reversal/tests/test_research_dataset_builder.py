from __future__ import annotations

import pandas as pd
import pytest

from pump_exhaustion_reversal.research import dataset_builder
from pump_exhaustion_reversal.tests.test_metrics import _event


def test_build_event_level_frame_flattens_key_fields():
    events = [_event(event_id="e1", symbol="AAAUSDT", pump_class="SPOT_ORGANIC", pump_class_reasons=["r1"])]
    frame = dataset_builder.build_event_level_frame(events)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["event_id"] == "e1"
    assert row["symbol"] == "AAAUSDT"
    assert row["pump_class_reasons"] == ["r1"]
    assert row["has_trade"] == False  # noqa: E712


def test_build_event_level_frame_empty():
    frame = dataset_builder.build_event_level_frame([])
    assert frame.empty


def test_join_forward_paths_left_join_by_event_id():
    event_frame = pd.DataFrame([{"event_id": "e1", "symbol": "AAAUSDT"}, {"event_id": "e2", "symbol": "BBBUSDT"}])
    forward_paths = pd.DataFrame([{"event_id": "e1", "maximum_additional_upside": 0.3}])
    joined = dataset_builder.join_forward_paths(event_frame, forward_paths)
    assert joined.set_index("event_id").loc["e1", "maximum_additional_upside"] == pytest.approx(0.3)
    assert pd.isna(joined.set_index("event_id").loc["e2", "maximum_additional_upside"])


def test_join_forward_paths_drops_colliding_columns_from_forward_side():
    event_frame = pd.DataFrame([{"event_id": "e1", "symbol": "AAAUSDT"}])
    forward_paths = pd.DataFrame([{"event_id": "e1", "symbol": "SHOULD_NOT_OVERWRITE"}])
    joined = dataset_builder.join_forward_paths(event_frame, forward_paths)
    assert joined.iloc[0]["symbol"] == "AAAUSDT"


def test_join_underlying_event_linking_maps_both_sides():
    event_frame = pd.DataFrame([{"event_id": "bin1"}, {"event_id": "byb1"}, {"event_id": "solo1"}])
    linked_df = pd.DataFrame([{"underlying_event_id": "u1", "binance_event_id": "bin1", "bybit_event_id": "byb1"}])
    joined = dataset_builder.join_underlying_event_linking(event_frame, linked_df)
    idx = joined.set_index("event_id")
    assert idx.loc["bin1", "underlying_event_id"] == "u1"
    assert idx.loc["byb1", "underlying_event_id"] == "u1"
    assert pd.isna(idx.loc["solo1", "underlying_event_id"])


def test_join_underlying_event_linking_handles_empty_linked_df():
    event_frame = pd.DataFrame([{"event_id": "e1"}])
    joined = dataset_builder.join_underlying_event_linking(event_frame, pd.DataFrame())
    assert "underlying_event_id" in joined.columns
    assert pd.isna(joined.iloc[0]["underlying_event_id"])


def test_save_research_dataset_roundtrip(tmp_path):
    events = [_event(event_id="e1", exhaustion_component_first_true_bar={"structure_break": 3})]
    frame = dataset_builder.build_event_level_frame(events)
    out_path = tmp_path / "nested" / "dataset.parquet"
    saved_path = dataset_builder.save_research_dataset(frame, out_path)
    assert saved_path.exists()
    reloaded = pd.read_parquet(saved_path)
    assert reloaded.iloc[0]["event_id"] == "e1"
    assert reloaded.iloc[0]["exhaustion_component_first_true_bar"]["structure_break"] == 3
