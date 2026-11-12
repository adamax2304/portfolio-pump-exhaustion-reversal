"""Spec section 30: assembles pump_event_research_dataset.parquet - one row
per event, joining EventRecord's own fields with research/forward_study.py's
forward-path output and (when available) research/cross_exchange.py's
underlying-event linking. Pure assembly - no new computation of its own
beyond flattening/joining already-computed pieces.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..backtest.simulator import EventRecord


def _event_to_flat_row(event: EventRecord) -> dict:
    """Flattens one EventRecord into a dict for a wide DataFrame row.
    Dict/list-typed research fields (exhaustion_component_first_true_bar
    etc.) are kept as native Python objects - parquet/pyarrow round-trips
    dict/list columns fine (verified separately against a real write+read),
    not JSON-stringified."""
    return {
        "event_id": event.event_id, "exchange": event.exchange, "symbol": event.symbol,
        "pump_start_ts": event.pump_start_ts, "pump_detected_ts": event.pump_detected_ts, "pump_peak_ts": event.pump_peak_ts,
        "pump_score": event.pump_score, "pump_class": event.pump_class, "pump_class_confidence": event.pump_class_confidence,
        "pump_class_reasons": event.pump_class_reasons, "pump_class_data_confidence_notes": event.pump_class_data_confidence_notes,
        "data_confidence": event.data_confidence,
        "volume_zscore": event.volume_zscore, "oi_zscore": event.oi_zscore, "funding_percentile": event.funding_percentile,
        "short_liq_zscore": event.short_liq_zscore, "btc_return_concurrent": event.btc_return_concurrent, "news_flag": event.news_flag,
        "final_state": event.final_state, "exhaustion_score_peak": event.exhaustion_score_peak,
        "exhaustion_component_first_true_bar": event.exhaustion_component_first_true_bar,
        "exhaustion_component_contribution_at_peak": event.exhaustion_component_contribution_at_peak,
        "oi_drop_pct_from_peak_at_peak": event.oi_drop_pct_from_peak_at_peak,
        "volume_decay_ratio_at_peak": event.volume_decay_ratio_at_peak,
        "cvd_divergence_magnitude_at_peak": event.cvd_divergence_magnitude_at_peak,
        "structure_break_bar_local_idx": event.structure_break_bar_local_idx,
        "dynamic_invalidation_bar_local_idx": event.dynamic_invalidation_bar_local_idx,
        "climax_local_idx": event.climax_local_idx, "attempt_count": event.attempt_count,
        "invalidation_reason": event.invalidation_reason,
        "subsequent_return_15m": event.subsequent_return_15m, "subsequent_return_1h": event.subsequent_return_1h,
        "subsequent_return_4h": event.subsequent_return_4h, "subsequent_return_24h": event.subsequent_return_24h,
        "max_adverse_after_detection": event.max_adverse_after_detection,
        "max_favorable_after_detection": event.max_favorable_after_detection,
        "has_trade": event.trade is not None,
    }


def build_event_level_frame(events: list[EventRecord]) -> pd.DataFrame:
    """One row per EventRecord (single-exchange level, before any
    cross-exchange linking) - the base frame everything else joins onto."""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([_event_to_flat_row(e) for e in events])


def join_forward_paths(event_frame: pd.DataFrame, forward_paths_df: pd.DataFrame) -> pd.DataFrame:
    """Left-joins forward_study.py's per-event forward-path columns onto
    event_frame by event_id. Any column name that already exists on
    event_frame (none expected in practice, since forward_study's columns
    use forward_*/eventual_*/drawdown_*/right_censored names) is dropped
    from the forward side rather than silently overwritten - flagged here
    since it would otherwise be a silent bug, not because it's expected to
    happen."""
    if event_frame.empty or forward_paths_df.empty:
        return event_frame
    overlap = (set(event_frame.columns) & set(forward_paths_df.columns)) - {"event_id"}
    fp = forward_paths_df.drop(columns=list(overlap)) if overlap else forward_paths_df
    return event_frame.merge(fp, on="event_id", how="left")


def join_underlying_event_linking(event_frame: pd.DataFrame, linked_df: pd.DataFrame) -> pd.DataFrame:
    """Left-joins research/cross_exchange.py's underlying_event_id onto
    event_frame, matching on whichever of binance_event_id/bybit_event_id
    equals this row's event_id. Optional - callers running a single-exchange
    diagnostic pass can skip this step entirely."""
    if event_frame.empty or linked_df.empty:
        out = event_frame.copy()
        out["underlying_event_id"] = pd.NA
        return out
    id_map: dict[str, str] = {}
    for _, row in linked_df.iterrows():
        binance_id = row.get("binance_event_id")
        bybit_id = row.get("bybit_event_id")
        if pd.notna(binance_id):
            id_map[binance_id] = row["underlying_event_id"]
        if pd.notna(bybit_id):
            id_map[bybit_id] = row["underlying_event_id"]
    out = event_frame.copy()
    out["underlying_event_id"] = out["event_id"].map(id_map)
    return out


def save_research_dataset(dataset: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    return out_path
