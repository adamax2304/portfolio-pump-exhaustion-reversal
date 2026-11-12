"""Phase 3A.1 point 12 (user's #3 reprioritized hypothesis): does
NON-CONFIRMATION across exchanges - one venue still making new highs while
the other isn't, or one showing much stronger volume/OI participation than
the other for the same underlying pump - raise the directional reversal
probability? Phase 3A deliberately deferred this (its point 21) because it
needs a cross-exchange linked-event join that most single-exchange snapshots
won't have; research.cross_exchange.link_cross_exchange_events already does
that join at the EVENT level (see its docstring for the greedy nearest-match
algorithm) - this module extends it to the SNAPSHOT level.

Snapshots whose event has no cross-exchange match get NaN/False throughout -
Phase 3A's cross_exchange_audit already found most events are solo (the
~402-Binance-vs-~166-Bybit symbol universe mismatch), so this is the expected
majority case, not a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CROSS_EXCHANGE_OUTPUT_COLUMNS = (
    "other_exchange_present",
    "one_exchange_new_high_only",
    "return_divergence",
    "volume_divergence",
    "oi_divergence",
    "leader_lagger_state",
)

_COMPARE_COLUMNS = ("is_new_high", "current_return_from_start", "volume_current_over_climax", "oi_change_60m")


def _empty_output(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["other_exchange_present"] = False
    out["one_exchange_new_high_only"] = None  # object dtype (True/False/None) - a plain float NaN column can't hold booleans without an upcast pandas now refuses on assignment
    out["return_divergence"] = np.nan
    out["volume_divergence"] = np.nan
    out["oi_divergence"] = np.nan
    out["leader_lagger_state"] = None
    return out


def compute_cross_exchange_snapshot_features(snapshots_df: pd.DataFrame, linked_df: pd.DataFrame) -> pd.DataFrame:
    """For snapshots belonging to an event with a matched cross-exchange
    counterpart (per `link_cross_exchange_events`'s `both-present` rows),
    attaches the OTHER exchange's most-recently-known state as of THIS
    snapshot's own timestamp (`pd.merge_asof(..., direction="backward")` -
    never looks at the other exchange's future bars) and derives
    non-confirmation signals: `one_exchange_new_high_only` (this exchange's
    is_new_high disagrees with the other's), `return_divergence` /
    `volume_divergence` / `oi_divergence` (this exchange's value minus the
    other's, at the same wall-clock moment), and `leader_lagger_state`
    ("leader"/"lagger"/"simultaneous", from the linked event's own
    detection-time ordering - constant per event, not time-varying).

    Returns a DataFrame aligned to `snapshots_df`'s own index (same length,
    same row order) so the caller can `pd.concat([snapshots_df, result],
    axis=1)` directly."""
    out = _empty_output(snapshots_df.index)
    if "event_id" not in snapshots_df.columns or linked_df.empty:
        return out
    both = linked_df[linked_df["binance_present"] & linked_df["bybit_present"]]
    if both.empty:
        return out

    available_cols = [c for c in _COMPARE_COLUMNS if c in snapshots_df.columns]
    if not available_cols or "timestamp" not in snapshots_df.columns:
        return out

    by_event = {eid: g for eid, g in snapshots_df[["event_id", "timestamp"] + available_cols].groupby("event_id")}

    for _, link_row in both.iterrows():
        b_id, y_id = link_row["binance_event_id"], link_row["bybit_event_id"]
        b_snap, y_snap = by_event.get(b_id), by_event.get(y_id)
        if b_snap is None or y_snap is None:
            continue
        b_sorted = b_snap.sort_values("timestamp").reset_index()
        y_sorted = y_snap.sort_values("timestamp").reset_index()
        other_cols = ["timestamp"] + available_cols
        b_aug = pd.merge_asof(b_sorted, y_sorted[other_cols], on="timestamp", direction="backward", suffixes=("", "_other"))
        y_aug = pd.merge_asof(y_sorted, b_sorted[other_cols], on="timestamp", direction="backward", suffixes=("", "_other"))

        leader = link_row["lead_exchange"]
        for aug, own_exchange in ((b_aug, "binance"), (y_aug, "bybit")):
            state = "leader" if leader == own_exchange else ("lagger" if leader is not None else "simultaneous")
            idx = aug["index"].to_numpy()
            out.loc[idx, "other_exchange_present"] = True
            out.loc[idx, "leader_lagger_state"] = state

            if "is_new_high" in available_cols:
                mask = aug["is_new_high_other"].notna().to_numpy()
                out.loc[idx[mask], "one_exchange_new_high_only"] = (aug["is_new_high"] != aug["is_new_high_other"])[mask].to_numpy()
            if "current_return_from_start" in available_cols:
                mask = aug["current_return_from_start_other"].notna().to_numpy()
                out.loc[idx[mask], "return_divergence"] = (aug["current_return_from_start"] - aug["current_return_from_start_other"])[mask].to_numpy()
            if "volume_current_over_climax" in available_cols:
                mask = aug["volume_current_over_climax_other"].notna().to_numpy()
                out.loc[idx[mask], "volume_divergence"] = (aug["volume_current_over_climax"] - aug["volume_current_over_climax_other"])[mask].to_numpy()
            if "oi_change_60m" in available_cols:
                mask = aug["oi_change_60m_other"].notna().to_numpy()
                out.loc[idx[mask], "oi_divergence"] = (aug["oi_change_60m"] - aug["oi_change_60m_other"])[mask].to_numpy()

    return out
