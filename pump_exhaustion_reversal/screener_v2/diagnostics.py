"""Phase 3A deliverables 2-3: snapshot_data_quality.md and
label_distribution.md - pure computation (reports.py does the rendering).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import validation

IDENTITY_COLUMNS = frozenset(
    {
        "event_id", "exchange", "symbol", "timestamp", "pump_start_timestamp", "pump_detected_timestamp",
        "pump_age_minutes", "current_price", "event_high_so_far", "event_low_so_far", "climax_price", "snapshot_abs_idx",
        "quote_volume_at_pump_start",
    }
)
LABEL_FAMILY_NAMES = ("primary", "fast", "deep")


def _is_label_column(col: str) -> bool:
    if col == "peak_hazard" or col.startswith("peak_hazard_"):
        return True
    return any(col.startswith(f"{family}_") for family in LABEL_FAMILY_NAMES)


def compute_snapshot_data_quality(snapshots_df: pd.DataFrame) -> dict:
    """Per-column NaN rate (excluding identity and label columns, which are
    reported separately in label_distribution.md, not here) + overall
    event/symbol/exchange coverage counts."""
    if snapshots_df.empty:
        return {"n_snapshots": 0, "n_events": 0, "n_symbols": 0}
    feature_cols = [c for c in snapshots_df.columns if c not in IDENTITY_COLUMNS and not _is_label_column(c)]
    nan_rates = {c: float(snapshots_df[c].isna().mean()) for c in feature_cols}
    return {
        "n_snapshots": len(snapshots_df),
        "n_events": snapshots_df["event_id"].nunique(),
        "n_symbols": snapshots_df["symbol"].nunique(),
        "n_exchanges": snapshots_df["exchange"].nunique() if "exchange" in snapshots_df.columns else 0,
        "snapshots_per_event": snapshots_df.groupby("event_id").size().describe().to_dict(),
        "feature_nan_rates": dict(sorted(nan_rates.items(), key=lambda kv: -kv[1])),
        "fully_missing_features": [c for c, r in nan_rates.items() if r >= 0.999],
    }


def compute_label_distribution(snapshots_df: pd.DataFrame, label_families: tuple[str, ...] = ("primary", "fast", "deep")) -> pd.DataFrame:
    """Event-weighted AND raw snapshot-weighted outcome rates per label
    family - the gap between the two IS the point-3 long-event-domination
    check made visible (a family whose snapshot-weighted rate diverges a lot
    from its event-weighted rate is being pulled around by a few long
    events)."""
    rows = []
    weights = validation.event_weights(snapshots_df) if "event_id" in snapshots_df.columns else pd.Series(1.0, index=snapshots_df.index)
    for family in label_families:
        col = f"{family}_outcome"
        if col not in snapshots_df.columns:
            continue
        valid = snapshots_df[col].notna()
        n_snapshots = int(valid.sum())
        n_events = snapshots_df.loc[valid, "event_id"].nunique() if "event_id" in snapshots_df.columns else n_snapshots
        row = {"label_family": family, "n_snapshots": n_snapshots, "n_events": n_events}
        for outcome in ("REVERSAL", "CONTINUATION", "NEUTRAL"):
            indicator = (snapshots_df[col] == outcome).astype(float)
            row[f"snapshot_weighted_p_{outcome.lower()}"] = float(indicator[valid].mean()) if n_snapshots else np.nan
            w = weights[valid]
            row[f"event_weighted_p_{outcome.lower()}"] = float((w * indicator[valid]).sum() / w.sum()) if w.sum() else np.nan
        rows.append(row)

    if "peak_hazard" in snapshots_df.columns:
        valid = snapshots_df["peak_hazard"].notna()
        n_snapshots = int(valid.sum())
        w = weights[valid]
        indicator = snapshots_df["peak_hazard"].astype(float)
        rows.append(
            {
                "label_family": "peak_hazard", "n_snapshots": n_snapshots,
                "n_events": snapshots_df.loc[valid, "event_id"].nunique() if "event_id" in snapshots_df.columns else n_snapshots,
                "snapshot_weighted_p_reversal": float(indicator[valid].mean()) if n_snapshots else np.nan,
                "event_weighted_p_reversal": float((w * indicator[valid]).sum() / w.sum()) if w.sum() else np.nan,
                "snapshot_weighted_p_continuation": np.nan, "event_weighted_p_continuation": np.nan,
                "snapshot_weighted_p_neutral": np.nan, "event_weighted_p_neutral": np.nan,
            }
        )
    return pd.DataFrame(rows)
