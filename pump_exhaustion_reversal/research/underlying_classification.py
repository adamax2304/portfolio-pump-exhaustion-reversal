"""Stage 2A Part 3: underlying cross-exchange pump classification - combines
each exchange's OWN classify_pump() verdict for the SAME real-world pump
(linked via research/cross_exchange.py's link_cross_exchange_events) into one
"underlying event" classification using whichever exchange actually had the
data, instead of treating "Binance says SPOT_ORGANIC" as a standalone
statement when Binance simply had no OI data available for that event.

SPOT_ORGANIC is pump_classifier.py's own documented residual fallback (see
its module docstring), not a positive detection - so a positive
(non-SPOT_ORGANIC) classification from EITHER exchange should win over a
SPOT_ORGANIC fallback from the other. A Binance-only event with no
derivatives signal at all becomes UNKNOWN_INSUFFICIENT_DATA, not
SPOT_ORGANIC - the point of this whole module.
"""

from __future__ import annotations

import pandas as pd

from ..backtest.simulator import EventRecord
from ..config import AppConfig

_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def classify_short_squeeze_state(event: EventRecord, cfg: AppConfig) -> str:
    """Phase 2C point 9: splits SHORT_SQUEEZE into ACTIVE (forced buying
    plausibly still ongoing) vs DEPLETED (OI has meaningfully unwound from its
    peak and the squeeze never needed to extend via a genuine new high) -
    the hypothesis being that ACTIVE squeezes are NOT a good short candidate
    (continuation risk, per Phase 2B's finding that SHORT_SQUEEZE as a whole
    had the worst first-passage/net-expectancy of any pump class) while
    DEPLETED ones might be. Uses only fields already captured on EventRecord
    (oi_drop_pct_from_peak_at_peak, attempt_invalidation_reason_codes) and
    reuses cfg.oi.min_oi_drop_pct_after_peak - the SAME threshold
    features/oi.py's own oi_rollover flag uses - rather than inventing a new
    magic number. Only meaningful for events the underlying classifier
    already called SHORT_SQUEEZE; calling this on any other class is the
    caller's mistake, not guarded against here."""
    # attempt_invalidation_reason_codes round-trips through parquet as a
    # numpy array, not a Python list - `x or []` raises "truth value of an
    # array is ambiguous" for any array with more than one element (the same
    # class of parquet round-trip gotcha documented elsewhere in research/*;
    # `list(...)` sidesteps it regardless of which type the field currently is).
    retried_on_new_high = "new_pump_high" in list(event.attempt_invalidation_reason_codes)
    oi_drop = event.oi_drop_pct_from_peak_at_peak
    oi_meaningfully_dropped = pd.notna(oi_drop) and oi_drop >= cfg.oi.min_oi_drop_pct_after_peak
    if retried_on_new_high or not oi_meaningfully_dropped:
        return "SHORT_SQUEEZE_ACTIVE"
    return "SHORT_SQUEEZE_DEPLETED"


def _has_derivatives_signal(event: EventRecord) -> bool:
    return (not pd.isna(event.oi_zscore)) or (not pd.isna(event.short_liq_zscore))


def classify_underlying_event(binance_event: EventRecord | None, bybit_event: EventRecord | None) -> dict:
    """Returns {"underlying_pump_class": str, "confidence": str, "source_exchange": str | None}."""
    candidates = [e for e in (binance_event, bybit_event) if e is not None]
    if not candidates:
        return {"underlying_pump_class": "UNKNOWN", "confidence": "low", "source_exchange": None}

    positive = [e for e in candidates if e.pump_class != "SPOT_ORGANIC"]
    if positive:
        best = max(positive, key=lambda e: _CONFIDENCE_RANK.get(e.pump_class_confidence, 0))
        return {"underlying_pump_class": best.pump_class, "confidence": best.pump_class_confidence, "source_exchange": best.exchange}

    if any(_has_derivatives_signal(e) for e in candidates):
        # every available exchange fell through to SPOT_ORGANIC despite at
        # least one having real derivatives data to check - a genuine (if
        # still low-confidence) residual verdict, not a data-availability gap.
        source = next(e for e in candidates if _has_derivatives_signal(e))
        return {"underlying_pump_class": "SPOT_ORGANIC", "confidence": "low", "source_exchange": source.exchange}

    return {"underlying_pump_class": "UNKNOWN_INSUFFICIENT_DATA", "confidence": "low", "source_exchange": None}


def classify_underlying_events(
    linked_df: pd.DataFrame, binance_events: list[EventRecord], bybit_events: list[EventRecord]
) -> pd.DataFrame:
    """linked_df: research/cross_exchange.py's link_cross_exchange_events()
    output (one row per underlying event, binance_event_id/bybit_event_id
    nullable). Returns linked_df with three new columns appended:
    underlying_pump_class, underlying_confidence, underlying_source_exchange."""
    binance_by_id = {e.event_id: e for e in binance_events}
    bybit_by_id = {e.event_id: e for e in bybit_events}
    rows = []
    for _, row in linked_df.iterrows():
        b = binance_by_id.get(row.get("binance_event_id")) if pd.notna(row.get("binance_event_id")) else None
        y = bybit_by_id.get(row.get("bybit_event_id")) if pd.notna(row.get("bybit_event_id")) else None
        rows.append(classify_underlying_event(b, y))
    result = linked_df.copy()
    result["underlying_pump_class"] = [r["underlying_pump_class"] for r in rows]
    result["underlying_confidence"] = [r["confidence"] for r in rows]
    result["underlying_source_exchange"] = [r["source_exchange"] for r in rows]
    return result


def underlying_class_distribution(classified_df: pd.DataFrame) -> pd.DataFrame:
    """Class counts/shares on the underlying-event level (spec section 6's
    A/B/C/D/E/UNKNOWN breakdown, but deduplicated across exchanges - the
    same fix classifier_audit.class_distribution applies per-exchange)."""
    if classified_df.empty:
        return pd.DataFrame(columns=["underlying_pump_class", "n", "share"])
    total = len(classified_df)
    out = classified_df.groupby("underlying_pump_class").size().rename("n").reset_index()
    out["share"] = out["n"] / total
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def render_underlying_classification_report(classified_df: pd.DataFrame, distribution: pd.DataFrame) -> str:
    lines = ["# Underlying Cross-Exchange Classification", "", f"Total underlying events: {len(classified_df)}", ""]
    lines.append("## Class distribution (deduplicated across exchanges - best available data wins)")
    lines.append("")
    lines.append("| Class | N | Share |")
    lines.append("|---|---|---|")
    for _, row in distribution.iterrows():
        lines.append(f"| {row['underlying_pump_class']} | {row['n']} | {row['share']:.1%} |")
    lines.append("")
    if not classified_df.empty:
        source_counts = classified_df["underlying_source_exchange"].value_counts(dropna=False)
        lines.append("## Which exchange's classification won")
        lines.append("")
        lines.append(", ".join(f"{k if pd.notna(k) else 'none (insufficient data)'}={v}" for k, v in source_counts.items()))
    return "\n".join(lines)
