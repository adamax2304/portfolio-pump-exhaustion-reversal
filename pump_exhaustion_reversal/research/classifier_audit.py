"""Spec sections 5/6: Pump Classifier Audit. Quantifies (not just re-asserts)
why SPOT_ORGANIC dominates - primarily testing the OI-retention-window
hypothesis found during Stage 1.5 planning (Binance's real OI depth via REST
is ~1.7 days regardless of requested range; Bybit's is ~30 days) against
actual classification outcomes, plus a per-class confidence/reasons
breakdown. Pure read of already-computed EventRecord fields - no
re-classification, no trading logic.
"""

from __future__ import annotations

import pandas as pd

from ..backtest.simulator import EventRecord


def class_distribution(events: list[EventRecord]) -> pd.DataFrame:
    """Pump class counts + share, plus each class's own share of
    confidence == 'low' events (SPOT_ORGANIC's residual-fallback marker per
    pump_classifier.py, but tracked for every class since any of them can in
    principle be low-confidence)."""
    if not events:
        return pd.DataFrame(columns=["pump_class", "n", "share", "pct_low_confidence"])
    df = pd.DataFrame({"pump_class": [e.pump_class for e in events], "confidence": [e.pump_class_confidence for e in events]})
    total = len(df)
    out = df.groupby("pump_class").size().rename("n").reset_index()
    out["share"] = out["n"] / total
    low_conf_share = df.groupby("pump_class")["confidence"].apply(lambda s: (s == "low").mean())
    out["pct_low_confidence"] = out["pump_class"].map(low_conf_share)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def oi_availability_vs_class(events: list[EventRecord]) -> pd.DataFrame:
    """The core hypothesis test: does the pump_class distribution differ
    between events where OI data was actually available at detection
    (oi_zscore not NaN) vs not? SHORT_SQUEEZE/LEVERAGED_LONG structurally
    require oi_pct_change/oi_zscore or the OI-drop-gated liquidation proxy to
    have any chance of firing (pump_classifier.py's oi_up_strong /
    short_liq_zscore checks) - if those two classes cluster almost
    exclusively in the "OI available" group, that's direct evidence the
    OI-retention-window limitation (not the classifier's decision logic)
    explains most of the SPOT_ORGANIC fallback rate."""
    if not events:
        return pd.DataFrame(columns=["oi_available", "pump_class", "n", "share_within_group"])
    df = pd.DataFrame(
        {"pump_class": [e.pump_class for e in events], "oi_available": [not pd.isna(e.oi_zscore) for e in events]}
    )
    out = df.groupby(["oi_available", "pump_class"]).size().rename("n").reset_index()
    group_totals = out.groupby("oi_available")["n"].transform("sum")
    out["share_within_group"] = out["n"] / group_totals
    return out.sort_values(["oi_available", "n"], ascending=[False, False]).reset_index(drop=True)


def reasons_and_notes_sample(events: list[EventRecord], pump_class: str, limit: int = 15) -> list[dict]:
    """Raw reasons/data_confidence_notes for a sample of events of one class -
    the "show your work" audit trail spec section 5 asks for per event."""
    matching = [e for e in events if e.pump_class == pump_class][:limit]
    return [
        {
            "event_id": e.event_id, "symbol": e.symbol, "exchange": e.exchange,
            "confidence": e.pump_class_confidence, "reasons": e.pump_class_reasons,
            "data_confidence_notes": e.pump_class_data_confidence_notes,
        }
        for e in matching
    ]


def fallback_bias_check(events: list[EventRecord]) -> dict:
    """Spec section 6's specific ask: does missing OI/liquidation signal
    correlate with landing on SPOT_ORGANIC specifically? Reports
    SPOT_ORGANIC's share within "no derivatives signal at all" events vs
    within "has at least one derivatives signal"."""
    if not events:
        return {}

    def _has_derivatives_signal(e: EventRecord) -> bool:
        return (not pd.isna(e.oi_zscore)) or (not pd.isna(e.short_liq_zscore))

    def _spot_share(group: list[EventRecord]) -> float:
        return sum(1 for e in group if e.pump_class == "SPOT_ORGANIC") / len(group) if group else float("nan")

    no_signal = [e for e in events if not _has_derivatives_signal(e)]
    has_signal = [e for e in events if _has_derivatives_signal(e)]
    return {
        "n_no_oi_or_liq_signal": len(no_signal), "spot_organic_share_no_signal": _spot_share(no_signal),
        "n_has_oi_or_liq_signal": len(has_signal), "spot_organic_share_has_signal": _spot_share(has_signal),
    }


def render_classifier_audit_report(exchange_events: dict[str, list[EventRecord]]) -> str:
    lines = ["# Pump Classifier Audit", ""]
    for exchange, events in exchange_events.items():
        lines.append(f"## {exchange} (n={len(events)})")
        lines.append("")
        if not events:
            lines.append("No events.")
            lines.append("")
            continue
        dist = class_distribution(events)
        lines.append("### Class distribution")
        lines.append("")
        lines.append("| Class | N | Share | % low confidence |")
        lines.append("|---|---|---|---|")
        for _, row in dist.iterrows():
            lines.append(f"| {row['pump_class']} | {row['n']} | {row['share']:.1%} | {row['pct_low_confidence']:.1%} |")
        lines.append("")
        lines.append("### OI availability vs class (retention-window hypothesis)")
        lines.append("")
        oi_vs_class = oi_availability_vs_class(events)
        lines.append("| OI available at detection | Class | N | Share within group |")
        lines.append("|---|---|---|---|")
        for _, row in oi_vs_class.iterrows():
            lines.append(f"| {row['oi_available']} | {row['pump_class']} | {row['n']} | {row['share_within_group']:.1%} |")
        lines.append("")
        bias = fallback_bias_check(events)
        lines.append("### Fallback bias check (spec section 6)")
        lines.append("")
        lines.append(
            f"- Events with NO OI/liquidation signal at all (n={bias['n_no_oi_or_liq_signal']}): "
            f"{bias['spot_organic_share_no_signal']:.1%} classified SPOT_ORGANIC"
        )
        lines.append(
            f"- Events with at least one OI/liquidation signal (n={bias['n_has_oi_or_liq_signal']}): "
            f"{bias['spot_organic_share_has_signal']:.1%} classified SPOT_ORGANIC"
        )
        lines.append("")
    lines.append("## Classifier design note (verified from code, not inferred)")
    lines.append("")
    lines.append(
        "`classify_pump()` (strategy/pump_classifier.py) already marks SPOT_ORGANIC as a residual "
        "fallback with confidence='low' and an explicit note ('SPOT_ORGANIC is a residual fallback in "
        "Stage 1, not a positive detection') whenever no other signature matches - it does not silently "
        "claim organic spot buying. The tables above measure *how often* the classifier is forced into "
        "that fallback by missing derivatives data, not whether it mislabels data it does have."
    )
    return "\n".join(lines)
