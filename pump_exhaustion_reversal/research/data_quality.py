"""§7/§45/§46: Feature Availability Report and per-event data-confidence
classification. Pure read from data/storage.py plus already-computed
EventRecord fields - no simulation, no trading logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord
from ..data import storage

SPOT_DATA_UNAVAILABLE = (
    "SPOT_DATA_UNAVAILABLE - Stage 1 is a linear-perp-only project; no spot kline/trades/CVD "
    "connector exists on either exchange (verified: zero matches for any spot fetch path repo-wide)."
)


def compute_symbol_level_availability(exchange: str, timeframe: str = "5m") -> pd.DataFrame:
    """One row per symbol with cached OHLCV on `exchange`: whether OI/funding
    files exist at all, and (read directly from the parquet, not assumed)
    whether taker_buy_base_volume / open_interest_value are ever non-null -
    both columns always exist in the schema on every exchange for parity,
    but are always-null placeholders on Bybit specifically."""
    ohlcv_symbols = storage.list_symbols(exchange, "ohlcv", timeframe=timeframe)
    oi_symbols = set(storage.list_symbols(exchange, "oi", timeframe=timeframe))
    funding_symbols = set(storage.list_symbols(exchange, "funding"))

    rows = []
    for symbol in ohlcv_symbols:
        has_oi_file = symbol in oi_symbols
        has_funding_file = symbol in funding_symbols
        ohlcv = storage.read_range(exchange, "ohlcv", symbol, timeframe=timeframe)
        has_taker_buy_volume = bool(
            "taker_buy_base_volume" in ohlcv.columns and ohlcv["taker_buy_base_volume"].notna().any()
        )
        has_oi_notional = False
        if has_oi_file:
            oi_df = storage.read_range(exchange, "oi", symbol, timeframe=timeframe)
            has_oi_notional = bool("open_interest_value" in oi_df.columns and oi_df["open_interest_value"].notna().any())
        rows.append(
            {
                "exchange": exchange, "symbol": symbol, "has_ohlcv": True,
                "has_oi_file": has_oi_file, "has_funding_file": has_funding_file,
                "has_taker_buy_volume_cvd": has_taker_buy_volume, "has_oi_notional": has_oi_notional,
            }
        )
    return pd.DataFrame(rows)


def summarize_availability(availability_df: pd.DataFrame) -> dict:
    n = len(availability_df)
    if n == 0:
        return {
            "n_symbols": 0, "pct_with_oi_file": np.nan, "pct_with_funding_file": np.nan,
            "pct_with_taker_buy_volume_cvd": np.nan, "pct_with_oi_notional": np.nan,
        }
    return {
        "n_symbols": n,
        "pct_with_oi_file": float(availability_df["has_oi_file"].mean()),
        "pct_with_funding_file": float(availability_df["has_funding_file"].mean()),
        "pct_with_taker_buy_volume_cvd": float(availability_df["has_taker_buy_volume_cvd"].mean()),
        "pct_with_oi_notional": float(availability_df["has_oi_notional"].mean()),
    }


def classify_event_data_confidence(event: EventRecord) -> str:
    """HIGH/MEDIUM/LOW per spec section 45, read from an already-computed
    EventRecord's own detection-time signal values (not re-reading raw
    files): if a component's underlying series genuinely had data at this
    event's detection bar, its z-score/percentile/magnitude field is a real
    number, not NaN - that's the operational definition of "had the data"
    used here, distinct from (and finer-grained than) symbol-level archive
    coverage from compute_symbol_level_availability."""
    checks = (
        not np.isnan(event.oi_zscore),
        not np.isnan(event.short_liq_zscore),
        not np.isnan(event.funding_percentile),
        not np.isnan(event.cvd_divergence_magnitude_at_peak),
    )
    present = sum(checks)
    if present == len(checks):
        return "HIGH"
    if present == 0:
        return "LOW"
    return "MEDIUM"


def classify_events_data_confidence(events: list[EventRecord]) -> pd.Series:
    return pd.Series([classify_event_data_confidence(e) for e in events], name="research_data_confidence")


def render_data_quality_report(
    exchange_summaries: dict[str, dict], exchange_event_confidence: dict[str, pd.Series]
) -> str:
    """exchange_summaries: {exchange: summarize_availability() output}.
    exchange_event_confidence: {exchange: classify_events_data_confidence() output}."""
    lines = ["# Data Quality Report", "", "## Feature availability by exchange (symbol-level, read from disk)", ""]
    exchanges = list(exchange_summaries)
    lines.append("| Feature | " + " | ".join(exchanges) + " |")
    lines.append("|---|" + "---|" * len(exchanges))
    lines.append("| Symbols with cached OHLCV | " + " | ".join(str(exchange_summaries[e]["n_symbols"]) for e in exchanges) + " |")
    for label, key in (
        ("OI file present", "pct_with_oi_file"), ("Funding file present", "pct_with_funding_file"),
        ("Real taker-buy-volume CVD", "pct_with_taker_buy_volume_cvd"), ("OI notional (USD) field", "pct_with_oi_notional"),
    ):
        cells = []
        for e in exchanges:
            v = exchange_summaries[e][key]
            cells.append("n/a" if np.isnan(v) else f"{v:.0%}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(f"| Spot volume / CVD | {SPOT_DATA_UNAVAILABLE} " + "| " * (len(exchanges) - 1) + "|")
    lines.append("")
    lines.append("## Per-event research data confidence (HIGH/MEDIUM/LOW, spec section 45)")
    lines.append("")
    lines.append("HIGH = OI z-score, liquidation-proxy z-score, funding percentile, and CVD divergence magnitude were ALL real (non-NaN) at detection. LOW = none were. MEDIUM = some.")
    lines.append("")
    for exchange, confidence in exchange_event_confidence.items():
        if len(confidence) == 0:
            lines.append(f"- {exchange}: no events")
            continue
        counts = confidence.value_counts()
        total = len(confidence)
        parts = ", ".join(f"{k}={v} ({v/total:.0%})" for k, v in counts.items())
        lines.append(f"- {exchange} (n={total}): {parts}")
    return "\n".join(lines)
