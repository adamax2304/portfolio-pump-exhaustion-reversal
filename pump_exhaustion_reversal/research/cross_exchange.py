"""Stage 1.5 research/diagnostic phase: cross-exchange event linkage. Pure
read of already-computed EventRecord lists from the Binance and Bybit event
studies - no re-detection, no trading logic, nothing here is imported by
entry_engine/risk_engine/exit_engine/simulator.

Motivation (verified from code, not inferred): `backtest/reports.py` and the
README already flag that `PumpClass.LIQUIDITY_MANIPULATION` "has no
cross-exchange confirmation - single-venue data only" because Stage 1 detects
and classifies each exchange's events independently. This module is the
empirical follow-up: given the two exchanges' independently-produced
EventRecord lists, how often does the same real-world pump actually get
detected on both venues at all, how far apart in time, and does the
classifier land on the same pump_class both times? The loaded universe is
~402 Binance symbols vs ~166 Bybit symbols, and even for a symbol listed on
both, detections routinely don't line up in time - so most events are
expected to be "solo" (only one exchange has a matching detection at all).
That is treated as the normal case throughout, not an error condition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord

LINKED_COLUMNS = [
    "underlying_event_id",
    "symbol",
    "binance_event_id",
    "bybit_event_id",
    "binance_present",
    "bybit_present",
    "lead_exchange",
    "detection_time_delta_minutes",
    "binance_return",
    "bybit_return",
    "binance_pump_class",
    "bybit_pump_class",
]


def link_cross_exchange_events(
    binance_events: list[EventRecord], bybit_events: list[EventRecord], window_minutes: int = 60
) -> pd.DataFrame:
    """Link Binance and Bybit EventRecords that represent the SAME
    real-world pump into one "underlying event" row each, one row per
    underlying event (matched pair or solo leftover).

    Matching rule: same `symbol` (exact string match on the already-
    normalized uppercase symbol, e.g. "BTCUSDT" - no fuzzy matching) AND
    `pump_detected_ts` within `window_minutes` minutes of each other
    (inclusive) on both exchanges.

    Matching algorithm (deliberate design decision, not the only possible
    one - documented here because it matters for how to read the output):
    greedy global-nearest-match, not "first Binance event takes whichever
    Bybit event is closest to IT". Concretely:
      1. Build every candidate (binance_event, bybit_event) pair with equal
         symbol and abs(time delta) <= window_minutes.
      2. Sort ALL candidate pairs across the whole run by absolute time
         delta, ascending (ties broken by ascending Binance-list index then
         Bybit-list index, for determinism - real timestamps essentially
         never tie exactly).
      3. Walk the sorted list and greedily claim pairs: a pair is matched
         only if neither its Binance event nor its Bybit event has already
         been claimed by an earlier (strictly closer, or equal-and-earlier-
         in-tiebreak) pair in this same walk.
    This means the single closest pair ANYWHERE in the whole candidate set
    wins first, even if that leaves a "locally closest" pair elsewhere
    unmatched because one of its two events got claimed by something even
    closer. Any event (either exchange) never claimed by this process
    becomes its own solo underlying-event row.

    Output: one row per underlying event with columns
    `LINKED_COLUMNS` (underlying_event_id, symbol, binance_event_id,
    bybit_event_id, binance_present, bybit_present, lead_exchange,
    detection_time_delta_minutes, binance_return, bybit_return,
    binance_pump_class, bybit_pump_class). `lead_exchange` is the exchange
    whose `pump_detected_ts` is earlier ("binance"/"bybit"), or None when
    only one side is present or both sides tie exactly.
    `detection_time_delta_minutes` is signed
    `(bybit_detected_ts - binance_detected_ts) / 60000.0` - positive means
    Bybit detected later than Binance - and only populated when both sides
    are present. `binance_return`/`bybit_return` pass through each side's
    own `subsequent_return_1h` verbatim (including NaN if that field itself
    was NaN on the EventRecord); they are NaN when that side is absent.
    """
    window_ms = window_minutes * 60_000

    candidates: list[tuple[int, int, int, int]] = []  # (abs_delta_ms, binance_idx, bybit_idx, signed_delta_ms)
    for bi, be in enumerate(binance_events):
        for yi, ye in enumerate(bybit_events):
            if be.symbol != ye.symbol:
                continue
            delta_ms = ye.pump_detected_ts - be.pump_detected_ts
            if abs(delta_ms) <= window_ms:
                candidates.append((abs(delta_ms), bi, yi, delta_ms))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    claimed_binance: set[int] = set()
    claimed_bybit: set[int] = set()
    matched_pairs: list[tuple[int, int, int]] = []  # (binance_idx, bybit_idx, signed_delta_ms)
    for _abs_delta, bi, yi, delta_ms in candidates:
        if bi in claimed_binance or yi in claimed_bybit:
            continue
        claimed_binance.add(bi)
        claimed_bybit.add(yi)
        matched_pairs.append((bi, yi, delta_ms))

    rows: list[dict] = []
    for bi, yi, delta_ms in matched_pairs:
        be, ye = binance_events[bi], bybit_events[yi]
        if be.pump_detected_ts < ye.pump_detected_ts:
            lead_exchange = "binance"
        elif ye.pump_detected_ts < be.pump_detected_ts:
            lead_exchange = "bybit"
        else:
            lead_exchange = None
        rows.append(
            {
                "underlying_event_id": f"underlying_{be.symbol}_{min(be.pump_detected_ts, ye.pump_detected_ts)}",
                "symbol": be.symbol,
                "binance_event_id": be.event_id,
                "bybit_event_id": ye.event_id,
                "binance_present": True,
                "bybit_present": True,
                "lead_exchange": lead_exchange,
                "detection_time_delta_minutes": delta_ms / 60_000.0,
                "binance_return": be.subsequent_return_1h,
                "bybit_return": ye.subsequent_return_1h,
                "binance_pump_class": be.pump_class,
                "bybit_pump_class": ye.pump_class,
            }
        )

    for bi, be in enumerate(binance_events):
        if bi in claimed_binance:
            continue
        rows.append(
            {
                "underlying_event_id": f"underlying_{be.symbol}_{be.pump_detected_ts}",
                "symbol": be.symbol,
                "binance_event_id": be.event_id,
                "bybit_event_id": None,
                "binance_present": True,
                "bybit_present": False,
                "lead_exchange": None,
                "detection_time_delta_minutes": np.nan,
                "binance_return": be.subsequent_return_1h,
                "bybit_return": np.nan,
                "binance_pump_class": be.pump_class,
                "bybit_pump_class": None,
            }
        )

    for yi, ye in enumerate(bybit_events):
        if yi in claimed_bybit:
            continue
        rows.append(
            {
                "underlying_event_id": f"underlying_{ye.symbol}_{ye.pump_detected_ts}",
                "symbol": ye.symbol,
                "binance_event_id": None,
                "bybit_event_id": ye.event_id,
                "binance_present": False,
                "bybit_present": True,
                "lead_exchange": None,
                "detection_time_delta_minutes": np.nan,
                "binance_return": np.nan,
                "bybit_return": ye.subsequent_return_1h,
                "binance_pump_class": None,
                "bybit_pump_class": ye.pump_class,
            }
        )

    return pd.DataFrame(rows, columns=LINKED_COLUMNS)


def confusion_matrix(linked_df: pd.DataFrame) -> pd.DataFrame:
    """Pump-class agreement between exchanges, restricted to underlying
    events where both `binance_present` and `bybit_present` are True (a
    solo event has no second class to compare against). Rows are Binance's
    `pump_class`, columns are Bybit's. Returns an empty DataFrame (not a
    crash) when zero rows qualify - e.g. no cross-exchange matches at all in
    this run's window, which is expected given the ~402-vs-~166-symbol
    universe mismatch."""
    if len(linked_df) == 0:
        return pd.DataFrame()
    both = linked_df[linked_df["binance_present"] & linked_df["bybit_present"]]
    if len(both) == 0:
        return pd.DataFrame()
    return pd.crosstab(both["binance_pump_class"], both["bybit_pump_class"])


def lead_lag_summary(linked_df: pd.DataFrame) -> dict:
    """Detection-timing summary restricted to both-present underlying
    events. `n_simultaneous` counts both-present rows with `lead_exchange`
    None (identical `pump_detected_ts` on both sides, not "only one side
    present" - solo rows are excluded from `both` before this is computed).
    If there are zero both-present rows this returns zeros/NaN for every
    key rather than raising - solo-only linkage output is a normal result,
    not an error."""
    empty_result = {
        "n_both_present": 0,
        "n_binance_led": 0,
        "n_bybit_led": 0,
        "n_simultaneous": 0,
        "pct_binance_led": np.nan,
        "pct_bybit_led": np.nan,
        "median_abs_delta_minutes": np.nan,
        "p25_abs_delta_minutes": np.nan,
        "p75_abs_delta_minutes": np.nan,
    }
    if len(linked_df) == 0:
        return empty_result
    both = linked_df[linked_df["binance_present"] & linked_df["bybit_present"]]
    n_both = len(both)
    if n_both == 0:
        return empty_result

    n_binance_led = int((both["lead_exchange"] == "binance").sum())
    n_bybit_led = int((both["lead_exchange"] == "bybit").sum())
    n_simultaneous = int(both["lead_exchange"].isna().sum())
    abs_delta = both["detection_time_delta_minutes"].abs()
    return {
        "n_both_present": n_both,
        "n_binance_led": n_binance_led,
        "n_bybit_led": n_bybit_led,
        "n_simultaneous": n_simultaneous,
        "pct_binance_led": n_binance_led / n_both,
        "pct_bybit_led": n_bybit_led / n_both,
        "median_abs_delta_minutes": float(abs_delta.median()),
        "p25_abs_delta_minutes": float(abs_delta.quantile(0.25)),
        "p75_abs_delta_minutes": float(abs_delta.quantile(0.75)),
    }


def render_cross_exchange_report(linked_df: pd.DataFrame, confusion_df: pd.DataFrame, lead_lag: dict) -> str:
    """Markdown report over already-computed `link_cross_exchange_events` /
    `confusion_matrix` / `lead_lag_summary` outputs (this function does not
    recompute anything, matching the other research/*.py render_* functions'
    convention of taking precomputed inputs)."""
    lines = ["# Cross-Exchange Event Linkage Report", ""]
    total = len(linked_df)
    lines.append(f"Total underlying events: {total}")
    lines.append("")
    if total == 0:
        lines.append("No events on either exchange - nothing to link.")
        return "\n".join(lines)

    n_both = int((linked_df["binance_present"] & linked_df["bybit_present"]).sum())
    n_solo_binance = int((linked_df["binance_present"] & ~linked_df["bybit_present"]).sum())
    n_solo_bybit = int((~linked_df["binance_present"] & linked_df["bybit_present"]).sum())
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Both exchanges present: {n_both} ({n_both / total:.1%})")
    lines.append(f"- Solo Binance (no Bybit match): {n_solo_binance} ({n_solo_binance / total:.1%})")
    lines.append(f"- Solo Bybit (no Binance match): {n_solo_bybit} ({n_solo_bybit / total:.1%})")
    lines.append("")

    lines.append("## Lead/lag summary (both-present events only)")
    lines.append("")
    if lead_lag.get("n_both_present", 0) == 0:
        lines.append("No both-present events - lead/lag statistics not available.")
    else:
        lines.append(
            f"Of {lead_lag['n_both_present']} both-present events, Binance detected first in "
            f"{lead_lag['n_binance_led']} ({lead_lag['pct_binance_led']:.1%}), Bybit detected first in "
            f"{lead_lag['n_bybit_led']} ({lead_lag['pct_bybit_led']:.1%}), and {lead_lag['n_simultaneous']} "
            "were simultaneous."
        )
        lines.append("")
        lines.append("| Stat | Value (minutes) |")
        lines.append("|---|---|")
        lines.append(f"| Median absolute detection delta | {lead_lag['median_abs_delta_minutes']:.2f} |")
        lines.append(f"| P25 absolute detection delta | {lead_lag['p25_abs_delta_minutes']:.2f} |")
        lines.append(f"| P75 absolute detection delta | {lead_lag['p75_abs_delta_minutes']:.2f} |")
    lines.append("")

    lines.append("## Pump-class confusion matrix (rows=Binance, columns=Bybit, both-present events)")
    lines.append("")
    if confusion_df.empty:
        lines.append("Not enough overlapping (both-present) events to build a confusion matrix.")
    else:
        lines.append("| binance \\ bybit | " + " | ".join(str(c) for c in confusion_df.columns) + " |")
        lines.append("|---|" + "---|" * len(confusion_df.columns))
        for idx, row in confusion_df.iterrows():
            lines.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")
    lines.append("")
    return "\n".join(lines)
