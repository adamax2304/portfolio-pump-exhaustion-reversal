"""Stage 1.5 diagnostic chart generation (see project plan): renders per-event
price/volume PNG charts for visual QA of the pump-exhaustion-reversal event
study, grouped into named cohorts (top reversals, top continuations, by
pump_class, by exhaustion score) for fast eyeballing of interesting cases.
Pure rendering over already-computed EventRecords and already-loaded raw
OHLCV - no simulation, no trading logic, never imported by the trading
codepath (see research/__init__.py's boundary).

This runs from a headless CLI script, never an interactive session, so the
matplotlib backend is forced to "Agg" below before pyplot is imported -
without this, matplotlib may try to open a GUI window and hang/crash in an
automated run.

select_cohorts()'s forward_paths_df parameter expects a DataFrame matching
research/forward_study.py's eventual output: one row per event, with at
least columns `event_id`, `maximum_additional_upside`, and
`drawdown_from_eventual_peak_24h`.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord

# Cohorts defined purely by event.pump_class equality, keyed by the cohort
# name the report expects; value is the exact pump_class string produced by
# strategy/pump_classifier.py (see EventRecord.pump_class).
_PUMP_CLASS_COHORTS: tuple[tuple[str, str], ...] = (
    ("leveraged_long", "LEVERAGED_LONG"),
    ("liquidity_manipulation", "LIQUIDITY_MANIPULATION"),
    ("short_squeeze", "SHORT_SQUEEZE"),
)

HIGH_EXHAUSTION_SCORE_THRESHOLD = 65

_INVALIDATION_REASON_MAX_CHARS = 60

# event_id is built elsewhere as f"{exchange}:{symbol}:{event_seq}" (see
# backtest/simulator.py) - colons (and any other punctuation that shows up)
# are not safe filename characters, so anything outside this set is replaced.
_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")


def select_cohorts(events: list[EventRecord], forward_paths_df: pd.DataFrame, top_n: int = 10) -> dict[str, list[str]]:
    """Group event_ids into named cohorts for chart generation.

    A pump event can legitimately land in more than one cohort (e.g. a top
    reversal that also happens to be LEVERAGED_LONG) - cohorts are
    independent selections, not a partition, and event_ids are never
    deduplicated across them. A cohort key is omitted entirely from the
    returned dict if it would be empty (e.g. zero SHORT_SQUEEZE events in a
    given run is a real, expected outcome per the project's own findings so
    far, not a bug).

    forward_paths_df: one row per event, at least columns `event_id`,
    `maximum_additional_upside`, `drawdown_from_eventual_peak_24h` (matching
    research/forward_study.py's output). Rows with NaN in the ranked column
    are excluded from that ranking entirely (not just sorted last).
    """
    cohorts: dict[str, list[str]] = {}

    ranked_reversals = forward_paths_df.dropna(subset=["drawdown_from_eventual_peak_24h"]).sort_values(
        "drawdown_from_eventual_peak_24h", ascending=False, kind="stable"
    )
    top_reversals = ranked_reversals["event_id"].head(top_n).tolist()
    if top_reversals:
        cohorts["top_reversals"] = top_reversals

    ranked_continuations = forward_paths_df.dropna(subset=["maximum_additional_upside"]).sort_values(
        "maximum_additional_upside", ascending=False, kind="stable"
    )
    top_continuations = ranked_continuations["event_id"].head(top_n).tolist()
    if top_continuations:
        cohorts["top_continuations"] = top_continuations

    high_exhaustion = [e.event_id for e in events if e.exhaustion_score_peak > HIGH_EXHAUSTION_SCORE_THRESHOLD]
    if high_exhaustion:
        cohorts["high_exhaustion_score"] = high_exhaustion

    for cohort_name, pump_class in _PUMP_CLASS_COHORTS:
        matches = [e.event_id for e in events if e.pump_class == pump_class]
        if matches:
            cohorts[cohort_name] = matches

    return cohorts


def _locate_event_window(
    ohlcv: pd.DataFrame, event: EventRecord, window_bars_before: int, window_bars_after: int
) -> pd.DataFrame:
    """Slice ohlcv (assumed sorted ascending by timestamp, as everything on
    disk is - see data/storage.py's merge_write) down to the display window:
    window_bars_before bars before event.pump_start_ts through
    window_bars_after bars after event.pump_detected_ts. Clips at the
    available data's edges instead of raising when the event sits near the
    start/end of what's on disk; returns an empty (but same-columns)
    DataFrame, never an error, if ohlcv has no rows or the computed slice is
    empty."""
    if ohlcv.empty or "timestamp" not in ohlcv.columns:
        return ohlcv.iloc[0:0]
    timestamps = ohlcv["timestamp"].to_numpy()
    start_idx = int(np.searchsorted(timestamps, event.pump_start_ts))
    detect_idx = int(np.searchsorted(timestamps, event.pump_detected_ts))
    lo = max(0, start_idx - window_bars_before)
    hi = min(len(ohlcv), detect_idx + window_bars_after + 1)
    if hi <= lo:
        return ohlcv.iloc[0:0]
    return ohlcv.iloc[lo:hi].reset_index(drop=True)


def plot_event_chart(
    event: EventRecord,
    ohlcv: pd.DataFrame,
    out_path: Path | str,
    window_bars_before: int = 100,
    window_bars_after: int = 200,
) -> None:
    """Render and save one PNG chart for one event: close-price line (top
    panel) and volume bars (bottom panel) over a window around the pump,
    with vertical markers for pump_detected_ts/pump_peak_ts and a short
    annotation of invalidation_reason when set.

    Never raises. This is called in a loop across dozens of events from an
    orchestrating script, so a bad/short/empty window must not abort the
    whole batch - it still produces a valid PNG (a placeholder annotation if
    the window is truly empty, or just whatever few bars are available if
    it's short but nonempty).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = (
        f"{event.symbol} ({event.exchange}) - {event.pump_class} - "
        f"exhaustion_peak={event.exhaustion_score_peak:.0f} - {event.final_state}"
    )

    fig, (price_ax, volume_ax) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title)

    window = _locate_event_window(ohlcv, event, window_bars_before, window_bars_after)

    if window.empty:
        price_ax.text(
            0.5, 0.5, "insufficient data for chart window",
            transform=price_ax.transAxes, ha="center", va="center", fontsize=11,
        )
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return

    x = pd.to_datetime(window["timestamp"], unit="ms")
    price_ax.plot(x, window["close"], linewidth=1.0, color="tab:blue")
    price_ax.set_ylabel("close")

    ts_values = window["timestamp"].to_numpy()
    if len(ts_values) > 1:
        spacing_ms = float(np.median(np.diff(ts_values)))
    else:
        spacing_ms = 5 * 60 * 1000.0  # single-bar window: arbitrary sane default bar width
    bar_width_days = max((spacing_ms / 86_400_000.0) * 0.8, 1e-6)
    volume_ax.bar(x, window["volume"], width=bar_width_days, color="tab:gray")
    volume_ax.set_ylabel("volume")

    window_start_ts = int(ts_values[0])
    window_end_ts = int(ts_values[-1])
    label_trans = price_ax.get_xaxis_transform()  # x in data coords, y in axes-fraction coords

    if window_start_ts <= event.pump_detected_ts <= window_end_ts:
        detected_x = pd.to_datetime(event.pump_detected_ts, unit="ms")
        price_ax.axvline(detected_x, color="darkorange", linestyle="--", linewidth=1.0)
        price_ax.text(
            detected_x, 0.97, " pump detected", transform=label_trans,
            rotation=90, va="top", ha="left", fontsize=8, color="darkorange",
        )

    if event.pump_peak_ts and window_start_ts <= event.pump_peak_ts <= window_end_ts:
        peak_x = pd.to_datetime(event.pump_peak_ts, unit="ms")
        price_ax.axvline(peak_x, color="crimson", linestyle="--", linewidth=1.0)
        price_ax.text(
            peak_x, 0.97, " peak", transform=label_trans,
            rotation=90, va="top", ha="left", fontsize=8, color="crimson",
        )

    if event.invalidation_reason:
        reason = event.invalidation_reason
        if len(reason) > _INVALIDATION_REASON_MAX_CHARS:
            reason = reason[: _INVALIDATION_REASON_MAX_CHARS - 3] + "..."
        price_ax.text(
            0.98, 0.02, f"invalidation: {reason}", transform=price_ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="dimgray",
        )

    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_cohort_charts(
    cohorts: dict[str, list[str]],
    events: list[EventRecord],
    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame],
    out_dir: Path | str,
    window_bars_before: int = 100,
    window_bars_after: int = 200,
) -> dict[str, list[str]]:
    """Render every event_id in every cohort via plot_event_chart, saved to
    out_dir/{cohort_name}/{event_id_sanitized}.png.

    ohlcv_by_symbol is keyed by (exchange, symbol); an event whose key is
    missing is skipped, not an error - this runs across whatever OHLCV
    happens to be cached on disk, and one missing symbol must not crash the
    whole batch.

    Returns {cohort_name: [saved_file_path_str, ...]} - one entry for every
    cohort_name present in the input `cohorts` dict (even if every event in
    it ended up skipped, in which case the list is empty), for a report
    script to embed relative paths from.
    """
    out_dir = Path(out_dir)
    events_by_id = {e.event_id: e for e in events}
    saved: dict[str, list[str]] = {}

    for cohort_name, event_ids in cohorts.items():
        saved_paths: list[str] = []
        for event_id in event_ids:
            event = events_by_id.get(event_id)
            if event is None:
                continue
            ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
            if ohlcv is None:
                continue
            filename = _FILENAME_SANITIZE_RE.sub("_", event_id) + ".png"
            out_path = out_dir / cohort_name / filename
            plot_event_chart(event, ohlcv, out_path, window_bars_before, window_bars_after)
            saved_paths.append(str(out_path))
        saved[cohort_name] = saved_paths

    return saved
