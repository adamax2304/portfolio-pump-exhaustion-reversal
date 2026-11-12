"""Spec sections 14/15/16: Forward Event Study - retrospective "what actually
happened next" analytics for already-detected pump events (forward returns at
fixed horizons, the eventual peak's price/timing, drawdown from that eventual
peak, and how fast price gives back X% off that peak).

LOOKAHEAD-SAFETY BOUNDARY (read before touching this file): every function
here reads OHLCV bars strictly AFTER an event's pump_detected_ts - the exact
thing a live/backtest trading decision must never do. That is fine for
retrospective research (this package's whole purpose - see
research/__init__.py) but would be a lookahead bug the instant it reached a
trading decision. This module must NEVER be imported by backtest/simulator.py
or anything under strategy/ - only by research scripts/notebooks and its own
tests.

"Eventual peak" is bounded, not truly unbounded: it is the highest high found
within max_forward_bars bars of detection (default
DEFAULT_MAX_EVENT_HORIZON_BARS, imported from backtest/simulator.py so this
module's default bound stays consistent with the rest of the codebase -
~6.9 days of 5m bars), or however much OHLCV history actually exists past
detection if that runs out first. A true unbounded "eventual" peak does not
exist - our archive is finite - so any event this bound actually binds on is
flagged via right_censored=True rather than silently reported as if it were a
complete picture: its forward/eventual-peak/drawdown numbers may understate
the true (unknowable) values.

Assumes `ohlcv` is sorted ascending by timestamp, one row per bar - the shape
data/storage.py's read_range() always returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import DEFAULT_MAX_EVENT_HORIZON_BARS, EventRecord

# label -> horizon in milliseconds, from the detection bar's close.
FORWARD_RETURN_HORIZONS: list[tuple[str, int]] = [
    ("15m", 15 * 60_000),
    ("30m", 30 * 60_000),
    ("1h", 3_600_000),
    ("2h", 7_200_000),
    ("4h", 14_400_000),
    ("8h", 28_800_000),
    ("12h", 43_200_000),
    ("24h", 86_400_000),
]

# label -> horizon in milliseconds, from the eventual-peak bar.
DRAWDOWN_FROM_PEAK_HORIZONS: list[tuple[str, int]] = [
    ("1h", 3_600_000),
    ("4h", 14_400_000),
    ("8h", 28_800_000),
    ("24h", 86_400_000),
]

# Percent-off-peak thresholds for "how long until price is down X% from the
# eventual peak". Spec section 14's prose lists only 5/10/20, but section 16's
# breakdown table explicitly asks for a prob_minus30pct column too (see
# breakdown_by_pump_class) - extended here to 30 so that column has an
# underlying per-event field to aggregate, rather than leaving section 16
# unimplementable. See module-level NOTE below and the docstrings on
# compute_forward_path/breakdown_by_pump_class for the full explanation.
PEAK_DROP_THRESHOLDS_PCT: list[int] = [5, 10, 20, 30]

# Hour windows used by breakdown_by_pump_class's probability-of-large-move columns.
PROB_WITHIN_HOURS: list[int] = [1, 4, 8, 24]

# NOTE on the two deviations from the literal spec text in this file:
#   1. compute_forward_path() returns a `time_peak_to_minus{X}pct_minutes`
#      companion alongside every `time_peak_to_minus{X}pct_bars` field (spec
#      section 14 only named the _bars fields). breakdown_by_pump_class()'s
#      signature is `(forward_paths_df) -> pd.DataFrame` - it is never handed
#      bar_interval_ms, so it cannot itself convert bars to minutes. Without
#      these companions, section 16's "converted to minutes" probability
#      columns would be impossible to compute from forward_paths_df alone.
#   2. PEAK_DROP_THRESHOLDS_PCT includes 30 (spec section 14's prose lists
#      only 5/10/20) so that section 16's prob_minus30pct_* columns have a
#      real per-event field behind them.


def _pct_from_reference(price: float, reference_price: float) -> float:
    """(price - reference_price) / reference_price, NaN (never 0) if reference_price is falsy/zero."""
    return (price - reference_price) / reference_price if reference_price else np.nan


def _nan_forward_return_fields() -> dict:
    return {f"forward_return_{label}": np.nan for label, _ in FORWARD_RETURN_HORIZONS}


def _nan_forward_extrema_fields() -> dict:
    out = {
        "maximum_additional_upside": np.nan,
        "maximum_downside": np.nan,
        "time_to_eventual_peak_bars": np.nan,
        "time_to_eventual_peak_minutes": np.nan,
        "eventual_peak_price": np.nan,
        "eventual_peak_return_from_detection": np.nan,
    }
    for label, _ in DRAWDOWN_FROM_PEAK_HORIZONS:
        out[f"drawdown_from_eventual_peak_{label}"] = np.nan
    for pct in PEAK_DROP_THRESHOLDS_PCT:
        out[f"time_peak_to_minus{pct}pct_bars"] = np.nan
        out[f"time_peak_to_minus{pct}pct_minutes"] = np.nan
    return out


def _nan_forward_fields() -> dict:
    return {**_nan_forward_return_fields(), **_nan_forward_extrema_fields()}


FORWARD_PATH_COLUMNS: list[str] = (
    ["event_id", "symbol", "exchange", "pump_class"]
    + [f"forward_return_{label}" for label, _ in FORWARD_RETURN_HORIZONS]
    + [
        "maximum_additional_upside",
        "maximum_downside",
        "time_to_eventual_peak_bars",
        "time_to_eventual_peak_minutes",
        "eventual_peak_price",
        "eventual_peak_return_from_detection",
    ]
    + [f"drawdown_from_eventual_peak_{label}" for label, _ in DRAWDOWN_FROM_PEAK_HORIZONS]
    + [f"time_peak_to_minus{pct}pct_{suffix}" for pct in PEAK_DROP_THRESHOLDS_PCT for suffix in ("bars", "minutes")]
    + ["right_censored"]
)


def compute_forward_path(
    event: EventRecord,
    ohlcv: pd.DataFrame,
    bar_interval_ms: int,
    max_forward_bars: int = DEFAULT_MAX_EVENT_HORIZON_BARS,
) -> dict:
    """One event's full forward path: fixed-horizon returns, the bounded
    window's extremes, the eventual peak's timing/price, drawdown from that
    peak at several horizons, and how fast price gave back X% off the peak.

    `ohlcv` must be the event's own symbol's full raw OHLCV (as loaded by
    data/storage.py's read_range) - this function locates the bar at/right
    after event.pump_detected_ts itself, so it does not need a pre-sliced
    frame. Every number here is computed strictly from bars AFTER that
    detection bar (see module docstring's lookahead-safety boundary) except
    the detection bar's own close, used only as the fixed reference price
    every return/drawdown is measured against.

    Returns NaN (never 0) for any field that would need forward data beyond
    what's actually available in `ohlcv` - see right_censored.
    """
    passthrough = {"event_id": event.event_id, "symbol": event.symbol, "exchange": event.exchange, "pump_class": event.pump_class}
    n = len(ohlcv)
    if n == 0:
        return {**passthrough, **_nan_forward_fields(), "right_censored": True}

    timestamps = ohlcv["timestamp"].to_numpy()
    detect_idx = int(np.searchsorted(timestamps, event.pump_detected_ts, side="left"))
    if detect_idx >= n:
        # pump_detected_ts is after every bar we have on disk for this symbol -
        # no anchor bar at all, so nothing forward-looking is computable either.
        return {**passthrough, **_nan_forward_fields(), "right_censored": True}

    last_idx = n - 1
    detect_close = float(ohlcv["close"].iloc[detect_idx])
    n_available_forward = last_idx - detect_idx
    n_forward_used = min(n_available_forward, max_forward_bars)
    window_end_idx = detect_idx + n_forward_used
    # True only when we actually ran out of data before max_forward_bars bars
    # were available - not when max_forward_bars itself is what capped the
    # window. This is the per-event marker for the module docstring's "true
    # unbounded eventual peak does not exist" limitation.
    right_censored = bool(n_available_forward < max_forward_bars)

    highs = ohlcv["high"].to_numpy()
    lows = ohlcv["low"].to_numpy()
    closes = ohlcv["close"].to_numpy()

    out = dict(passthrough)

    for label, horizon_ms in FORWARD_RETURN_HORIZONS:
        bars_ahead = max(round(horizon_ms / bar_interval_ms), 1)
        target_idx = detect_idx + bars_ahead
        out[f"forward_return_{label}"] = (
            _pct_from_reference(float(closes[target_idx]), detect_close) if target_idx <= window_end_idx else np.nan
        )

    if n_forward_used == 0:
        out.update(_nan_forward_extrema_fields())
        out["right_censored"] = right_censored
        return out

    fwd_highs = highs[detect_idx + 1 : window_end_idx + 1]
    fwd_lows = lows[detect_idx + 1 : window_end_idx + 1]
    out["maximum_additional_upside"] = _pct_from_reference(float(fwd_highs.max()), detect_close)
    out["maximum_downside"] = _pct_from_reference(float(fwd_lows.min()), detect_close)

    peak_offset = int(fwd_highs.argmax())  # 0-based position within fwd_highs
    peak_abs_idx = detect_idx + 1 + peak_offset
    eventual_peak_price = float(highs[peak_abs_idx])
    bars_to_peak = peak_abs_idx - detect_idx
    out["time_to_eventual_peak_bars"] = bars_to_peak
    out["time_to_eventual_peak_minutes"] = bars_to_peak * bar_interval_ms / 60_000.0
    out["eventual_peak_price"] = eventual_peak_price
    out["eventual_peak_return_from_detection"] = _pct_from_reference(eventual_peak_price, detect_close)

    for label, horizon_ms in DRAWDOWN_FROM_PEAK_HORIZONS:
        bars_ahead = max(round(horizon_ms / bar_interval_ms), 1)
        # Bounded by available data (min against window_end_idx); NaN only
        # when there is literally no bar after the peak to measure a
        # subsequent low from - a real, if partial, window still yields a
        # real (if conservative) drawdown rather than being discarded.
        sub_end = min(window_end_idx, peak_abs_idx + bars_ahead)
        if sub_end > peak_abs_idx:
            sub_low = float(lows[peak_abs_idx : sub_end + 1].min())
            out[f"drawdown_from_eventual_peak_{label}"] = (
                (eventual_peak_price - sub_low) / eventual_peak_price if eventual_peak_price else np.nan
            )
        else:
            out[f"drawdown_from_eventual_peak_{label}"] = np.nan

    for pct in PEAK_DROP_THRESHOLDS_PCT:
        bars_key, minutes_key = f"time_peak_to_minus{pct}pct_bars", f"time_peak_to_minus{pct}pct_minutes"
        if peak_abs_idx >= window_end_idx:
            out[bars_key] = np.nan
            out[minutes_key] = np.nan
            continue
        threshold_price = eventual_peak_price * (1 - pct / 100.0)
        sub_lows_after_peak = lows[peak_abs_idx + 1 : window_end_idx + 1]
        hit_positions = np.nonzero(sub_lows_after_peak <= threshold_price)[0]
        if hit_positions.size:
            bars_val = int(hit_positions[0]) + 1  # +1: sub_lows_after_peak starts one bar past the peak
            out[bars_key] = bars_val
            out[minutes_key] = bars_val * bar_interval_ms / 60_000.0
        else:
            # Never happened within the bounded window - NaN here (genuinely
            # unresolved), NOT the same as the False used for this in
            # breakdown_by_pump_class's probability columns (see there).
            out[bars_key] = np.nan
            out[minutes_key] = np.nan

    out["right_censored"] = right_censored
    return out


def compute_forward_paths_for_events(
    events: list[EventRecord],
    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame],
    bar_interval_ms: int,
    max_forward_bars: int = DEFAULT_MAX_EVENT_HORIZON_BARS,
) -> pd.DataFrame:
    """Batch version of compute_forward_path: one output row per event.
    `ohlcv_by_symbol` is keyed by (exchange, symbol) so the caller can load
    each distinct symbol's OHLCV once even when many events share a symbol.
    An event whose (exchange, symbol) key is missing from ohlcv_by_symbol is
    silently skipped (not crashed on) - the caller is expected to have loaded
    OHLCV for exactly the symbols its events need."""
    rows = []
    for event in events:
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None:
            continue
        rows.append(compute_forward_path(event, ohlcv, bar_interval_ms, max_forward_bars=max_forward_bars))
    if not rows:
        return pd.DataFrame(columns=FORWARD_PATH_COLUMNS)
    return pd.DataFrame(rows)


BREAKDOWN_COLUMNS: list[str] = (
    [
        "pump_class",
        "n",
        "median_additional_upside",
        "p75_additional_upside",
        "p90_additional_upside",
        "median_drawdown_from_peak_24h",
        "p75_drawdown_from_peak_24h",
        "p90_drawdown_from_peak_24h",
        "median_time_to_peak_minutes",
    ]
    + [f"prob_minus{pct}pct_{hours}h" for pct in PEAK_DROP_THRESHOLDS_PCT for hours in PROB_WITHIN_HOURS]
)


def breakdown_by_pump_class(forward_paths_df: pd.DataFrame) -> pd.DataFrame:
    """One row per pump_class: upside/drawdown/time-to-peak distribution
    stats, plus P(price is down X% from the EVENTUAL PEAK within Y hours of
    that peak) for every {5,10,20,30}% x {1,4,8,24}h combination.

    The within-Y-hours indicator is `time_peak_to_minus{X}pct_minutes <=
    Y*60`. A row where that minutes field is NaN - the drop never happened
    within compute_forward_path's own bounded window - evaluates that
    comparison to False (NaN comparisons are always False in numpy/pandas,
    which is relied on here deliberately, not by accident): "never observed
    to happen that fast" is a real, meaningfully different answer from
    "unknown", and it is a genuine No for a probability-within-Y-hours
    question, so it counts in the denominator as a non-event rather than
    being dropped from it.
    """
    if forward_paths_df.empty:
        return pd.DataFrame(columns=BREAKDOWN_COLUMNS)

    rows = []
    for pump_class, group in forward_paths_df.groupby("pump_class"):
        if len(group) == 0:
            continue
        row = {
            "pump_class": pump_class,
            "n": len(group),
            "median_additional_upside": group["maximum_additional_upside"].median(),
            "p75_additional_upside": group["maximum_additional_upside"].quantile(0.75),
            "p90_additional_upside": group["maximum_additional_upside"].quantile(0.90),
            "median_drawdown_from_peak_24h": group["drawdown_from_eventual_peak_24h"].median(),
            "p75_drawdown_from_peak_24h": group["drawdown_from_eventual_peak_24h"].quantile(0.75),
            "p90_drawdown_from_peak_24h": group["drawdown_from_eventual_peak_24h"].quantile(0.90),
            "median_time_to_peak_minutes": group["time_to_eventual_peak_minutes"].median(),
        }
        for pct in PEAK_DROP_THRESHOLDS_PCT:
            minutes = group[f"time_peak_to_minus{pct}pct_minutes"]
            for hours in PROB_WITHIN_HOURS:
                row[f"prob_minus{pct}pct_{hours}h"] = float((minutes <= hours * 60).mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS)


def _fmt_pct(value, digits: int = 1) -> str:
    """Mirrors backtest/reports.py's _fmt_pct exactly - same NaN-safe convention, kept local to avoid importing a private helper cross-package."""
    if value is None or value != value:  # NaN check without importing numpy just for this
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value, digits: int = 3) -> str:
    """Mirrors backtest/reports.py's _fmt_num exactly."""
    if value is None or value != value:
        return "n/a"
    if value in (float("inf"), float("-inf")):
        return str(value)
    return f"{value:.{digits}f}"


def render_forward_event_study_report(forward_paths_df: pd.DataFrame, by_class_df: pd.DataFrame) -> str:
    """Markdown report: overall N and right-censoring, an overall
    upside/eventual-peak/drawdown summary table, then the full by-pump-class
    breakdown from breakdown_by_pump_class."""
    lines = [
        "# Forward Event Study Report",
        "",
        "Stage 1.5 research/diagnostic only - retrospective analysis of bars strictly after detection, never fed into a trading decision (see module docstring).",
        "",
    ]

    n = len(forward_paths_df)
    lines.append(f"- Events analyzed: {n}")
    if n:
        n_censored = int(forward_paths_df["right_censored"].sum())
        lines.append(
            f"- Right-censored (ran out of forward data before the bounded horizon): {n_censored} ({_fmt_pct(n_censored / n)})"
        )
    else:
        lines.append("- Right-censored: n/a (no events)")
    lines.append("")

    lines += ["## Overall summary", "", "| Metric | Median | P75 | P90 |", "|---|---|---|---|"]
    for label, col in (
        ("Maximum additional upside", "maximum_additional_upside"),
        ("Eventual peak return from detection", "eventual_peak_return_from_detection"),
        ("Drawdown from eventual peak (24h)", "drawdown_from_eventual_peak_24h"),
    ):
        if n and col in forward_paths_df:
            series = forward_paths_df[col]
            median, p75, p90 = series.median(), series.quantile(0.75), series.quantile(0.90)
        else:
            median = p75 = p90 = float("nan")
        lines.append(f"| {label} | {_fmt_pct(median)} | {_fmt_pct(p75)} | {_fmt_pct(p90)} |")
    lines.append("")

    lines += ["## By pump class", ""]
    if by_class_df.empty:
        lines.append("No events.")
    else:
        cols = list(by_class_df.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, row in by_class_df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if c == "pump_class":
                    cells.append(str(v))
                elif c == "n":
                    cells.append(str(int(v)))
                elif c == "median_time_to_peak_minutes":
                    cells.append(_fmt_num(v, 1))
                else:
                    cells.append(_fmt_pct(v))
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
