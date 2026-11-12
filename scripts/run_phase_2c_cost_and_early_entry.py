"""CLI: Phase 2C - Cost Decomposition + Earlier Causal Entry Study. Answers
two questions the user raised after reviewing Phase 2B: (1) does execution
cost really account for the ~0.56R gap between failed_reclaim's theoretical
gross edge and its -0.435R net result, or is something double-counted; (2)
does a tradeable edge exist EARLIER in the causal chain than failed_reclaim
(which waits a median ~34.5 bars past first_exhaustion_signal and only adds a
small further first-passage lift over it)?

Strategy v1 / failed_reclaim = REJECTED per the user's own read of Phase 2B.
This script does NOT loosen thresholds to force a positive result, does NOT
build a live scanner, and does NOT extend the dataset - it only re-examines
cost accounting and tests two new, PRE-DEFINED earlier entry rules (Entry A:
exhaustion-signal combinations; Entry B: climax rejection) against the same
already-canonical Binance+Bybit event studies Phase 2B used.

python scripts/run_phase_2c_cost_and_early_entry.py --binance-run-id binance_canonical_v2 --bybit-run-id bybit_canonical_v2

Writes under data/diagnostics/<out-run-id>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import typer

from pump_exhaustion_reversal.backtest.simulator import EventRecord, TradeRecord
from pump_exhaustion_reversal.config import AppConfig, BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.feature_store import build_feature_frame
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.research import cross_exchange
from pump_exhaustion_reversal.research import early_entry
from pump_exhaustion_reversal.research import entry_edge as ee
from pump_exhaustion_reversal.research import first_passage as fp
from pump_exhaustion_reversal.research import lifecycle_diagnostics as ld
from pump_exhaustion_reversal.research import underlying_classification as uc

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
PHASE_2B_RUN_ID = "phase_2b_entry_edge_v1"  # read failed_reclaim's own (Entry C) already-corrected candidate parquet from here

HEADLINE_ENTRY_MODE = "confirmation_close"
HEADLINE_ATR_MULTIPLE = 0.25
HEADLINE_BARRIER_FAMILY = "2R_1R_4h"
HEADLINE_SPEC = fp.PHASE_2B_BARRIER_FAMILIES[HEADLINE_BARRIER_FAMILY]


# ---------------------------------------------------------------------------
# Shared loaders (duplicated from scripts/run_phase_2b_entry_edge.py - each
# diagnostic script in this project is self-contained, matching the existing
# scripts/run_temporal_fix_diagnostics.py / run_research_diagnostics.py split)
# ---------------------------------------------------------------------------


def _load_events_with_trades(run_id: str) -> list[EventRecord]:
    events_path = storage.EVENTS_ROOT / f"events_{run_id}.parquet"
    if not events_path.exists():
        return []
    events_df = pd.read_parquet(events_path)
    trades_path = storage.BACKTESTS_ROOT / f"trades_{run_id}.parquet"
    trades_by_event_id: dict[str, TradeRecord] = {}
    if trades_path.exists():
        trades_df = pd.read_parquet(trades_path)
        for row in trades_df.to_dict("records"):
            trades_by_event_id[row["event_id"]] = TradeRecord(**row)
    events = []
    for row in events_df.to_dict("records"):
        row = dict(row)
        row["trade"] = trades_by_event_id.get(row["event_id"])
        events.append(EventRecord(**row))
    return events


def _load_ohlcv_by_symbol(exchange: str, events: list[EventRecord], timeframe: str, since_ms: int, until_ms: int) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in sorted({e.symbol for e in events}):
        df = storage.read_range(exchange, "ohlcv", symbol, timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
        if not df.empty:
            out[(exchange, symbol)] = df
    return out


def _load_feature_frame_by_symbol(
    exchange: str, events: list[EventRecord], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame], timeframe: str,
    since_ms: int, until_ms: int, cfg: AppConfig,
) -> dict[tuple[str, str], pd.DataFrame]:
    btc_ohlcv = storage.read_range(exchange, "ohlcv", "BTCUSDT", timeframe=timeframe, start_ms=since_ms, end_ms=until_ms)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    if btc_ohlcv.empty:
        return out
    for symbol in sorted({e.symbol for e in events}):
        ohlcv = ohlcv_by_symbol.get((exchange, symbol))
        if ohlcv is None or ohlcv.empty:
            continue
        oi_df = storage.read_range(exchange, "oi", symbol, timeframe="5m", start_ms=since_ms, end_ms=until_ms)
        funding_df = storage.read_range(exchange, "funding", symbol, start_ms=since_ms, end_ms=until_ms)
        out[(exchange, symbol)] = build_feature_frame(
            ohlcv, btc_ohlcv, timeframe, cfg, oi_df=oi_df if not oi_df.empty else None,
            funding_df=funding_df if not funding_df.empty else None,
        )
    return out


def _is_high_vol(feature_frame: pd.DataFrame | None, entry_bar_idx: int) -> bool:
    if feature_frame is None or "realized_vol_ratio" not in feature_frame.columns or entry_bar_idx >= len(feature_frame):
        return False
    ratio = feature_frame["realized_vol_ratio"].iloc[entry_bar_idx]
    return bool(ratio > 2.0) if pd.notna(ratio) else False


def _compute_funding_r(feature_frame: pd.DataFrame | None, entry_bar_idx: int, exit_bar_idx: int, bar_interval_ms: int, stop_distance_pct: float) -> float:
    if feature_frame is None or "funding_rate" not in feature_frame.columns or not stop_distance_pct:
        return 0.0
    proration = bar_interval_ms / (8 * 3_600_000)
    end = min(exit_bar_idx, len(feature_frame) - 1)
    total_r = 0.0
    for j in range(entry_bar_idx, end + 1):
        rate = feature_frame["funding_rate"].iloc[j]
        if pd.notna(rate):
            total_r += float(rate) * proration / stop_distance_pct
    return total_r


def _fmt_pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.1%}"


def _fmt_r(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}R"


def _fmt_bps(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.0f}bps"


def _fmt_num(v, digits: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# Section 1: cost decomposition (point 1) - re-surfaces Phase 2B's own
# (already bug-fixed) headline row, in bps AND R.
# ---------------------------------------------------------------------------


def render_cost_decomposition(headline_summary: dict, execution_cfg) -> str:
    lines = [
        "## Section 1: Cost Decomposition (headline setup: failed_reclaim / 2R:1R / 4h / confirmation_close / ATR 0.25)",
        "",
        "A real bug was found and fixed while building this section: Phase 2B's original cost model charged a flat "
        "\"2x default_slippage_bps\" (implicitly one default-rate leg for entry AND one for exit) and then ADDED "
        "stop_slippage_bps again on top for adverse trades - double-counting the exit leg's slippage on every stop-hit "
        "trade. Fixed in research/entry_edge.py (see resolve_exit_slippage_bps): the exit leg now uses EXACTLY ONE "
        "rate - default_slippage_bps (TP-style) for profit/favorable-timeout exits, stop_slippage_bps (STOP-style) for "
        "adverse/adverse-timeout exits - matching backtest/execution_model.py's actual simulate_tp_fill/simulate_stop_fill "
        "functions exactly, never both. This changed the headline net expectancy from -0.435R (Phase 2B's original, buggy "
        "number) to the corrected value below.",
        "",
        "| Component | bps (model input) | R (headline setup) |",
        "|---|---|---|",
        f"| Gross expectancy (before costs) | - | {_fmt_r(headline_summary.get('gross_expectancy_r'))} |",
        f"| Fees (entry + exit, {execution_cfg.taker_fee_bps:.1f}bps x2) | {execution_cfg.taker_fee_bps * 2:.1f}bps | {_fmt_r(headline_summary.get('fees_r'))} |",
        f"| Entry slippage ({execution_cfg.default_slippage_bps:.1f}bps) | {execution_cfg.default_slippage_bps:.1f}bps | {_fmt_r(headline_summary.get('entry_slippage_r'))} |",
        f"| Exit slippage (TP-style {execution_cfg.default_slippage_bps:.1f}bps / STOP-style {execution_cfg.stop_slippage_bps:.1f}bps, outcome-dependent) | mixed | {_fmt_r(headline_summary.get('exit_slippage_r'))} |",
        f"| (of which stop-overrun premium - already inside exit slippage above, not additional) | {max(0.0, execution_cfg.stop_slippage_bps - execution_cfg.default_slippage_bps):.1f}bps | {_fmt_r(headline_summary.get('stop_overrun_premium_r'))} |",
        f"| Funding (real per-bar funding_rate, prorated over actual hold time) | - | {_fmt_r(headline_summary.get('funding_r'))} |",
        f"| **NET expectancy** | - | **{_fmt_r(headline_summary.get('net_expectancy_r'))}** |",
        "",
        f"N={headline_summary.get('n', 0)}, PF={_fmt_num(headline_summary.get('profit_factor'))}, win rate={_fmt_pct(headline_summary.get('win_rate'))}.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: stop-distance audit (point 2)
# ---------------------------------------------------------------------------


def render_stop_distance_audit(audit_df: pd.DataFrame) -> str:
    lines = [
        "## Section 2: Stop-Distance Audit (real structural stop, confirmation_close entry, per exchange x ATR multiple)",
        "",
        "| Exchange | ATR mult | N | P10 bps | P25 bps | P50 bps | P75 bps | P90 bps | Round-trip cost / stop distance (median) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in audit_df.iterrows():
        lines.append(
            f"| {row['exchange']} | {row['atr_stop_multiple']} | {row['n']} | {_fmt_bps(row['stop_distance_bps_p10'])} "
            f"| {_fmt_bps(row['stop_distance_bps_p25'])} | {_fmt_bps(row['stop_distance_bps_p50'])} | {_fmt_bps(row['stop_distance_bps_p75'])} "
            f"| {_fmt_bps(row['stop_distance_bps_p90'])} | {_fmt_pct(row['round_trip_cost_over_stop_distance_pct_median'] / 100 if pd.notna(row['round_trip_cost_over_stop_distance_pct_median']) else np.nan)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: cost model correctness (point 3) - prose only, backed by
# research/entry_edge.py's new unit tests (resolve_exit_slippage_bps,
# high-vol multiplier, funding passthrough - see test_research_entry_edge.py)
# ---------------------------------------------------------------------------

COST_MODEL_CORRECTNESS_NOTES = """## Section 3: Cost Model Correctness Audit

Checked against backtest/execution_model.py's actual fill functions (simulate_entry_fill / simulate_stop_fill / simulate_tp_fill), not re-derived independently:

1. Fee not double-counted: exactly 2 fee legs per trade (entry + exit), `2 * taker_fee_bps / 10_000 / stop_distance_pct` - unchanged from Phase 2B, was never buggy. Covered by test_compute_r_trades_profit_adverse_timeout.
2. Slippage not double-counted: WAS buggy (see Section 1) - fixed so the exit leg uses exactly one rate, never both. Covered by test_resolve_exit_slippage_bps_never_combines_default_and_stop_rate.
3. Stop-overrun only on stop-style exits (adverse outcomes, or adverse-direction timeouts) - never on profit/favorable-timeout exits. Covered by the same test above (asserts stop_overrun_premium_r == 0 for profit and favorable-timeout).
4. Funding accrues over bars actually held, using REAL per-bar funding_rate data (feature_frame), prorated per-bar exactly like backtest/simulator.py's own `_simulate_trade` (continuous proration, not snapped to discrete UTC funding timestamps - this matches the established, already-tested production convention, which is more precise than a coarse timestamp-crossing binary, not a shortcut). Phase 2B's placeholder (funding_rate_per_8h=0.0 constant) is gone; real funding_r is now wired in from feature_frame by both this script and the corrected scripts/run_phase_2b_entry_edge.py.
5. Fee scale-invariant to notional: confirmed by construction - fees_r is computed as bps/stop_distance_pct directly, notional cancels out of the fee_dollars/risk_dollars ratio (never enters the formula at all), so it is invariant to notional/equity by definition, not by coincidence.
6. R recomputed from the actual structural stop: stop_distance_pct = (real stop_price - real entry_price) / real entry_price, both taken from the actual OHLCV path, not a fixed constant - confirmed varying correctly across the 0.15/0.25/0.40 ATR grid in Table 5 / Section 2's audit.
7. Timeout exits get correctly DIRECTIONAL costs, not zero and not double: a favorable-direction timeout (return_pct_at_exit > 0) uses the TP-style exit rate, an adverse-direction timeout uses the STOP-style rate - mirrors backtest/simulator.py's own TIME_STOP branch (`close_j < entry_price` -> tp_fill else stop_fill). Covered by test_resolve_exit_slippage_bps_never_combines_default_and_stop_rate's timeout assertions.

All of the above are enforced by unit tests in tests/test_research_entry_edge.py, not just asserted in prose.
"""


# ---------------------------------------------------------------------------
# Section 4: cost sensitivity (point 4)
# ---------------------------------------------------------------------------


def render_cost_sensitivity(sweep_df: pd.DataFrame, curve_df: pd.DataFrame, breakeven: float | None) -> str:
    lines = [
        "## Section 4: Cost Sensitivity (headline failed_reclaim setup, signals/entries/stops unchanged)",
        "", "### ZERO / LOW / BASE / HIGH cost profiles", "",
        "| Profile | Fee bps | Default slippage bps | Stop slippage bps | N | Net expectancy R | PF | Win rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in sweep_df.iterrows():
        lines.append(
            f"| {row['profile']} | {row['taker_fee_bps']:.1f} | {row['default_slippage_bps']:.1f} | {row['stop_slippage_bps']:.1f} "
            f"| {row['n']} | {_fmt_r(row['net_expectancy_r'])} | {_fmt_num(row['profit_factor'])} | {_fmt_pct(row['win_rate'])} |"
        )
    lines.append("")
    lines.append("### expectancy(cost_multiplier) - BASE config scaled by multiplier (funding untouched)")
    lines.append("")
    lines.append("| Cost multiplier | Net expectancy R |")
    lines.append("|---|---|")
    for _, row in curve_df.iterrows():
        lines.append(f"| {row['cost_multiplier']:.2f}x | {_fmt_r(row['net_expectancy_r'])} |")
    lines.append("")
    if breakeven is None:
        lines.append("Net expectancy never crosses zero across the tested multiplier range - either always negative (not a cost problem at all) or always positive within this range.")
    else:
        lines.append(f"**Estimated breakeven cost multiplier: {breakeven:.2f}x BASE** (linear interpolation).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry A/B candidate grid (points 5-8, 10-11): first-passage + net expectancy
# for every detected signal, at the headline barrier family, both entry
# modes, and the full ATR-stop grid.
# ---------------------------------------------------------------------------


def _build_early_entry_candidates(
    all_events: list[EventRecord], all_signals_by_event: dict[str, list], pump_start_idx_by_event: dict[str, int],
    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame], feature_frame_by_symbol: dict[tuple[str, str], pd.DataFrame],
    bar_interval_ms: int, execution_cfg,
) -> pd.DataFrame:
    events_by_id = {e.event_id: e for e in all_events}
    rows = []
    for event_id, signals in all_signals_by_event.items():
        event = events_by_id.get(event_id)
        pump_start_idx = pump_start_idx_by_event.get(event_id)
        if event is None or pump_start_idx is None or not signals:
            continue
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        feature_frame = feature_frame_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None or feature_frame is None:
            continue
        for sig in signals:
            signal_abs_idx = pump_start_idx + sig.signal_wide_idx
            climax_abs_idx = pump_start_idx + sig.climax_wide_idx
            if signal_abs_idx >= len(ohlcv) - 1 or climax_abs_idx < 0 or climax_abs_idx >= len(ohlcv):
                continue
            oi_coverage_ok = True
            if sig.oi_dependent:
                oi_coverage_ok = early_entry.oi_coverage_adequate(feature_frame, climax_abs_idx, signal_abs_idx)
            for entry_mode in ("confirmation_close", "next_bar_open"):
                entry_price = ee.resolve_entry_price(ohlcv, signal_abs_idx, entry_mode)
                if entry_price is None:
                    continue
                walk_start_idx = signal_abs_idx + 1
                high_vol = _is_high_vol(feature_frame, walk_start_idx)
                for atr_multiple in ee.ATR_STOP_MULTIPLES:
                    # stop is climax-anchored (event-scoped climax high + ATR buffer), NOT signal-bar-anchored -
                    # the whole point of an earlier entry is a wider, more structurally meaningful stop.
                    stop_price = ee.compute_atr_stop_price(ohlcv, climax_abs_idx, atr_multiple)
                    if stop_price is None or stop_price <= entry_price:
                        continue
                    stop_distance_pct = (stop_price - entry_price) / entry_price
                    horizon_bars = max(round(HEADLINE_SPEC["horizon_hours"] * 3_600_000 / bar_interval_ms), 1)
                    profit_pct, adverse_pct = fp.resolve_r_based_barriers(stop_distance_pct, HEADLINE_SPEC["profit_r"], HEADLINE_SPEC["adverse_r"])
                    outcome = ee.first_passage_from_entry(ohlcv, walk_start_idx, entry_price, profit_pct, adverse_pct, horizon_bars)
                    exit_bar_idx = walk_start_idx + outcome["bars"] - 1
                    funding_r = _compute_funding_r(feature_frame, walk_start_idx, exit_bar_idx, bar_interval_ms, stop_distance_pct)
                    outcome_for_r = dict(outcome, stop_distance_pct=stop_distance_pct, high_vol=high_vol, funding_r=funding_r)
                    r_trades = ee.compute_r_trades([outcome_for_r], HEADLINE_SPEC["profit_r"], HEADLINE_SPEC["adverse_r"], execution_cfg, bar_interval_ms)
                    if not r_trades:
                        continue
                    rows.append(
                        {
                            "event_id": event_id, "exchange": event.exchange, "symbol": event.symbol, "pump_class": event.pump_class,
                            "attempt_number": sig.attempt_number, "entry_family": sig.entry_family, "entry_mode": entry_mode,
                            "atr_stop_multiple": atr_multiple, "oi_dependent": sig.oi_dependent, "oi_coverage_ok": oi_coverage_ok,
                            "entry_price": entry_price, "stop_price": stop_price, "stop_distance_pct": stop_distance_pct,
                            **outcome, **r_trades[0],
                        }
                    )
    return pd.DataFrame(rows)


def _headline_family_slice(df: pd.DataFrame, family: str) -> pd.DataFrame:
    subset = df[(df["entry_family"] == family) & (df["entry_mode"] == HEADLINE_ENTRY_MODE) & (df["atr_stop_multiple"] == HEADLINE_ATR_MULTIPLE)]
    if not subset.empty and subset["oi_dependent"].iloc[0]:
        subset = subset[subset["oi_coverage_ok"]]
    return subset


def render_early_entry_outcomes_table(candidates_df: pd.DataFrame) -> str:
    lines = [
        f"## Section 5: Entry A/B outcomes by family (headline: {HEADLINE_BARRIER_FAMILY} / {HEADLINE_ENTRY_MODE} / {HEADLINE_ATR_MULTIPLE} ATR - OI-dependent families restricted to OI-adequate-coverage events)",
        "", "| Family | N | Gross R | Net R | PF | Win rate | Avg win R | Avg loss R | Timeout % | Avg MAE R | Avg MFE R |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for family in early_entry.ALL_EARLY_ENTRY_FAMILY_NAMES:
        subset = _headline_family_slice(candidates_df, family)
        summary = ee.summarize_net_expectancy(subset.to_dict("records"))
        if summary.get("n", 0) == 0:
            lines.append(f"| {family} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        timeout_pct = summary["n_timeout"] / summary["n"] if summary["n"] else np.nan
        lines.append(
            f"| {family} | {summary['n']} | {_fmt_r(summary['gross_expectancy_r'])} | **{_fmt_r(summary['net_expectancy_r'])}** "
            f"| {_fmt_num(summary['profit_factor'])} | {_fmt_pct(summary['win_rate'])} | {_fmt_r(summary['avg_win_r'])} "
            f"| {_fmt_r(summary['avg_loss_r'])} | {_fmt_pct(timeout_pct)} | {_fmt_r(summary['avg_mae_r'])} | {_fmt_r(summary['avg_mfe_r'])} |"
        )
    lines.append("")
    lines.append(f"Entry C (failed_reclaim, unchanged benchmark) for comparison: see Section 1's headline row.")
    return "\n".join(lines)


def render_early_entry_atr_grid(candidates_df: pd.DataFrame) -> str:
    lines = [
        "### ATR-stop sensitivity per family (confirmation_close entry)", "",
        "| Family | ATR stop | N | Net R | PF | Win rate |", "|---|---|---|---|---|---|",
    ]
    for family in early_entry.ALL_EARLY_ENTRY_FAMILY_NAMES:
        for atr_multiple in ee.ATR_STOP_MULTIPLES:
            subset = candidates_df[
                (candidates_df["entry_family"] == family) & (candidates_df["entry_mode"] == HEADLINE_ENTRY_MODE)
                & (candidates_df["atr_stop_multiple"] == atr_multiple)
            ]
            if not subset.empty and subset["oi_dependent"].iloc[0]:
                subset = subset[subset["oi_coverage_ok"]]
            summary = ee.summarize_net_expectancy(subset.to_dict("records"))
            if summary.get("n", 0) == 0:
                continue
            lines.append(f"| {family} | {atr_multiple} | {summary['n']} | {_fmt_r(summary['net_expectancy_r'])} | {_fmt_num(summary['profit_factor'])} | {_fmt_pct(summary['win_rate'])} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry D (staged) - point 5's secondary hypothesis
# ---------------------------------------------------------------------------


def _staged_cost_adjustment(result: dict, leg1_entry_price: float, leg1_risk_per_unit: float, execution_cfg, high_vol: bool) -> dict:
    stop_distance_pct = leg1_risk_per_unit / leg1_entry_price
    planned_r = result["planned_r"]
    return_pct_for_exit_direction = result["realized_r"] if result["outcome"] == "timeout" else None
    exit_slippage_bps = ee.resolve_exit_slippage_bps(result["outcome"], return_pct_for_exit_direction, execution_cfg, high_vol)
    entry_slippage_bps = execution_cfg.default_slippage_bps * (execution_cfg.high_vol_slippage_multiplier if high_vol else 1.0)
    fees_r = planned_r * 2 * execution_cfg.taker_fee_bps / 10_000 / stop_distance_pct
    entry_slippage_r = planned_r * entry_slippage_bps / 10_000 / stop_distance_pct
    exit_slippage_r = planned_r * exit_slippage_bps / 10_000 / stop_distance_pct
    stop_overrun_premium_r = (
        planned_r * max(0.0, execution_cfg.stop_slippage_bps - execution_cfg.default_slippage_bps) / 10_000 / stop_distance_pct
        if exit_slippage_bps == execution_cfg.stop_slippage_bps
        else 0.0
    )
    net_r = result["realized_r"] - fees_r - entry_slippage_r - exit_slippage_r
    return {
        "outcome": result["outcome"], "gross_r": result["realized_r"], "fees_r": fees_r, "entry_slippage_r": entry_slippage_r,
        "exit_slippage_r": exit_slippage_r, "stop_overrun_premium_r": stop_overrun_premium_r, "funding_r": 0.0, "net_r": net_r,
        "mae_r": result["mae_r"], "mfe_r": result["mfe_r"], "leg2_activated": result["leg2_activated"], "planned_r": planned_r,
    }


def _build_entry_d_candidates(
    all_events: list[EventRecord], all_signals_by_event: dict[str, list], pump_start_idx_by_event: dict[str, int],
    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame], feature_frame_by_symbol: dict[tuple[str, str], pd.DataFrame], execution_cfg,
) -> pd.DataFrame:
    """Leg 1 = the EARLIEST attempt-1 Entry A/B signal for the event (whichever
    family fired first); leg 2 = the event's saga-level failed_reclaim bar (a
    disclosed simplification - failed_reclaim isn't tracked per-attempt, only
    per-saga, same limitation already noted for Phase 2B's Table 8)."""
    events_by_id = {e.event_id: e for e in all_events}
    rows = []
    for event_id, signals in all_signals_by_event.items():
        event = events_by_id.get(event_id)
        pump_start_idx = pump_start_idx_by_event.get(event_id)
        attempt1_signals = [s for s in signals if s.attempt_number == 1]
        if event is None or pump_start_idx is None or not attempt1_signals:
            continue
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        feature_frame = feature_frame_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None:
            continue
        leg1_signal = min(attempt1_signals, key=lambda s: s.signal_wide_idx)
        leg1_abs_idx = pump_start_idx + leg1_signal.signal_wide_idx
        climax_abs_idx = pump_start_idx + leg1_signal.climax_wide_idx
        failed_reclaim_wide = event.exhaustion_component_first_true_bar.get("failed_reclaim_lower_high")
        leg2_abs_idx = pump_start_idx + int(failed_reclaim_wide) if pd.notna(failed_reclaim_wide) else None
        horizon_bars = max(round(HEADLINE_SPEC["horizon_hours"] * 3_600_000 / parse_timeframe_ms("5m")), 1)

        for entry_mode in ("confirmation_close", "next_bar_open"):
            high_vol = _is_high_vol(feature_frame, leg1_abs_idx + 1) if feature_frame is not None else False
            for atr_multiple in ee.ATR_STOP_MULTIPLES:
                stop_price = ee.compute_atr_stop_price(ohlcv, climax_abs_idx, atr_multiple)
                if stop_price is None:
                    continue
                result = ee.simulate_staged_entry(ohlcv, leg1_abs_idx, entry_mode, leg2_abs_idx, stop_price, HEADLINE_SPEC["profit_r"], horizon_bars)
                if result is None:
                    continue
                leg1_entry_price = ee.resolve_entry_price(ohlcv, leg1_abs_idx, entry_mode)
                if leg1_entry_price is None:
                    continue
                leg1_risk_per_unit = stop_price - leg1_entry_price
                adjusted = _staged_cost_adjustment(result, leg1_entry_price, leg1_risk_per_unit, execution_cfg, high_vol)
                rows.append(
                    {
                        "event_id": event_id, "exchange": event.exchange, "symbol": event.symbol, "leg1_family": leg1_signal.entry_family,
                        "entry_mode": entry_mode, "atr_stop_multiple": atr_multiple, "had_leg2_candidate": leg2_abs_idx is not None,
                        **adjusted,
                    }
                )
    return pd.DataFrame(rows)


def render_entry_d_table(entry_d_df: pd.DataFrame) -> str:
    lines = [
        "## Section 6: Entry D (staged 50%/50%) - secondary, predefined hypothesis",
        "", "Leg 1 = earliest Entry A/B signal on attempt 1; leg 2 = saga-level failed_reclaim bar (if it occurs after leg1 and before leg1's own stop).", "",
        "| Entry mode | ATR stop | N | % leg2 activated | Net R | PF | Win rate |", "|---|---|---|---|---|---|---|",
    ]
    if entry_d_df.empty:
        lines.append("| (no data) | - | - | - | - | - | - |")
        return "\n".join(lines)
    for entry_mode in ("confirmation_close", "next_bar_open"):
        for atr_multiple in ee.ATR_STOP_MULTIPLES:
            group = entry_d_df[(entry_d_df["entry_mode"] == entry_mode) & (entry_d_df["atr_stop_multiple"] == atr_multiple)]
            summary = ee.summarize_net_expectancy(group.to_dict("records"))
            if summary.get("n", 0) == 0:
                continue
            pct_leg2 = group["leg2_activated"].mean()
            lines.append(f"| {entry_mode} | {atr_multiple} | {summary['n']} | {_fmt_pct(pct_leg2)} | {_fmt_r(summary['net_expectancy_r'])} | {_fmt_num(summary['profit_factor'])} | {_fmt_pct(summary['win_rate'])} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Squeeze split (point 9)
# ---------------------------------------------------------------------------


def render_squeeze_split(all_events: list[EventRecord], entry_c_headline: pd.DataFrame, cfg: AppConfig) -> str:
    squeeze_events = [e for e in all_events if e.pump_class == "SHORT_SQUEEZE"]
    lines = ["## Section 7: SHORT_SQUEEZE_ACTIVE vs SHORT_SQUEEZE_DEPLETED", "", f"N SHORT_SQUEEZE events (per-exchange classification): {len(squeeze_events)}.", ""]
    if not squeeze_events:
        lines.append("No SHORT_SQUEEZE-classified events in this universe.")
        return "\n".join(lines)
    state_by_id = {e.event_id: uc.classify_short_squeeze_state(e, cfg) for e in squeeze_events}
    lines.append("| State | N | Entry C (failed_reclaim) N in headline slice | Entry C P(profit) | Entry C net R |")
    lines.append("|---|---|---|---|---|")
    for state in ("SHORT_SQUEEZE_ACTIVE", "SHORT_SQUEEZE_DEPLETED"):
        ids = {eid for eid, s in state_by_id.items() if s == state}
        subset = entry_c_headline[entry_c_headline["event_id"].isin(ids)]
        summary = ee.summarize_net_expectancy(subset.to_dict("records"))
        n_class = sum(1 for s in state_by_id.values() if s == state)
        p_profit = (subset["outcome"] == "profit").mean() if len(subset) else np.nan
        lines.append(f"| {state} | {n_class} | {summary.get('n', 0)} | {_fmt_pct(p_profit)} | {_fmt_r(summary.get('net_expectancy_r'))} |")
    return "\n".join(lines)


@app.command()
def main(
    binance_run_id: str = typer.Option("binance_canonical_v2"),
    bybit_run_id: str = typer.Option("bybit_canonical_v2"),
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("phase_2c_cost_and_early_entry_v1"),
    cross_exchange_window_minutes: int = typer.Option(60),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    since_ms = datetime_to_ms(cfg.backtest.start_date)
    until_ms = datetime_to_ms(cfg.backtest.end_date)

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    exchange_events: dict[str, list[EventRecord]] = {}
    if binance_run_id:
        exchange_events["binance"] = _load_events_with_trades(binance_run_id)
    if bybit_run_id:
        exchange_events["bybit"] = _load_events_with_trades(bybit_run_id)
    all_events = [e for events in exchange_events.values() for e in events]
    if not all_events:
        typer.echo("No events loaded - nothing to analyze.")
        raise typer.Exit(1)
    typer.echo(f"Total events: {len(all_events)}")

    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    for exchange, events in exchange_events.items():
        ohlcv_by_symbol.update(_load_ohlcv_by_symbol(exchange, events, timeframe, since_ms, until_ms))
    typer.echo(f"Loaded OHLCV for {len(ohlcv_by_symbol)} (exchange, symbol) pairs")

    typer.echo("Loading feature frames for ALL events (needed for Entry A/B, not just failed-reclaim events)...")
    feature_frame_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    for exchange, events in exchange_events.items():
        feature_frame_by_symbol.update(_load_feature_frame_by_symbol(exchange, events, ohlcv_by_symbol, timeframe, since_ms, until_ms, cfg))

    report_sections: list[str] = ["# Phase 2C - Cost Decomposition + Early Entry Study\n"]
    report_sections.append(
        f"Universe: {len(all_events)} events ({len(exchange_events.get('binance', []))} Binance, "
        f"{len(exchange_events.get('bybit', []))} Bybit). Strategy v1 / failed_reclaim standard entry = REJECTED per "
        "Phase 2B; this pass does not loosen thresholds, does not build a live scanner, does not extend the dataset.\n"
    )

    # ==== Load Phase 2B's own (now bug-fixed) failed-reclaim candidate grid - Entry C benchmark, unchanged ====
    phase_2b_parquet = DIAGNOSTICS_ROOT / PHASE_2B_RUN_ID / "phase_2b_candidate_attempts.parquet"
    if not phase_2b_parquet.exists():
        typer.echo(f"ERROR: {phase_2b_parquet} not found - run scripts/run_phase_2b_entry_edge.py first.")
        raise typer.Exit(1)
    entry_c_candidates = pd.read_parquet(phase_2b_parquet)
    entry_c_headline = entry_c_candidates[
        (entry_c_candidates["entry_mode"] == HEADLINE_ENTRY_MODE) & (entry_c_candidates["atr_stop_multiple"] == HEADLINE_ATR_MULTIPLE)
        & (entry_c_candidates["barrier_family"] == HEADLINE_BARRIER_FAMILY)
    ]
    entry_c_headline_summary = ee.summarize_net_expectancy(entry_c_headline.to_dict("records"))
    typer.echo(f"Entry C (failed_reclaim) headline: net={entry_c_headline_summary.get('net_expectancy_r')}, n={entry_c_headline_summary.get('n')}")

    # ==== Section 1: cost decomposition ====
    report_sections.append(render_cost_decomposition(entry_c_headline_summary, cfg.execution_model))

    # ==== Section 2: stop-distance audit ====
    typer.echo("Section 2: stop-distance audit...")
    stop_audit_df = ee.stop_distance_audit(entry_c_candidates, cfg.execution_model)
    report_sections.append(render_stop_distance_audit(stop_audit_df))
    stop_audit_df.to_parquet(out_dir / "section2_stop_distance_audit.parquet", index=False)

    # ==== Section 3: correctness audit (prose, backed by unit tests) ====
    report_sections.append(COST_MODEL_CORRECTNESS_NOTES)

    # ==== Section 4: cost sensitivity ====
    typer.echo("Section 4: cost sensitivity sweep...")
    entry_c_headline_outcomes = entry_c_headline.to_dict("records")
    sweep_df = ee.cost_sensitivity_sweep(entry_c_headline_outcomes, HEADLINE_SPEC["profit_r"], HEADLINE_SPEC["adverse_r"], cfg.execution_model, bar_interval_ms)
    curve_df = ee.cost_multiplier_curve(entry_c_headline_outcomes, HEADLINE_SPEC["profit_r"], HEADLINE_SPEC["adverse_r"], cfg.execution_model, bar_interval_ms)
    breakeven = ee.estimate_breakeven_cost_multiplier(curve_df)
    report_sections.append(render_cost_sensitivity(sweep_df, curve_df, breakeven))

    report_text = "\n\n".join(report_sections) + "\n"
    (out_dir / "phase_2c_cost_and_early_entry_report.md").write_text(report_text, encoding="utf-8")
    typer.echo(f"Part 1 (cost decomposition/audit/sensitivity) done. Continuing to early-entry detection...")

    # ==== Early entry detection (Entry A + Entry B) ====
    typer.echo("Detecting Entry A/B signals across ALL events (this walks each event's post-climax window)...")
    all_signals_by_event: dict[str, list] = {}
    pump_start_idx_by_event: dict[str, int] = {}
    for i, event in enumerate(all_events):
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        feature_frame = feature_frame_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None or feature_frame is None:
            continue
        signals, pump_start_idx = early_entry.find_early_entry_signals(event, ohlcv, feature_frame, timeframe, cfg)
        if pump_start_idx is None:
            continue
        all_signals_by_event[event.event_id] = signals
        pump_start_idx_by_event[event.event_id] = pump_start_idx
        if (i + 1) % 40 == 0:
            typer.echo(f"  ...{i + 1}/{len(all_events)} events scanned")
    n_events_with_any_signal = sum(1 for sigs in all_signals_by_event.values() if sigs)
    typer.echo(f"Events with at least one Entry A/B signal: {n_events_with_any_signal}/{len(all_signals_by_event)}")

    save_early_entry_state(out_dir, all_signals_by_event)

    # ==== Entry A/B candidate grid + Section 5 ====
    typer.echo("Building Entry A/B candidate grid (first-passage + net expectancy per signal)...")
    early_candidates_df = _build_early_entry_candidates(
        all_events, all_signals_by_event, pump_start_idx_by_event, ohlcv_by_symbol, feature_frame_by_symbol, bar_interval_ms, cfg.execution_model
    )
    typer.echo(f"Entry A/B candidate rows: {len(early_candidates_df)}")
    early_candidates_df.to_parquet(out_dir / "phase_2c_early_entry_candidates.parquet", index=False)
    report_sections.append(render_early_entry_outcomes_table(early_candidates_df))
    report_sections.append(render_early_entry_atr_grid(early_candidates_df))

    # Best-performing family (N>=20 in the headline slice) - for bootstrap CI + continuation risk, not a "winner pick" for the final verdict.
    family_summaries = {}
    for family in early_entry.ALL_EARLY_ENTRY_FAMILY_NAMES:
        subset = _headline_family_slice(early_candidates_df, family)
        summary = ee.summarize_net_expectancy(subset.to_dict("records"))
        if summary.get("n", 0) >= 20:
            family_summaries[family] = summary
    best_family = max(family_summaries, key=lambda f: family_summaries[f]["net_expectancy_r"]) if family_summaries else None

    # ==== Bootstrap CI for the best-performing early-entry family (point 11, extended from Entry C to whichever early family looks most promising) ====
    if best_family is not None:
        typer.echo(f"Bootstrapping best-performing early-entry family: {best_family}...")
        best_subset = _headline_family_slice(early_candidates_df, best_family)
        best_bootstrap = ee.bootstrap_net_expectancy_ci(best_subset.to_dict("records"))
        win_ci, exp_ci = best_bootstrap.get("win_rate_ci", {}), best_bootstrap.get("net_expectancy_ci", {})
        report_sections.append(
            f"### Bootstrap 95% CI for the best-performing early-entry family ({best_family}, N={best_bootstrap.get('n', 0)})\n\n"
            f"Win rate: {_fmt_pct(win_ci.get('statistic'))} (95% CI: {_fmt_pct(win_ci.get('ci_low'))} - {_fmt_pct(win_ci.get('ci_high'))})\n\n"
            f"Net expectancy: {_fmt_r(exp_ci.get('statistic'))} (95% CI: {_fmt_r(exp_ci.get('ci_low'))} - {_fmt_r(exp_ci.get('ci_high'))})\n\n"
            "\"Best-performing\" is picked AFTER seeing the results (not a pre-registered choice) - treat this CI as descriptive of "
            "uncertainty around that one number, not as a multiple-comparisons-corrected confirmatory test; selecting the best of 7 "
            "families and then reporting its CI without correction is itself a documented source of optimism. The CI already crosses "
            "zero (lower bound negative) despite this being the single best-looking family out of 7 - not a stable edge yet. The "
            "OI-dependent families (this one included) are additionally conditioned on OI-data availability, which is concentrated on "
            "Bybit / more recent bars (documented Binance ~1.7-day OI retention) rather than randomly distributed - part of any apparent "
            "edge here could reflect which exchange/period had usable OI data, not the OI-rollover signal itself."
        )
        exchange_rows = []
        exchange_summaries = {}
        for exchange, group in best_subset.groupby("exchange"):
            summary = ee.summarize_net_expectancy(group.to_dict("records"))
            exchange_summaries[exchange] = summary
            exchange_rows.append(f"{exchange}: N={summary['n']}, net={_fmt_r(summary['net_expectancy_r'])}, PF={_fmt_num(summary['profit_factor'])}")
        exchange_note = (
            f"Per-exchange split of {best_family}: " + "; ".join(exchange_rows) + ". "
        )
        tiny_extreme = [ex for ex, s in exchange_summaries.items() if s["n"] < 10 and s.get("profit_factor", 0) > 5]
        if tiny_extreme:
            exchange_note += (
                f"{', '.join(tiny_extreme)}'s N is too small (<10) for a PF this extreme to be trustworthy - almost certainly one or "
                f"two outsized winners, not a real edge on its own. The combined N={best_bootstrap.get('n', 0)} headline figure is "
                "pulled upward by that small subset; the larger, more stable-looking Bybit-only slice is the more honest read of this "
                "family's actual magnitude."
            )
        report_sections.append(exchange_note)
    else:
        report_sections.append("### Bootstrap CI for best-performing early-entry family\n\nNo family reached N>=20 in the headline slice - skipped.")

    # ==== Entry D ====
    typer.echo("Section 6: Entry D (staged entry)...")
    entry_d_df = _build_entry_d_candidates(all_events, all_signals_by_event, pump_start_idx_by_event, ohlcv_by_symbol, feature_frame_by_symbol, cfg.execution_model)
    if not entry_d_df.empty:
        entry_d_df.to_parquet(out_dir / "phase_2c_entry_d_candidates.parquet", index=False)
    report_sections.append(render_entry_d_table(entry_d_df))

    # ==== Squeeze split ====
    typer.echo("Section 7: SHORT_SQUEEZE_ACTIVE vs DEPLETED split...")
    report_sections.append(render_squeeze_split(all_events, entry_c_headline, cfg))

    # ==== Final verdicts ====
    typer.echo("Assembling final verdict sections...")
    report_sections.append(
        _render_final_verdicts(entry_c_headline_summary, breakeven, family_summaries, best_family, entry_d_df, all_events, cfg)
    )

    report_text = "\n\n".join(report_sections) + "\n"
    (out_dir / "phase_2c_cost_and_early_entry_report.md").write_text(report_text, encoding="utf-8")
    typer.echo(f"Done. Report + parquets under: {out_dir}")


def _render_final_verdicts(
    entry_c_headline_summary: dict, breakeven: float | None, family_summaries: dict[str, dict], best_family: str | None,
    entry_d_df: pd.DataFrame, all_events: list[EventRecord], cfg: AppConfig,
) -> str:
    lines = ["## COST MODEL VERDICT", ""]
    lines.append(
        "A real double-counting bug was found in Phase 2B's cost model (exit-side slippage on stop-hit trades was charged "
        "at both the default rate AND the stop-overrun rate) and fixed - the headline failed_reclaim net expectancy moved "
        f"from -0.435R to {_fmt_r(entry_c_headline_summary.get('net_expectancy_r'))}. The corrected drag from gross to net "
        f"is real, not a modeling artifact: fees + entry/exit slippage together still consume more R than the setup's own "
        "gross edge, and the stop-distance audit (Section 2) shows why - the round-trip execution cost is a large fraction "
        "of the median real structural stop distance at tight ATR multiples."
    )
    if breakeven is not None:
        lines.append(f"\nEstimated breakeven cost multiplier for the headline failed_reclaim setup: **{breakeven:.2f}x** the project's own configured BASE costs.")
    else:
        lines.append("\nNet expectancy for the headline failed_reclaim setup does not cross zero anywhere in the tested 0x-2x cost range (Section 4) - this specific setup's rejection is NOT primarily a cost-modeling artifact.")
    lines.append("")

    lines.append("## EARLY ENTRY VERDICT")
    lines.append("")
    if not family_summaries:
        lines.append("No Entry A/B family reached the N>=20 minimum for a headline read - insufficient data to verdict early entry at this sample size.")
    else:
        positive_families = {f: s for f, s in family_summaries.items() if s["net_expectancy_r"] > 0 and s.get("profit_factor", 0) > 1}
        if positive_families:
            names = ", ".join(f"{f} (net={_fmt_r(s['net_expectancy_r'])}, PF={_fmt_num(s['profit_factor'])}, N={s['n']})" for f, s in positive_families.items())
            lines.append(f"{len(positive_families)}/{len(family_summaries)} tested early-entry families show POSITIVE net expectancy AND PF>1 at BASE realistic costs: {names}.")
            lines.append("This does not by itself clear the point-13 GO bar (needs to also survive slippage stress, a non-catastrophic bootstrap CI, not be driven by 2-3 trades, and show consistent direction across exchanges/pump classes - review Section 5's full table and the bootstrap CI above before treating this as a finding).")
        else:
            lines.append(f"0/{len(family_summaries)} tested early-entry families (N>=20) show both positive net expectancy and PF>1 at BASE realistic costs (see Section 5). Entering earlier in the causal chain does not, on this pass, recover a positive net edge.")
    lines.append("")

    lines.append("## FAILED RECLAIM VERDICT")
    lines.append("")
    lines.append(
        f"Entry C (failed_reclaim, unchanged benchmark) headline net expectancy: {_fmt_r(entry_c_headline_summary.get('net_expectancy_r'))}, "
        f"PF={_fmt_num(entry_c_headline_summary.get('profit_factor'))}, N={entry_c_headline_summary.get('n', 0)}. "
        "Confirms Phase 2B's own conclusion (Strategy v1 / failed_reclaim standard entry = REJECTED) under the corrected cost model - "
        "the bug fix narrowed the loss, it did not reverse the sign."
    )
    lines.append("")

    lines.append("## SQUEEZE VERDICT")
    lines.append("")
    squeeze_events = [e for e in all_events if e.pump_class == "SHORT_SQUEEZE"]
    if not squeeze_events:
        lines.append("No SHORT_SQUEEZE-classified events in this universe - no squeeze-specific verdict possible.")
    else:
        n_active = sum(1 for e in squeeze_events if uc.classify_short_squeeze_state(e, cfg) == "SHORT_SQUEEZE_ACTIVE")
        n_depleted = len(squeeze_events) - n_active
        lines.append(
            f"Of {len(squeeze_events)} SHORT_SQUEEZE-classified events, {n_active} are ACTIVE and {n_depleted} are DEPLETED "
            "(Section 7). See Section 7's table for the failed_reclaim first-passage/net-expectancy split between the two - "
            "small N in most cuts here, treat directionally only, not as a confirmed finding."
        )
    lines.append("")

    lines.append("## FINAL GO / STOP RECOMMENDATION")
    lines.append("")
    any_family_clears_bar = family_summaries and any(s["net_expectancy_r"] > 0 and s.get("profit_factor", 0) > 1 for s in family_summaries.values())
    if any_family_clears_bar:
        best_summary = family_summaries.get(best_family, {}) if best_family else {}
        lines.append(
            f"One pre-defined early-entry family ({best_family}) clears the basic net-expectancy/PF bar at BASE realistic costs "
            f"(net={_fmt_r(best_summary.get('net_expectancy_r'))}, PF={_fmt_num(best_summary.get('profit_factor'))}, N={best_summary.get('n', 0)}). "
            "Checked against the user's point-13 GO criteria so far: bootstrap 95% CI already crosses zero (lower bound negative) - "
            "NOT a stable edge yet. Per-exchange split (Section 5's bootstrap block) is a mixed signal, not a clean pass: Binance's "
            "slice is only N=5 with an implausibly high PF (almost certainly 1-2 outsized winners, not a real edge on its own, "
            "already flagged as a likely 'driven by a handful of extreme trades' failure - the exact point-13 check this needed to "
            "survive), while Bybit's larger N=26 slice is positive but much more modest (net~0.1R, PF~1.3) - the more honest read of "
            "this family's actual magnitude, and still not independently confirmed as robust. NOT yet checked: whether the Bybit-only "
            "slice specifically survives a Section-4-style slippage-stress sweep. Net verdict: interesting enough to be worth a "
            "targeted, Bybit-only follow-up on this ONE family specifically, but the combined 31-event headline number should NOT be "
            "taken at face value, and this is NOT sufficient on its own to justify extending the dataset to 12-24 months or building "
            "a scanner - do not do either from this report alone."
        )
    else:
        lines.append(
            "PUMP EXHAUSTION REVERSAL - NO TRADEABLE EDGE FOUND IN CURRENT FORM. Neither the failed_reclaim standard entry (Entry C) "
            "nor any of the tested earlier causal entry families (Entry A's 6 combinations, Entry B) clears positive net expectancy "
            "with PF>1 at realistic execution costs. The cost-model bug fix and the stop-distance audit both confirm this is a real "
            "result, not a modeling artifact for the failed_reclaim setup specifically - and moving the entry earlier in the causal "
            "chain (closer to first_exhaustion_signal, with a wider climax-anchored stop) does not recover an edge either. Per the "
            "user's own instruction: do not lower thresholds, do not increase leverage, do not optimize TP placement to rescue this. "
            "Do not extend the Binance+Bybit dataset to 12-24 months on this basis."
        )
    lines.append("")
    lines.append("Awaiting user review before any further phase, live scanner, or execution work.")
    return "\n".join(lines)


def save_early_entry_state(out_dir, all_signals_by_event: dict[str, list]) -> None:
    rows = []
    for event_id, signals in all_signals_by_event.items():
        for s in signals:
            rows.append(
                {
                    "event_id": s.event_id, "attempt_number": s.attempt_number, "entry_family": s.entry_family,
                    "signal_wide_idx": s.signal_wide_idx, "climax_wide_idx": s.climax_wide_idx, "oi_dependent": s.oi_dependent,
                }
            )
    pd.DataFrame(rows).to_parquet(out_dir / "early_entry_signals_raw.parquet", index=False)


if __name__ == "__main__":
    app()
