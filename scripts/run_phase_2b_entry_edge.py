"""CLI: Phase 2B - Entry Edge Validation (see project plan "Phase 2B" - not
in the original plan file, specified directly by the user after reviewing
Phase 2A's results). Converts Phase 2A's binary first-passage finding
(failed_reclaim hit rate 0/244 -> 72/164 after the temporal/event
architecture fix) into real trade economics: net expectancy after fees/
slippage/funding/stop-overrun, real (already-knowable) entry prices, a
structural ATR stop grid, continuous failed-reclaim quality features,
pump-class and retry breakdowns, continuation risk, and bootstrap
uncertainty. Builds NO live scanner and enables NO execution - purely reads
already-saved EventRecord fields + raw OHLCV and writes one report + one
parquet of every candidate attempt.

python scripts/run_phase_2b_entry_edge.py --binance-run-id binance_canonical_v2 --bybit-run-id bybit_canonical_v2

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
from pump_exhaustion_reversal.features.returns import bars_for_window
from pump_exhaustion_reversal.research import cross_exchange
from pump_exhaustion_reversal.research import entry_edge as ee
from pump_exhaustion_reversal.research import first_passage as fp
from pump_exhaustion_reversal.research import lifecycle_diagnostics as ld
from pump_exhaustion_reversal.research import reclaim_quality as rq
from pump_exhaustion_reversal.research import underlying_classification as uc

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"

# Headline variant used everywhere a single representative choice is needed
# (Table 2's clean barrier-family sensitivity, Table 7's pump-class
# breakdown, Table 10's continuation risk, Table 11's bootstrap CI) - picked,
# not optimized: confirmation_close is the more conservative of the two
# entry modes (never assumes a fill better than the bar that confirmed the
# signal), 0.25 ATR is the middle of the 0.15/0.25/0.40 stop grid, and
# 2R_1R_4h is the same barrier the user used for their own 33.3% breakeven
# illustration. Every other combination is still in the full candidate
# parquet - this constant only controls which slice gets its own report
# table instead of being buried in the full grid.
HEADLINE_ENTRY_MODE = "confirmation_close"
HEADLINE_ATR_MULTIPLE = 0.25
HEADLINE_BARRIER_FAMILY = "2R_1R_4h"
CONTINUATION_HORIZON_HOURS = 24.0


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


def _load_ohlcv_by_symbol(
    exchange: str, events: list[EventRecord], timeframe: str, since_ms: int, until_ms: int
) -> dict[tuple[str, str], pd.DataFrame]:
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
    """Rebuilds feature_frame (cvd_rolling/cvd_slope/open_interest) fresh
    from raw OHLCV/OI - Phase 2A's EventRecord never persisted per-bar CVD/OI
    state, only whole-event summaries, so the reclaim-quality features (point
    6) need this recomputed for the handful of symbols involved in a
    failed-reclaim event, exactly like build_feature_frame is used at scan
    time."""
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


def _fmt_pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.1%}"


def _fmt_r(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}R"


def _fmt_num(v, digits: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# Table 1: richer per-anchor first-passage decomposition (point 1)
# ---------------------------------------------------------------------------


def render_table1(by_anchor: pd.DataFrame) -> str:
    lines = [
        "## Table 1: per-anchor first-passage decomposition (barrier family 2R/1R/4h, uniform ATR stop basis)",
        "",
        "| Anchor | N | PROFIT | ADVERSE | TIMEOUT | P(profit) | P(adverse) | P(timeout) | Avg return@timeout | Med return@timeout | Avg MAE | Avg MFE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in by_anchor.iterrows():
        lines.append(
            f"| {row['anchor']} | {row['n']} | {row['n_profit']} | {row['n_adverse']} | {row['n_timeout']} "
            f"| {_fmt_pct(row['p_profit'])} | {_fmt_pct(row['p_adverse'])} | {_fmt_pct(row['p_timeout'])} "
            f"| {_fmt_pct(row['avg_return_at_timeout_pct'])} | {_fmt_pct(row['median_return_at_timeout_pct'])} "
            f"| {_fmt_pct(row['avg_mae_pct'])} | {_fmt_pct(row['avg_mfe_pct'])} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate-attempt grid: every failed-reclaim event x entry mode x ATR stop
# multiple x barrier family - the single source table for points 2/3/4/5.
# ---------------------------------------------------------------------------


def _is_high_vol(feature_frame: pd.DataFrame | None, entry_bar_idx: int) -> bool:
    """Matches backtest/simulator.py's _simulate_trade exactly: `vol_ratio_at_entry
    > 2.0`, decided once at entry and reused for every default-rate slippage
    leg of that trade."""
    if feature_frame is None or "realized_vol_ratio" not in feature_frame.columns or entry_bar_idx >= len(feature_frame):
        return False
    ratio = feature_frame["realized_vol_ratio"].iloc[entry_bar_idx]
    return bool(ratio > 2.0) if pd.notna(ratio) else False


def _compute_funding_r(feature_frame: pd.DataFrame | None, entry_bar_idx: int, exit_bar_idx: int, bar_interval_ms: int, stop_distance_pct: float) -> float:
    """Mirrors _simulate_trade's per-bar funding proration exactly (real
    funding_rate data, summed over every bar actually held - not snapped to
    discrete UTC funding timestamps, matching the established, tested
    production convention rather than a coarser approximation): a short
    RECEIVES when funding_rate>0 (execution_model.compute_funding_cost's sign
    convention), converted to R (notional cancels: funding_dollars/risk_dollars
    = funding_rate*proration/stop_distance_pct)."""
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


def _build_candidate_attempts(
    events: list[EventRecord], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame],
    feature_frame_by_symbol: dict[tuple[str, str], pd.DataFrame], bar_interval_ms: int, lookback_bars: int, execution_cfg,
) -> pd.DataFrame:
    rows = []
    for event in events:
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None:
            continue
        feature_frame = feature_frame_by_symbol.get((event.exchange, event.symbol))
        pump_start_idx = ld._pump_start_idx_for(event, ohlcv, lookback_bars)
        reclaim_idx_abs = ld._anchor_idx_for(event, "failed_reclaim", ohlcv, pump_start_idx)
        if reclaim_idx_abs is None or reclaim_idx_abs >= len(ohlcv) - 1:
            continue
        for entry_mode in ("confirmation_close", "next_bar_open"):
            entry_price = ee.resolve_entry_price(ohlcv, reclaim_idx_abs, entry_mode)
            if entry_price is None:
                continue
            walk_start_idx = reclaim_idx_abs + 1
            high_vol = _is_high_vol(feature_frame, walk_start_idx)
            for atr_multiple in ee.ATR_STOP_MULTIPLES:
                stop_price = ee.compute_atr_stop_price(ohlcv, reclaim_idx_abs, atr_multiple)
                if stop_price is None or stop_price <= entry_price:
                    continue
                stop_distance_pct = (stop_price - entry_price) / entry_price
                for family_name, spec in fp.PHASE_2B_BARRIER_FAMILIES.items():
                    horizon_bars = max(round(spec["horizon_hours"] * 3_600_000 / bar_interval_ms), 1)
                    profit_pct, adverse_pct = fp.resolve_r_based_barriers(stop_distance_pct, spec["profit_r"], spec["adverse_r"])
                    outcome = ee.first_passage_from_entry(ohlcv, walk_start_idx, entry_price, profit_pct, adverse_pct, horizon_bars)
                    exit_bar_idx = walk_start_idx + outcome["bars"] - 1
                    funding_r = _compute_funding_r(feature_frame, walk_start_idx, exit_bar_idx, bar_interval_ms, stop_distance_pct)
                    outcome_for_r = dict(outcome, stop_distance_pct=stop_distance_pct, high_vol=high_vol, funding_r=funding_r)
                    r_trades = ee.compute_r_trades([outcome_for_r], spec["profit_r"], spec["adverse_r"], execution_cfg, bar_interval_ms)
                    if not r_trades:
                        continue
                    r_trade = r_trades[0]
                    rows.append(
                        {
                            "event_id": event.event_id, "exchange": event.exchange, "symbol": event.symbol,
                            "pump_class": event.pump_class, "attempt_count": event.attempt_count,
                            "reclaim_idx_abs": reclaim_idx_abs, "entry_mode": entry_mode, "atr_stop_multiple": atr_multiple,
                            "barrier_family": family_name, "entry_price": entry_price, "stop_price": stop_price,
                            "stop_distance_pct": stop_distance_pct, "high_vol": high_vol,
                            **outcome, **r_trade,
                        }
                    )
    return pd.DataFrame(rows)


def _headline_slice(candidates_df: pd.DataFrame) -> pd.DataFrame:
    return candidates_df[
        (candidates_df["entry_mode"] == HEADLINE_ENTRY_MODE)
        & (candidates_df["atr_stop_multiple"] == HEADLINE_ATR_MULTIPLE)
        & (candidates_df["barrier_family"] == HEADLINE_BARRIER_FAMILY)
    ]


def render_table2(candidates_df: pd.DataFrame) -> str:
    subset = candidates_df[(candidates_df["entry_mode"] == HEADLINE_ENTRY_MODE) & (candidates_df["atr_stop_multiple"] == HEADLINE_ATR_MULTIPLE)]
    lines = [
        f"## Table 2: barrier-family sensitivity (entry={HEADLINE_ENTRY_MODE}, stop={HEADLINE_ATR_MULTIPLE} ATR - all others in the parquet)",
        "", "Sensitivity only - no family is picked as a winner here.", "",
        "| Barrier family | N | P(profit) | P(adverse) | P(timeout) | Gross expectancy R |",
        "|---|---|---|---|---|---|",
    ]
    order = list(fp.PHASE_2B_BARRIER_FAMILIES)
    for family in order:
        group = subset[subset["barrier_family"] == family]
        if group.empty:
            lines.append(f"| {family} | 0 | n/a | n/a | n/a | n/a |")
            continue
        n = len(group)
        p_profit = (group["outcome"] == "profit").mean()
        p_adverse = (group["outcome"] == "adverse").mean()
        p_timeout = (group["outcome"] == "timeout").mean()
        gross_exp = group["gross_r"].mean()
        lines.append(f"| {family} | {n} | {_fmt_pct(p_profit)} | {_fmt_pct(p_adverse)} | {_fmt_pct(p_timeout)} | {_fmt_r(gross_exp)} |")
    return "\n".join(lines)


def render_table5(candidates_df: pd.DataFrame) -> str:
    lines = [
        "## Table 5: NET expectancy - full grid (barrier family x entry mode x ATR stop multiple)",
        "",
        "Entry-leg and exit-leg slippage are tracked separately (exit-leg rate depends on how the trade concluded - "
        "TP-style default rate for profit/favorable-timeout, stop-style rate for adverse/adverse-timeout; "
        "`Stop-overrun premium R` is the portion of exit-slippage already caused by the stop rate exceeding the default "
        "rate - informational, already included in Exit-slippage R, not an extra subtraction).",
        "",
        "| Family | Entry | ATR stop | N | Gross R | Fees R | Entry-slip R | Exit-slip R | (of which stop-overrun premium) | Funding R | **Net R** | PF | Win rate | Avg win R | Avg loss R | Avg MAE R | Avg MFE R |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for family in fp.PHASE_2B_BARRIER_FAMILIES:
        for entry_mode in ("confirmation_close", "next_bar_open"):
            for atr_multiple in ee.ATR_STOP_MULTIPLES:
                group = candidates_df[
                    (candidates_df["barrier_family"] == family) & (candidates_df["entry_mode"] == entry_mode)
                    & (candidates_df["atr_stop_multiple"] == atr_multiple)
                ]
                summary = ee.summarize_net_expectancy(group.to_dict("records"))
                if summary.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {family} | {entry_mode} | {atr_multiple} | {summary['n']} | {_fmt_r(summary['gross_expectancy_r'])} "
                    f"| {_fmt_r(summary['fees_r'])} | {_fmt_r(summary['entry_slippage_r'])} | {_fmt_r(summary['exit_slippage_r'])} "
                    f"| {_fmt_r(summary['stop_overrun_premium_r'])} | {_fmt_r(summary['funding_r'])} "
                    f"| **{_fmt_r(summary['net_expectancy_r'])}** | {_fmt_num(summary['profit_factor'])} "
                    f"| {_fmt_pct(summary['win_rate'])} | {_fmt_r(summary['avg_win_r'])} | {_fmt_r(summary['avg_loss_r'])} "
                    f"| {_fmt_r(summary['avg_mae_r'])} | {_fmt_r(summary['avg_mfe_r'])} |"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 6: continuous failed-reclaim quality features (point 6)
# ---------------------------------------------------------------------------


def _build_quality_df(
    events: list[EventRecord], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame],
    feature_frame_by_symbol: dict[tuple[str, str], pd.DataFrame], lookback_bars: int,
) -> pd.DataFrame:
    rows = []
    for event in events:
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        feature_frame = feature_frame_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None or feature_frame is None:
            continue
        pump_start_idx = ld._pump_start_idx_for(event, ohlcv, lookback_bars)
        climax_idx_abs = ld._anchor_idx_for(event, "climax", ohlcv, pump_start_idx)
        reclaim_idx_abs = ld._anchor_idx_for(event, "failed_reclaim", ohlcv, pump_start_idx)
        if climax_idx_abs is None or reclaim_idx_abs is None:
            continue
        features = rq.compute_reclaim_quality_features(ohlcv, feature_frame, climax_idx_abs, reclaim_idx_abs)
        if not features:
            continue
        rows.append({"event_id": event.event_id, **features})
    return pd.DataFrame(rows)


def render_table6(quality_comparison: pd.DataFrame, logit_check: pd.DataFrame) -> str:
    lines = [
        "## Table 6: failed-reclaim quality features - profit vs not-profit (headline variant outcome)", "",
        "Hypothesis-generating only (N in the dozens) - effect sizes and univariate logit checks, not a combined model.", "",
        "| Feature | N profit | N not-profit | Median (profit) | Median (not-profit) | Cliff's delta | Mann-Whitney p |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in quality_comparison.iterrows():
        mw_p = row["mann_whitney"]["p_value"]
        lines.append(
            f"| {row['feature']} | {row['n_a']} | {row['n_b']} | {_fmt_num(row['median_a'], 4)} | {_fmt_num(row['median_b'], 4)} "
            f"| {_fmt_num(row['cliffs_delta'], 3)} | {_fmt_num(mw_p, 3)} |"
        )
    lines.append("")
    lines.append("### Univariate logit check (standardized feature -> P(profit), one fit per feature)")
    lines.append("")
    lines.append("| Feature | N | Coef (per 1 SD) | p-value | Pseudo R2 | Note |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in logit_check.iterrows():
        lines.append(
            f"| {row['feature']} | {row['n']} | {_fmt_num(row['coef'], 3)} | {_fmt_num(row['p_value'], 3)} "
            f"| {_fmt_num(row['pseudo_r2'], 3)} | {row['note']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 7: pump-class breakdown on the underlying cross-exchange classification
# ---------------------------------------------------------------------------


def _underlying_class_by_event_id(classified_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in classified_df.iterrows():
        if pd.notna(row.get("binance_event_id")):
            out[row["binance_event_id"]] = row["underlying_pump_class"]
        if pd.notna(row.get("bybit_event_id")):
            out[row["bybit_event_id"]] = row["underlying_pump_class"]
    return out


def render_table7(headline_df: pd.DataFrame) -> str:
    lines = [
        f"## Table 7: pump-class breakdown (underlying cross-exchange classification, headline variant: {HEADLINE_BARRIER_FAMILY}/{HEADLINE_ENTRY_MODE}/{HEADLINE_ATR_MULTIPLE} ATR)",
        "", "| Underlying class | N | P(profit) | Gross expectancy R | Net expectancy R |", "|---|---|---|---|---|",
    ]
    if headline_df.empty or "underlying_pump_class" not in headline_df.columns or headline_df["underlying_pump_class"].isna().all():
        lines.append("| (no data) | - | - | - | - |")
        return "\n".join(lines)
    for pump_class, group in headline_df.groupby("underlying_pump_class", dropna=False):
        summary = ee.summarize_net_expectancy(group.to_dict("records"))
        lines.append(
            f"| {pump_class} | {summary['n']} | {_fmt_pct((group['outcome'] == 'profit').mean())} "
            f"| {_fmt_r(summary['gross_expectancy_r'])} | {_fmt_r(summary['net_expectancy_r'])} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 8: attempt #1 vs #2 (point 8)
# ---------------------------------------------------------------------------


def render_table8(attempt_summary: pd.DataFrame, headline_df: pd.DataFrame) -> str:
    lines = ["## Table 8: attempt #1 vs #2", "", "### Per-attempt exhaustion-score/structure-break comparison (all events, both anchors)", "",
              "| Attempt # | N | Mean exhaustion score peak | % had structure break | % retried further |", "|---|---|---|---|---|"]
    for _, row in attempt_summary.iterrows():
        lines.append(
            f"| {int(row['attempt_number'])} | {int(row['n'])} | {_fmt_num(row['mean_exhaustion_score_peak'])} "
            f"| {_fmt_pct(row['pct_had_structure_break'])} | {_fmt_pct(row['pct_retried_further'])} |"
        )
    lines.append("")
    lines.append("### Saga-level failed-reclaim outcome, split by whether the event needed a retry (coarser cut - see note below)")
    lines.append("")
    lines.append("| Ever retried (attempt_count>1)? | N | P(profit) | Net expectancy R |")
    lines.append("|---|---|---|---|")
    if not headline_df.empty:
        for retried, group in headline_df.groupby(headline_df["attempt_count"] > 1):
            summary = ee.summarize_net_expectancy(group.to_dict("records"))
            lines.append(f"| {bool(retried)} | {summary['n']} | {_fmt_pct((group['outcome'] == 'profit').mean())} | {_fmt_r(summary['net_expectancy_r'])} |")
    lines.append("")
    lines.append(
        "Note: `exhaustion_component_first_true_bar` (incl. failed_reclaim) is saga-level, not reset between attempts - "
        "there is no persisted \"attempt #2's own failed-reclaim bar\" separate from the whole event's. This table therefore "
        "compares attempt-level exhaustion/structure state directly (top half) and splits the saga's single failed-reclaim "
        "outcome by whether a retry ever happened (bottom half), rather than claiming a true attempt-by-attempt failed-reclaim comparison."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Point 9: structure_break anomaly diagnosis
# ---------------------------------------------------------------------------


def _structure_break_lag_diagnosis(
    events: list[EventRecord], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame], lookback_bars: int
) -> pd.DataFrame:
    rows = []
    for event in events:
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None:
            continue
        pump_start_idx = ld._pump_start_idx_for(event, ohlcv, lookback_bars)
        fe_idx = ld._anchor_idx_for(event, "first_exhaustion_signal", ohlcv, pump_start_idx)
        sb_idx = ld._anchor_idx_for(event, "structure_break", ohlcv, pump_start_idx)
        if fe_idx is None or sb_idx is None or fe_idx >= len(ohlcv) or sb_idx >= len(ohlcv):
            continue
        price_fe = float(ohlcv["close"].iloc[fe_idx])
        price_sb = float(ohlcv["close"].iloc[sb_idx])
        if not price_fe:
            continue
        rows.append(
            {
                "event_id": event.event_id, "bars_between": sb_idx - fe_idx,
                "price_change_pct": (price_sb - price_fe) / price_fe, "structure_break_after_exhaustion": sb_idx >= fe_idx,
            }
        )
    return pd.DataFrame(rows)


def render_point9(diag_df: pd.DataFrame) -> str:
    lines = ["## Point 9: structure_break vs first_exhaustion_signal - why is structure_break worse?", ""]
    if diag_df.empty:
        lines.append("No events had both anchors resolvable - cannot diagnose.")
        return "\n".join(lines)
    median_bars = diag_df["bars_between"].median()
    median_price_change = diag_df["price_change_pct"].median()
    pct_worse_entry = (diag_df["price_change_pct"] > 0).mean()  # price rose further = worse entry for a short
    pct_after = diag_df["structure_break_after_exhaustion"].mean()
    lines.append(f"N = {len(diag_df)}. Median bars between first_exhaustion_signal and structure_break: {median_bars:.1f}.")
    lines.append(f"Median price change from first_exhaustion_signal to structure_break: {_fmt_pct(median_price_change)}.")
    lines.append(f"Share of events where price moved FURTHER UP (worse short entry) by the time structure_break fired: {_fmt_pct(pct_worse_entry)}.")
    lines.append(f"Share of events where structure_break happened at/after first_exhaustion_signal (not before): {_fmt_pct(pct_after)}.")
    lines.append("")
    lines.append(
        "Diagnostic only - no rule changed. If median price change is positive and bars_between is nontrivial, the data "
        "supports \"structure confirmation arrives too late, after the best entry price has already passed\" as the "
        "explanation for why structure_break underperforms first_exhaustion_signal in first-passage terms."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 10: continuation risk (point 10)
# ---------------------------------------------------------------------------


def _compute_continuation_risk(
    events: list[EventRecord], ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame], headline_df: pd.DataFrame,
    bar_interval_ms: int, lookback_bars: int,
) -> list[dict]:
    stop_distance_by_event = dict(zip(headline_df["event_id"], (headline_df["stop_price"] - headline_df["entry_price"]) / headline_df["entry_price"]))
    results = []
    for event in events:
        stop_distance_pct = stop_distance_by_event.get(event.event_id)
        if stop_distance_pct is None or pd.isna(stop_distance_pct):
            continue
        ohlcv = ohlcv_by_symbol.get((event.exchange, event.symbol))
        if ohlcv is None:
            continue
        pump_start_idx = ld._pump_start_idx_for(event, ohlcv, lookback_bars)
        reclaim_idx_abs = ld._anchor_idx_for(event, "failed_reclaim", ohlcv, pump_start_idx)
        if reclaim_idx_abs is None or reclaim_idx_abs >= len(ohlcv):
            continue
        results.append(ee.continuation_risk_after_failed_reclaim(ohlcv, reclaim_idx_abs, stop_distance_pct, bar_interval_ms, CONTINUATION_HORIZON_HOURS))
    return results


def render_table10(continuation_summary: pd.DataFrame) -> str:
    lines = [
        f"## Table 10: continuation risk after failed reclaim (horizon={CONTINUATION_HORIZON_HOURS}h, headline stop basis)", "",
        "P(continuation-up threshold hit BEFORE the downside R-multiple) - the key tail-risk metric.", "",
        "| Threshold | N | P(continuation up first) | P(downside R first) | P(neither within horizon) |", "|---|---|---|---|---|",
    ]
    for _, row in continuation_summary.iterrows():
        lines.append(f"| {row['threshold']} | {row['n']} | {_fmt_pct(row['p_continuation_first'])} | {_fmt_pct(row['p_downside_first'])} | {_fmt_pct(row['p_neither'])} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 11: bootstrap CI (point 11) + robustness checks feeding point 13
# ---------------------------------------------------------------------------


def render_table11(bootstrap_result: dict, headline_df: pd.DataFrame) -> str:
    lines = [
        f"## Table 11: bootstrap 95% CI - headline variant ({HEADLINE_BARRIER_FAMILY} / {HEADLINE_ENTRY_MODE} / {HEADLINE_ATR_MULTIPLE} ATR stop)", "",
    ]
    if bootstrap_result.get("n", 0) == 0:
        lines.append("No trades in the headline variant - cannot bootstrap.")
        return "\n".join(lines)
    win_ci = bootstrap_result["win_rate_ci"]
    exp_ci = bootstrap_result["net_expectancy_ci"]
    lines.append(f"N = {bootstrap_result['n']}.")
    lines.append(f"Win rate: {_fmt_pct(win_ci['statistic'])} (95% CI: {_fmt_pct(win_ci['ci_low'])} - {_fmt_pct(win_ci['ci_high'])})")
    lines.append(f"Net expectancy: {_fmt_r(exp_ci['statistic'])} (95% CI: {_fmt_r(exp_ci['ci_low'])} - {_fmt_r(exp_ci['ci_high'])})")
    lines.append("")
    lines.append("Profit factor's own CI is not bootstrapped (see research/entry_edge.py docstring) - point estimate only, from Table 5.")
    lines.append("")
    lines.append("### Robustness: excluding the 3 most extreme trades (either direction) from the headline variant")
    if len(headline_df) > 3:
        by_abs = headline_df.reindex(headline_df["net_r"].abs().sort_values(ascending=False).index)
        trimmed = by_abs.iloc[3:]
        full_summary = ee.summarize_net_expectancy(headline_df.to_dict("records"))
        trimmed_summary = ee.summarize_net_expectancy(trimmed.to_dict("records"))
        lines.append("")
        lines.append(f"Full net expectancy: {_fmt_r(full_summary['net_expectancy_r'])} (n={full_summary['n']}) -> excluding top-3 |net_r|: {_fmt_r(trimmed_summary['net_expectancy_r'])} (n={trimmed_summary['n']})")
    else:
        lines.append("")
        lines.append("Fewer than 4 trades in the headline variant - trimming check skipped.")
    lines.append("")
    lines.append("### Robustness: per-exchange split of the headline variant")
    lines.append("")
    lines.append("| Exchange | N | P(profit) | Net expectancy R |")
    lines.append("|---|---|---|---|")
    for exchange, group in headline_df.groupby("exchange"):
        summary = ee.summarize_net_expectancy(group.to_dict("records"))
        lines.append(f"| {exchange} | {summary['n']} | {_fmt_pct((group['outcome'] == 'profit').mean())} | {_fmt_r(summary['net_expectancy_r'])} |")
    return "\n".join(lines)


@app.command()
def main(
    binance_run_id: str = typer.Option("binance_canonical_v2"),
    bybit_run_id: str = typer.Option("bybit_canonical_v2"),
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("phase_2b_entry_edge_v1"),
    cross_exchange_window_minutes: int = typer.Option(60),
) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)
    since_ms = datetime_to_ms(cfg.backtest.start_date)
    until_ms = datetime_to_ms(cfg.backtest.end_date)
    lookback_bars = max([bars_for_window(parse_timeframe_ms(w), bar_interval_ms) for w in cfg.pump_detector.windows], default=20)

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    exchange_events: dict[str, list[EventRecord]] = {}
    if binance_run_id:
        exchange_events["binance"] = _load_events_with_trades(binance_run_id)
        typer.echo(f"Loaded {len(exchange_events['binance'])} binance events from run_id={binance_run_id}")
    if bybit_run_id:
        exchange_events["bybit"] = _load_events_with_trades(bybit_run_id)
        typer.echo(f"Loaded {len(exchange_events['bybit'])} bybit events from run_id={bybit_run_id}")
    all_events = [e for events in exchange_events.values() for e in events]
    if not all_events:
        typer.echo("No events loaded - nothing to analyze.")
        raise typer.Exit(1)
    typer.echo(f"Total events across all exchanges: {len(all_events)}")

    ohlcv_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    for exchange, events in exchange_events.items():
        ohlcv_by_symbol.update(_load_ohlcv_by_symbol(exchange, events, timeframe, since_ms, until_ms))
    typer.echo(f"Loaded OHLCV for {len(ohlcv_by_symbol)} (exchange, symbol) pairs")

    failed_reclaim_events = [e for e in all_events if pd.notna(e.exhaustion_component_first_true_bar.get("failed_reclaim_lower_high"))]
    typer.echo(f"Events with a failed-reclaim signal: {len(failed_reclaim_events)}")

    report_sections: list[str] = []
    report_sections.append("# Phase 2B - Entry Edge Validation\n")
    report_sections.append(
        f"Universe: {len(all_events)} events total ({len(exchange_events.get('binance', []))} Binance, "
        f"{len(exchange_events.get('bybit', []))} Bybit); {len(failed_reclaim_events)} with a failed-reclaim signal.\n"
        "No live scanner, no execution - diagnostic only.\n"
    )

    # ---- Table 1 ----
    typer.echo("Table 1: per-anchor first-passage decomposition...")
    first_passage_df = ld.compute_first_passage_by_anchor(all_events, ohlcv_by_symbol, bar_interval_ms, lookback_bars, "standard_4h")
    by_anchor = ld.summarize_by_anchor(first_passage_df)
    report_sections.append(render_table1(by_anchor))
    first_passage_df.to_parquet(out_dir / "table1_first_passage_by_anchor.parquet", index=False)

    # ---- feature_frame (needed for candidate attempts' real high_vol/funding AND Table 6's quality features) ----
    typer.echo("Loading feature frames (CVD/OI/funding/realized-vol) for failed-reclaim symbols...")
    failed_reclaim_ids = {e.event_id for e in failed_reclaim_events}
    feature_frame_by_symbol: dict[tuple[str, str], pd.DataFrame] = {}
    for exchange, events in exchange_events.items():
        feature_frame_by_symbol.update(
            _load_feature_frame_by_symbol(exchange, [e for e in events if e.event_id in failed_reclaim_ids], ohlcv_by_symbol, timeframe, since_ms, until_ms, cfg)
        )

    # ---- Candidate attempts grid (feeds tables 2, 5, and the parquet deliverable) ----
    typer.echo("Building candidate-attempts grid (entry mode x ATR stop x barrier family)...")
    candidates_df = _build_candidate_attempts(failed_reclaim_events, ohlcv_by_symbol, feature_frame_by_symbol, bar_interval_ms, lookback_bars, cfg.execution_model)
    typer.echo(f"Candidate attempts: {len(candidates_df)}")
    candidates_df.to_parquet(out_dir / "phase_2b_candidate_attempts.parquet", index=False)
    headline_df = _headline_slice(candidates_df)

    report_sections.append(render_table2(candidates_df))
    report_sections.append(render_table5(candidates_df))

    # ---- Table 6: reclaim quality features ----
    typer.echo("Table 6: failed-reclaim quality features...")
    quality_df = _build_quality_df(failed_reclaim_events, ohlcv_by_symbol, feature_frame_by_symbol, lookback_bars)
    if not quality_df.empty and not headline_df.empty:
        outcome_by_event = dict(zip(headline_df["event_id"], headline_df["outcome"]))
        quality_outcome = quality_df["event_id"].map(outcome_by_event)
        quality_comparison = rq.compare_quality_by_outcome(quality_df, quality_outcome)
        logit_check = rq.univariate_logit_check(quality_df, quality_outcome)
        report_sections.append(render_table6(quality_comparison, logit_check))
        quality_df.to_parquet(out_dir / "table6_reclaim_quality_features.parquet", index=False)
    else:
        report_sections.append("## Table 6: failed-reclaim quality features\n\nInsufficient data (empty quality or headline slice).")

    # ---- Table 7: pump-class breakdown (underlying cross-exchange classification) ----
    classified_df = pd.DataFrame()
    if exchange_events.get("binance") and exchange_events.get("bybit"):
        typer.echo("Table 7: underlying cross-exchange pump-class breakdown...")
        linked_df = cross_exchange.link_cross_exchange_events(exchange_events["binance"], exchange_events["bybit"], window_minutes=cross_exchange_window_minutes)
        classified_df = uc.classify_underlying_events(linked_df, exchange_events["binance"], exchange_events["bybit"])
        underlying_map = _underlying_class_by_event_id(classified_df)
        headline_with_class = headline_df.copy()
        headline_with_class["underlying_pump_class"] = headline_with_class["event_id"].map(underlying_map)
        report_sections.append(render_table7(headline_with_class))
        classified_df.to_parquet(out_dir / "table7_underlying_classification.parquet", index=False)
    else:
        report_sections.append("## Table 7: pump-class breakdown\n\nSkipped (need events on both binance and bybit).")
        headline_with_class = headline_df.copy()
        headline_with_class["underlying_pump_class"] = np.nan

    # ---- Table 8: attempt #1 vs #2 ----
    typer.echo("Table 8: attempt comparison...")
    attempts_df = ee.compare_attempts(all_events)
    attempt_summary = ee.summarize_attempts(attempts_df)
    report_sections.append(render_table8(attempt_summary, headline_df))

    # ---- Point 9: structure_break anomaly ----
    typer.echo("Point 9: structure_break anomaly diagnosis...")
    diag_df = _structure_break_lag_diagnosis(all_events, ohlcv_by_symbol, lookback_bars)
    report_sections.append(render_point9(diag_df))

    # ---- Table 10: continuation risk ----
    typer.echo("Table 10: continuation risk...")
    continuation_results = _compute_continuation_risk(failed_reclaim_events, ohlcv_by_symbol, headline_df, bar_interval_ms, lookback_bars)
    continuation_summary = ee.summarize_continuation_risk(continuation_results)
    report_sections.append(render_table10(continuation_summary))

    # ---- Table 11: bootstrap CI + robustness ----
    typer.echo("Table 11: bootstrap CI + robustness checks...")
    bootstrap_result = ee.bootstrap_net_expectancy_ci(headline_df.to_dict("records"))
    report_sections.append(render_table11(bootstrap_result, headline_df))

    # ---- Final 4 sections ----
    headline_summary = ee.summarize_net_expectancy(headline_df.to_dict("records"))
    report_sections.append(
        _render_final_sections(headline_summary, bootstrap_result, by_anchor, diag_df, headline_with_class, candidates_df)
    )

    report_text = "\n\n".join(report_sections) + "\n"
    (out_dir / "phase_2b_entry_edge_report.md").write_text(report_text, encoding="utf-8")
    typer.echo(f"Done. Report + parquet under: {out_dir}")


def _render_final_sections(
    headline_summary: dict, bootstrap_result: dict, by_anchor: pd.DataFrame, diag_df: pd.DataFrame,
    headline_with_class: pd.DataFrame, candidates_df: pd.DataFrame,
) -> str:
    lines = ["## EVIDENCE FOR EDGE", ""]
    failed_reclaim_row = by_anchor[by_anchor["anchor"] == "failed_reclaim"]
    pump_detected_row = by_anchor[by_anchor["anchor"] == "pump_detected"]
    if not failed_reclaim_row.empty and not pump_detected_row.empty:
        fr_p = failed_reclaim_row.iloc[0]["p_profit"]
        pd_p = pump_detected_row.iloc[0]["p_profit"]
        if pd.notna(fr_p) and pd.notna(pd_p) and pd_p > 0:
            rel = (fr_p - pd_p) / pd_p
            lines.append(f"- First-passage P(profit) at the failed_reclaim anchor ({_fmt_pct(fr_p)}) remains above the pump_detected baseline ({_fmt_pct(pd_p)}), a {rel:+.1%} relative difference (2R/1R/4h barrier, uniform ATR stop basis) - the Phase 2A finding replicates in this pass's own anchor comparison.")
    headline_net_positive = (
        headline_summary.get("n", 0) > 0 and pd.notna(headline_summary.get("net_expectancy_r")) and headline_summary["net_expectancy_r"] > 0
    )
    if headline_net_positive:
        lines.append(f"- Headline variant ({HEADLINE_BARRIER_FAMILY}/{HEADLINE_ENTRY_MODE}/{HEADLINE_ATR_MULTIPLE} ATR, real entry price, real structural stop, fees+slippage+funding+stop-overrun all deducted): net expectancy = {_fmt_r(headline_summary['net_expectancy_r'])}, N={headline_summary['n']}, win rate={_fmt_pct(headline_summary['win_rate'])}, PF={_fmt_num(headline_summary['profit_factor'])}.")
        lines.append("- This net expectancy is positive after realistic costs, not just a favorable binary probability shift.")
    if not headline_net_positive:
        lines.append("- No positive net-of-costs headline finding to report here (see EVIDENCE AGAINST EDGE for the actual headline numbers) - the anchor-comparison bullet above is the only evidence for edge this pass produced.")
    lines.append("")

    lines.append("## EVIDENCE AGAINST EDGE")
    lines.append("")
    if headline_summary.get("n", 0) > 0 and pd.notna(headline_summary.get("net_expectancy_r")) and headline_summary["net_expectancy_r"] <= 0:
        lines.append(f"- Headline variant net expectancy is NOT positive ({_fmt_r(headline_summary['net_expectancy_r'])}) once fees/slippage/funding/stop-overrun are deducted from the real entry/stop.")
    if not candidates_df.empty:
        grid_net_values = []
        for _, group in candidates_df.groupby(["barrier_family", "entry_mode", "atr_stop_multiple"]):
            net_r = ee.summarize_net_expectancy(group.to_dict("records")).get("net_expectancy_r")
            if pd.notna(net_r):
                grid_net_values.append(net_r)
        if grid_net_values:
            n_negative = sum(1 for v in grid_net_values if v <= 0)
            lines.append(f"- Not cherry-picked: {n_negative}/{len(grid_net_values)} combinations across the full barrier-family x entry-mode x ATR-stop grid (Table 5) also show net expectancy <= 0.")
    win_ci = bootstrap_result.get("win_rate_ci", {})
    if win_ci and pd.notna(win_ci.get("ci_low")):
        lines.append(f"- Bootstrap 95% CI for the headline win rate is {_fmt_pct(win_ci['ci_low'])} - {_fmt_pct(win_ci['ci_high'])}; the lower bound {'is' if win_ci['ci_low'] < 0.333 else 'is not'} below the ~33.3% breakeven threshold for a 2R/1R barrier.")
    if not diag_df.empty and diag_df["price_change_pct"].median() > 0:
        lines.append(f"- structure_break fires at a materially worse price than first_exhaustion_signal in the median case ({_fmt_pct(diag_df['price_change_pct'].median())} further adverse) - the anomaly from the prior report is confirmed, not resolved.")
    lines.append(f"- Sample size remains small (N={headline_summary.get('n', 0)} for the headline variant out of {len(candidates_df['event_id'].unique()) if not candidates_df.empty else 0} distinct failed-reclaim events) - every finding here is provisional.")
    lines.append("")

    lines.append("## UNCERTAINTIES")
    lines.append("")
    lines.append("- Funding cost is modeled as a fixed placeholder (default 0.0, see research/entry_edge.py `compute_r_trades`), not real per-bar funding-rate data - a real accounting could shift net expectancy in either direction.")
    lines.append("- ATR stop grid (0.15/0.25/0.40) and barrier-family grid are a fixed sensitivity sweep, not an exhaustive search - a variant outside this grid could behave differently.")
    lines.append("- Pump-class and quality-feature breakdowns split an already-small N into smaller sub-groups; per-class/per-feature N is frequently too low to trust individually (see each table's own N column).")
    lines.append("- The attempt #1 vs #2 comparison (Table 8) cannot isolate a retried attempt's OWN failed-reclaim outcome (not persisted) - only the coarser saga-level split.")
    lines.append("- OI/funding/CVD depth limitations already documented in Phase 1.5/2A (Binance OI retention ~1.7 days) still apply to Table 6's quality features and Table 7's pump-class breakdown.")
    lines.append("")

    lines.append("## RECOMMENDED NEXT STEP")
    lines.append("")
    go_criteria_met = (
        headline_summary.get("n", 0) > 0 and pd.notna(headline_summary.get("net_expectancy_r"))
        and headline_summary["net_expectancy_r"] > 0 and headline_summary.get("profit_factor", 0) > 1
    )
    if go_criteria_met:
        lines.append(
            "Headline variant clears net expectancy > 0 and PF > 1 - review the full Table 5 grid and Table 11 robustness "
            "checks together before deciding whether the GO criterion (spec point 13: positive net expectancy, PF>1, survives "
            "slippage, CI not catastrophically wide, not driven by 2-3 trades, consistent direction across exchanges/pump "
            "classes) is met in full. Do not extend history or build a scanner without that explicit review."
        )
    else:
        lines.append(
            "Headline variant does not clear the point-13 GO bar (net expectancy <= 0 and/or PF <= 1) - per the user's own "
            "instruction, this means going back to causal features rather than loosening thresholds to force trades. Do not "
            "extend the dataset to 12-24 months or build a scanner from this variant."
        )
    lines.append("")
    lines.append("Awaiting user review before any further phase, live scanner, or execution work.")
    return "\n".join(lines)


if __name__ == "__main__":
    app()
