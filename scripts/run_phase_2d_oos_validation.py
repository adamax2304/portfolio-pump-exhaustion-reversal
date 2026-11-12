"""CLI: Phase 2D - Pre-Registered OOS Validation of the FROZEN
OI_ROLLOVER + VOLUME_DECAY candidate (config/oi_volume_reversal_candidate_v1.yaml).

Runs the SINGLE frozen setup (entry_mode=confirmation_close, ATR
stop=0.25, barrier=2R:1R:4h) against events_bybit_historical_oos_v1
(2021-01-01..2026-02-24 - a period the discovery sample, 2026-02-25..
2026-08-25, never touched). No parameter sweep, no threshold changes -
if this run shows the candidate should be tuned differently, that is a
finding for a FUTURE frozen version, not something this script may act on.

python scripts/run_phase_2d_oos_validation.py --oos-run-id bybit_historical_oos_v1
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
import yaml

from pump_exhaustion_reversal.backtest import walk_forward
from pump_exhaustion_reversal.backtest.simulator import EventRecord, TradeRecord
from pump_exhaustion_reversal.config import BACKTEST_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.data import storage
from pump_exhaustion_reversal.data.feature_store import build_feature_frame
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, parse_timeframe_ms
from pump_exhaustion_reversal.research import early_entry
from pump_exhaustion_reversal.research import entry_edge as ee
from pump_exhaustion_reversal.research import first_passage as fp
from pump_exhaustion_reversal.research import lifecycle_diagnostics as ld
from pump_exhaustion_reversal.research import oos_validation as oos

app = typer.Typer(add_completion=False)

DIAGNOSTICS_ROOT = storage.DATA_ROOT / "diagnostics"
FROZEN_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "oi_volume_reversal_candidate_v1.yaml"
PRIMARY_FAMILY = "oi_plus_volume_decay"


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
    since_ms: int, until_ms: int, cfg,
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


def _is_high_vol(feature_frame, entry_bar_idx: int) -> bool:
    if feature_frame is None or "realized_vol_ratio" not in feature_frame.columns or entry_bar_idx >= len(feature_frame):
        return False
    ratio = feature_frame["realized_vol_ratio"].iloc[entry_bar_idx]
    return bool(ratio > 2.0) if pd.notna(ratio) else False


def _compute_funding_r(feature_frame, entry_bar_idx: int, exit_bar_idx: int, bar_interval_ms: int, stop_distance_pct: float) -> float:
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


def _pump_magnitude(ohlcv: pd.DataFrame, event: EventRecord, pump_start_idx: int) -> float:
    if pd.isna(event.climax_local_idx):
        return np.nan
    climax_abs_idx = pump_start_idx + int(event.climax_local_idx)
    if climax_abs_idx >= len(ohlcv):
        return np.nan
    start_price = float(ohlcv["open"].iloc[pump_start_idx])
    climax_price = float(ohlcv["high"].iloc[climax_abs_idx])
    return (climax_price - start_price) / start_price if start_price else np.nan


def _fmt_pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.1%}"


def _fmt_r(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}R"


def _fmt_num(v, digits: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}f}"


@app.command()
def main(
    oos_run_id: str = typer.Option("bybit_historical_oos_v1"),
    config: str = typer.Option(str(BACKTEST_CONFIG_PATH)),
    out_run_id: str = typer.Option("phase_2d_oos_validation_v1"),
) -> None:
    with open(FROZEN_CONFIG_PATH, encoding="utf-8") as f:
        frozen = yaml.safe_load(f)
    entry_mode = frozen["entry_price_logic"]["entry_mode"]
    atr_multiple = frozen["stop_definition"]["atr_buffer_multiplier"]
    profit_r = frozen["exit_logic"]["profit_r_multiple"]
    adverse_r = frozen["exit_logic"]["adverse_r_multiple"]
    horizon_hours = frozen["exit_logic"]["horizon_hours"]
    criteria = frozen["oos_acceptance_criteria"]
    typer.echo(f"Frozen setup: {PRIMARY_FAMILY} / {entry_mode} / ATR {atr_multiple} / {profit_r}R:{adverse_r}R:{horizon_hours}h")

    cfg = load_config(DEFAULT_CONFIG_PATH, config)
    timeframe = cfg.backtest.timeframe_base
    bar_interval_ms = parse_timeframe_ms(timeframe)

    all_events = _load_events_with_trades(oos_run_id)
    if not all_events:
        typer.echo(f"ERROR: no events found for run_id={oos_run_id} - run scripts/run_backtest.py first.")
        raise typer.Exit(1)
    typer.echo(f"Loaded {len(all_events)} historical OOS events")

    since_ms = datetime_to_ms("2021-01-01")
    until_ms = datetime_to_ms("2026-02-24")
    ohlcv_by_symbol = _load_ohlcv_by_symbol("bybit", all_events, timeframe, since_ms, until_ms)
    feature_frame_by_symbol = _load_feature_frame_by_symbol("bybit", all_events, ohlcv_by_symbol, timeframe, since_ms, until_ms, cfg)
    typer.echo(f"Loaded OHLCV/features for {len(ohlcv_by_symbol)} symbols")

    out_dir = DIAGNOSTICS_ROOT / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- detect the frozen signal on every OOS event, restricted to PRIMARY_FAMILY only ----
    typer.echo(f"Scanning for {PRIMARY_FAMILY} signals (no other family evaluated)...")
    rows = []
    magnitude_by_event: dict[str, float] = {}
    horizon_bars = max(round(horizon_hours * 3_600_000 / bar_interval_ms), 1)
    for event in all_events:
        ohlcv = ohlcv_by_symbol.get(("bybit", event.symbol))
        feature_frame = feature_frame_by_symbol.get(("bybit", event.symbol))
        if ohlcv is None or feature_frame is None:
            continue
        pump_start_idx = ld._pump_start_idx_for(event, ohlcv, 288)
        magnitude_by_event[event.event_id] = _pump_magnitude(ohlcv, event, pump_start_idx)

        signals, pump_start_idx_from_scan = early_entry.find_early_entry_signals(event, ohlcv, feature_frame, timeframe, cfg)
        if pump_start_idx_from_scan is None:
            continue
        for sig in signals:
            if sig.entry_family != PRIMARY_FAMILY:
                continue
            signal_abs_idx = pump_start_idx_from_scan + sig.signal_wide_idx
            climax_abs_idx = pump_start_idx_from_scan + sig.climax_wide_idx
            if signal_abs_idx >= len(ohlcv) - 1 or climax_abs_idx < 0 or climax_abs_idx >= len(ohlcv):
                continue
            oi_tier = oos.oi_quality_tier(feature_frame, climax_abs_idx, signal_abs_idx)

            entry_price = ee.resolve_entry_price(ohlcv, signal_abs_idx, entry_mode)
            if entry_price is None:
                continue
            walk_start_idx = signal_abs_idx + 1
            stop_price = ee.compute_atr_stop_price(ohlcv, climax_abs_idx, atr_multiple)
            if stop_price is None or stop_price <= entry_price:
                continue
            stop_distance_pct = (stop_price - entry_price) / entry_price
            profit_pct, adverse_pct = fp.resolve_r_based_barriers(stop_distance_pct, profit_r, adverse_r)
            outcome = ee.first_passage_from_entry(ohlcv, walk_start_idx, entry_price, profit_pct, adverse_pct, horizon_bars)
            exit_bar_idx = walk_start_idx + outcome["bars"] - 1
            high_vol = _is_high_vol(feature_frame, walk_start_idx)
            funding_r = _compute_funding_r(feature_frame, walk_start_idx, exit_bar_idx, bar_interval_ms, stop_distance_pct)
            outcome_for_r = dict(outcome, stop_distance_pct=stop_distance_pct, high_vol=high_vol, funding_r=funding_r)
            r_trades = ee.compute_r_trades([outcome_for_r], profit_r, adverse_r, cfg.execution_model, bar_interval_ms)
            if not r_trades:
                continue
            rows.append(
                {
                    "event_id": event.event_id, "symbol": event.symbol, "attempt_number": sig.attempt_number,
                    "oi_data_confidence": oi_tier, "entry_price": entry_price, "stop_price": stop_price,
                    "stop_distance_pct": stop_distance_pct, **outcome, **r_trades[0],
                }
            )

    candidates_df = pd.DataFrame(rows)
    candidates_df.to_parquet(out_dir / "oos_candidate_attempts.parquet", index=False)
    typer.echo(f"Total {PRIMARY_FAMILY} candidate attempts on OOS: {len(candidates_df)} (all quality tiers)")

    primary_df = candidates_df[candidates_df["oi_data_confidence"].isin(["HIGH", "MEDIUM"])] if not candidates_df.empty else candidates_df
    typer.echo(f"Primary result restricted to HIGH/MEDIUM OI coverage: N={len(primary_df)}")

    report_sections: list[str] = ["# Phase 2D - OI_ROLLOVER + VOLUME_DECAY OOS Validation\n"]
    report_sections.append(
        f"Frozen setup (config/oi_volume_reversal_candidate_v1.yaml): {PRIMARY_FAMILY} / {entry_mode} / ATR {atr_multiple} / "
        f"{profit_r}R:{adverse_r}R:{horizon_hours}h. OOS sample: {len(all_events)} events, Bybit, 2021-01-01..2026-02-24 "
        "(strictly before the 2026-02-25..2026-08-25 discovery sample - untouched historical external validation, not a "
        "true forward sample). No parameter sweep - this is the single primary setup, no other threshold was evaluated.\n"
    )

    # ---- DISCOVERY RESULT ----
    report_sections.append(
        "## DISCOVERY RESULT\n\n"
        f"- Combined (Binance+Bybit, discovery sample): N=31, net~+0.225R, PF~1.84, bootstrap 95% CI crosses zero.\n"
        f"- Binance subset: N=5, net~+0.851R, PF~15.38 - not treated as evidence (too few trades for that PF to be trustworthy).\n"
        f"- **Bybit subset (this candidate's actual basis): N=26, net~+0.105R, PF~1.34** - the number this OOS run tests.\n"
    )

    # ---- OOS RESULT ----
    oos_summary = ee.summarize_net_expectancy(primary_df.to_dict("records"))
    lines = ["## OOS RESULT\n"]
    if oos_summary.get("n", 0) == 0:
        lines.append(f"**N=0 HIGH/MEDIUM-OI-quality {PRIMARY_FAMILY} signals found on the OOS sample.** No result to evaluate.")
    else:
        lines.append(
            f"N={oos_summary['n']}, Gross R={_fmt_r(oos_summary['gross_expectancy_r'])}, **Net R={_fmt_r(oos_summary['net_expectancy_r'])}**, "
            f"PF={_fmt_num(oos_summary['profit_factor'])}, win rate={_fmt_pct(oos_summary['win_rate'])}, "
            f"avg win={_fmt_r(oos_summary['avg_win_r'])}, avg loss={_fmt_r(oos_summary['avg_loss_r'])}, "
            f"avg MAE={_fmt_r(oos_summary['avg_mae_r'])}, avg MFE={_fmt_r(oos_summary['avg_mfe_r'])}."
        )
        bootstrap = ee.bootstrap_net_expectancy_ci(primary_df.to_dict("records"))
        win_ci, exp_ci = bootstrap.get("win_rate_ci", {}), bootstrap.get("net_expectancy_ci", {})
        lines.append(
            f"\nBootstrap 95% CI: win rate {_fmt_pct(win_ci.get('ci_low'))}-{_fmt_pct(win_ci.get('ci_high'))}, "
            f"net expectancy {_fmt_r(exp_ci.get('ci_low'))}-{_fmt_r(exp_ci.get('ci_high'))}."
        )
        oi_tier_counts = candidates_df["oi_data_confidence"].value_counts().to_dict()
        lines.append(f"\nOI data confidence breakdown (all signals, before HIGH/MEDIUM filter): {oi_tier_counts}")
    report_sections.append("\n".join(lines))

    # ---- COST STRESS ----
    lines = ["## COST STRESS\n"]
    if oos_summary.get("n", 0) > 0:
        for mult in (1.5, 2.0):
            stressed_cfg = cfg.execution_model.model_copy(
                update={
                    "taker_fee_bps": cfg.execution_model.taker_fee_bps * mult,
                    "default_slippage_bps": cfg.execution_model.default_slippage_bps * mult,
                    "stop_slippage_bps": cfg.execution_model.stop_slippage_bps * mult,
                }
            )
            outcomes_with_stop = primary_df.to_dict("records")
            stressed_trades = ee.compute_r_trades(outcomes_with_stop, profit_r, adverse_r, stressed_cfg, bar_interval_ms)
            stressed_summary = ee.summarize_net_expectancy(stressed_trades)
            lines.append(f"- {mult}x BASE costs: Net R={_fmt_r(stressed_summary.get('net_expectancy_r'))}, PF={_fmt_num(stressed_summary.get('profit_factor'))}")
        zero_cfg = cfg.execution_model.model_copy(update={"taker_fee_bps": 0.0, "default_slippage_bps": 0.0, "stop_slippage_bps": 0.0})
        zero_trades = ee.compute_r_trades(primary_df.to_dict("records"), profit_r, adverse_r, zero_cfg, bar_interval_ms)
        zero_summary = ee.summarize_net_expectancy(zero_trades)
        lines.append(f"- ZERO cost (diagnostic only, never used for GO/STOP): Net R={_fmt_r(zero_summary.get('net_expectancy_r'))}")
    else:
        lines.append("N/A - no OOS signals to stress-test.")
    report_sections.append("\n".join(lines))

    # ---- CLUSTER ROBUSTNESS ----
    lines = ["## CLUSTER ROBUSTNESS\n"]
    if oos_summary.get("n", 0) > 0:
        symbol_summary = oos.symbol_clustering_summary(primary_df)
        cluster_bootstrap = oos.cluster_bootstrap_ci(primary_df)
        lines.append(
            f"N events={symbol_summary['n_events']}, N unique symbols={symbol_summary['n_unique_symbols']}, "
            f"max events/symbol={symbol_summary['max_events_per_symbol']}, top-5-symbol PnL share={_fmt_pct(symbol_summary['top5_symbol_pnl_share'])}."
        )
        lines.append(
            f"\nCluster (per-symbol) bootstrap: median expectancy={_fmt_r(cluster_bootstrap.get('median_expectancy'))}, "
            f"95% CI={_fmt_r(cluster_bootstrap.get('ci_low'))} to {_fmt_r(cluster_bootstrap.get('ci_high'))} (N symbols={cluster_bootstrap.get('n_symbols', 0)})."
        )
        events_with_regime = walk_forward.label_market_regime(pd.DataFrame([e.__dict__ for e in all_events]))
        regime_df = oos.regime_breakdown(primary_df, events_with_regime)
        if not regime_df.empty:
            lines.append("\nRegime breakdown (diagnostic, no regime picked as best):")
            for _, row in regime_df.iterrows():
                lines.append(f"  - {row['dimension']}={row['value']}: N={row['n']}, net={_fmt_r(row['net_expectancy_r'])}")
        magnitude_df = oos.pump_magnitude_buckets(primary_df, magnitude_by_event)
        if not magnitude_df.empty:
            lines.append("\nPump magnitude buckets (diagnostic):")
            for _, row in magnitude_df.iterrows():
                lines.append(f"  - {row['bucket']}: N={row['n']}, net={_fmt_r(row['net_expectancy_r'])}")

        # matched-control study
        all_events_features_df = pd.DataFrame(
            [{"event_id": e.event_id, "pump_score": e.pump_score, "volume_zscore": e.volume_zscore, "btc_return_concurrent": e.btc_return_concurrent} for e in all_events]
        )
        signal_ids = set(primary_df["event_id"])
        matched = oos.match_controls(signal_ids, all_events_features_df, k=3)
        matched.to_parquet(out_dir / "matched_controls.parquet", index=False)
        lines.append(f"\nMatched-control study: {len(signal_ids)} signal events matched against {len(matched)} nearest-neighbor controls (by pump_score/volume_zscore/btc_return_concurrent) - saved to matched_controls.parquet for manual review.")
    else:
        lines.append("N/A - no OOS signals.")
    report_sections.append("\n".join(lines))

    # ---- TAIL RISK ----
    lines = ["## TAIL RISK\n"]
    if oos_summary.get("n", 0) > 0:
        lines.append(f"Max adverse excursion (avg, R): {_fmt_r(oos_summary.get('avg_mae_r'))}. Max favorable excursion (avg, R): {_fmt_r(oos_summary.get('avg_mfe_r'))}.")
        worst_mae = primary_df["mae_r"].max() if "mae_r" in primary_df.columns and primary_df["mae_r"].notna().any() else np.nan
        lines.append(f"Worst single-event MAE: {_fmt_r(worst_mae)}.")
    else:
        lines.append("N/A - no OOS signals.")
    report_sections.append("\n".join(lines))

    # ---- FINAL VERDICT ----
    symbol_summary = oos.symbol_clustering_summary(primary_df) if oos_summary.get("n", 0) > 0 else {"top5_symbol_pnl_share": np.nan}
    cluster_bootstrap = oos.cluster_bootstrap_ci(primary_df) if oos_summary.get("n", 0) > 0 else {}
    verdict = oos.check_acceptance_criteria(oos_summary, cluster_bootstrap, symbol_summary, criteria)
    lines = ["## FINAL VERDICT\n", f"**{verdict['outcome']}**\n"]
    for k, v in verdict.items():
        if k != "outcome":
            lines.append(f"- {k}: {v}")
    if verdict["outcome"] == "INSUFFICIENT_SAMPLE":
        lines.append(
            f"\nN={verdict.get('n', 0)} < preferred minimum {verdict.get('n_required', 50)} - per the pre-registered criteria, "
            "no GO/STOP verdict is permitted at this sample size. The candidate is NOT rejected and NOT confirmed - "
            "status is explicitly undetermined pending more signals (forward collection continues per the frozen config)."
        )
    elif verdict["outcome"] == "C_REJECT_CLOSE_BRANCH":
        lines.append("\nOI_ROLLOVER + VOLUME_DECAY branch should be closed per the pre-registered criteria - do not return to the discovery-sample 6 months to search for a different threshold.")
    elif verdict["outcome"] == "A_CONFIRMED_ENOUGH_FOR_PAPER_TRADING":
        lines.append("\nMeaningful positive net EV and robustness survive OOS - next step per the pre-registered plan: live scanner + PAPER TRADING ONLY, no real money.")
    else:
        lines.append("\nWeak/uncertain - do not change the strategy. Continue accumulating forward/OOS events and re-check at the next pre-registered checkpoint.")
    report_sections.append("\n".join(lines))

    report_text = "\n\n".join(report_sections) + "\n"
    (out_dir / "phase_2d_oi_volume_oos_validation.md").write_text(report_text, encoding="utf-8")
    typer.echo(f"Done. Report + parquets under: {out_dir}")


if __name__ == "__main__":
    app()
