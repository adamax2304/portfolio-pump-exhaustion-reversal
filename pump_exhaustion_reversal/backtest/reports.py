"""Report generation (spec sections 22.7, 23): turns metrics.py's numbers
into a readable Markdown report, including the explicit GO/NO-GO verdict
against spec section 23's acceptance criteria - the actual gate between
Stage 1 and Stage 2 (project plan: if these don't hold, the strategy stays
research-only, no live scanner gets built on top of it).
"""

from __future__ import annotations

from pathlib import Path

from ..config import AcceptanceConfig
from ..data import storage

STANDARD_LIMITATIONS = [
    "OHLCV: 5m/15m/1h/4h/1d cover the full requested date range; 1m is only backfilled for a recent window "
    "(scripts/ingest.py --m1-lookback-days) - the base timeframe for this backtest is 5m, not 1m.",
    "CVD is high-confidence only where Binance's native taker_buy_base_volume field is available, or within "
    "event-anchored trades windows; Bybit CVD outside those windows is a low-confidence close-vs-open proxy.",
    "Open interest is best-effort and asymmetric by exchange (verified against the live API): Bybit returned a "
    "gapless 60-day 5m history; Binance's OI-history endpoint only ever returns roughly the last ~1.25 days "
    "regardless of the requested start time. Older bars (especially on Binance) have missing/low-confidence OI "
    "features - the scoring layer renormalizes around them rather than scoring them as zero (spec section 8).",
    "Liquidations use proxy_mode only (OI-drop + price-spike + volume-spike composite) - no free historical "
    "liquidation feed exists on Binance or Bybit.",
    "News/catalyst coverage is whatever was manually imported via CSV (spec section 4.7) - historically sparse, "
    "not a live announcement feed; most historical pumps will show news_flag=False even if a real catalyst existed.",
    "PumpClass.SPOT_ORGANIC is a residual/fallback classification (no spot orderflow data collected in this "
    "linear-perp-only universe), not a positive detection of organic spot buying.",
    "PumpClass.LIQUIDITY_MANIPULATION has no cross-exchange confirmation - single-venue data only.",
    "Entry fills assume next-bar-open; the 'limit retest' half of spec's entry_type is not modeled. TP2/TP3 "
    "targets fall back to entry_price (no incremental target beyond TP1) when AVWAP/breakout context is missing.",
    "Universe liquidity_score is a fixed value per run, not a live per-bar spread/depth feed (no orderbook history).",
]


def _fmt_pct(value, digits: int = 1) -> str:
    if value is None or value != value:  # NaN check without importing numpy just for this
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value, digits: int = 3) -> str:
    if value is None or value != value:
        return "n/a"
    if value in (float("inf"), float("-inf")):
        return str(value)
    return f"{value:.{digits}f}"


def evaluate_acceptance(
    oos_trade_metrics: dict, oos_event_metrics: dict | None, slippage_stress_metrics: dict | None, cfg: AcceptanceConfig
) -> dict:
    """spec section 23. Every check must pass for a GO verdict; any failure
    keeps the strategy research-only (project plan's Stage 1/2 gate)."""
    n_oos_trades = oos_trade_metrics.get("n_trades", 0)
    profit_factor = oos_trade_metrics.get("profit_factor")
    expectancy = oos_trade_metrics.get("expectancy_r")

    checks: dict[str, tuple[bool, str]] = {
        "min_oos_trades": (
            n_oos_trades >= cfg.min_oos_trades,
            f"{n_oos_trades} trades (need >= {cfg.min_oos_trades}, prefer >= {cfg.preferred_oos_trades})",
        ),
        "oos_profit_factor_min": (
            (profit_factor or 0) >= cfg.oos_profit_factor_min,
            f"PF={_fmt_num(profit_factor)} (need >= {cfg.oos_profit_factor_min}, prefer >= {cfg.preferred_profit_factor})",
        ),
        "oos_expectancy_r_min": (
            (expectancy if expectancy is not None else -999) >= cfg.oos_expectancy_r_min,
            f"expectancy={_fmt_num(expectancy)}R (need >= {cfg.oos_expectancy_r_min}R)",
        ),
        "no_single_trade_dominates_profit": (
            bool(oos_trade_metrics.get("no_single_trade_dominates_profit", False)),
            "single best trade must not exceed 50% of gross profit",
        ),
    }
    if slippage_stress_metrics is not None and cfg.slippage_stress_still_positive:
        stressed_expectancy = slippage_stress_metrics.get("expectancy_r")
        checks["slippage_stress_still_positive"] = (
            (stressed_expectancy if stressed_expectancy is not None else -999) > 0,
            f"stressed expectancy={_fmt_num(stressed_expectancy)}R under widened slippage assumptions",
        )
    passed = all(v[0] for v in checks.values())
    return {"go": passed, "checks": {k: {"passed": v[0], "detail": v[1]} for k, v in checks.items()}}


def render_markdown_report(
    run_id: str,
    event_metrics: dict,
    overall_trade_metrics: dict,
    oos_trade_metrics: dict,
    by_pump_class: dict,
    by_entry_mode: dict,
    acceptance: dict,
    limitations: list[str],
) -> str:
    lines = [f"# Pump Exhaustion Reversal - Backtest Report ({run_id})", "", "Stage 1: research/backtest only - no live trading.", ""]

    lines += ["## Event Study (spec 22.3)", ""]
    lines.append(f"- Pump events detected: {event_metrics.get('n_pump_events', 0)}")
    lines.append(
        f"- Events that reached a triggered trade: {event_metrics.get('n_events_triggered_trade', 0)} "
        f"({_fmt_pct(event_metrics.get('trigger_rate'))})"
    )
    lines.append(f"- Events by pump class: {event_metrics.get('events_by_class', {})}")
    lines.append("")
    lines.append("**Continuation risk after detection (the key risk metric - spec 22.7):**")
    lines.append(f"- P(price continues >= +20% against the short thesis): {_fmt_pct(event_metrics.get('prob_continuation_20pct'))}")
    lines.append(f"- P(price continues >= +50%): {_fmt_pct(event_metrics.get('prob_continuation_50pct'))}")
    lines.append(f"- P(price continues >= +100%): {_fmt_pct(event_metrics.get('prob_continuation_100pct'))}")

    lines += ["", "## Trade Performance - overall (spec 22.7)", ""]
    lines += _trade_metrics_lines(overall_trade_metrics)

    lines += ["", "## Trade Performance - out-of-sample only", ""]
    lines += _trade_metrics_lines(oos_trade_metrics)

    lines += ["", "## By Pump Class", ""]
    for cls, m in sorted(by_pump_class.items()):
        lines.append(
            f"- **{cls}**: n={m['n_trades']}, win_rate={_fmt_pct(m.get('win_rate'))}, "
            f"expectancy={_fmt_num(m.get('expectancy_r'))}R, PF={_fmt_num(m.get('profit_factor'))}"
        )

    lines += ["", "## By Entry Mode", ""]
    for mode, m in sorted(by_entry_mode.items()):
        lines.append(
            f"- **{mode}**: n={m['n_trades']}, win_rate={_fmt_pct(m.get('win_rate'))}, "
            f"expectancy={_fmt_num(m.get('expectancy_r'))}R, PF={_fmt_num(m.get('profit_factor'))}"
        )

    lines += ["", "## GO / NO-GO (spec section 23)", ""]
    lines.append(f"**Verdict: {'GO' if acceptance['go'] else 'NO-GO - strategy remains research-only'}**")
    lines.append("")
    for name, check in acceptance["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- [{mark}] {name}: {check['detail']}")

    lines += ["", "## Limitations (what this backtest does and does not cover)", ""]
    for lim in limitations:
        lines.append(f"- {lim}")

    return "\n".join(lines)


def _trade_metrics_lines(m: dict) -> list[str]:
    return [
        f"- Trades: {m.get('n_trades', 0)}",
        f"- Win rate: {_fmt_pct(m.get('win_rate'))}",
        f"- Avg R / Expectancy: {_fmt_num(m.get('avg_r'))}R",
        f"- Median R: {_fmt_num(m.get('median_r'))}R",
        f"- Profit factor: {_fmt_num(m.get('profit_factor'))}",
        f"- Max drawdown: {_fmt_num(m.get('max_drawdown_r'))}R",
        f"- Max consecutive losses: {m.get('max_consecutive_losses', 0)}",
        f"- Avg hold time: {_fmt_num(m.get('avg_hold_minutes'), 1)} min",
        f"- Avg MAE / MFE: {_fmt_num(m.get('avg_mae_r'))}R / {_fmt_num(m.get('avg_mfe_r'))}R",
        f"- Avg slippage: {_fmt_num(m.get('avg_slippage_bps_of_notional'), 1)} bps of notional",
        f"- Stop overrun: {_fmt_num(m.get('stop_overrun_bps'), 1)} bps",
        f"- Tail loss (p5): {_fmt_num(m.get('tail_loss_r_p05'))}R",
        f"- Best / worst trade: {_fmt_num(m.get('best_trade_r'))}R / {_fmt_num(m.get('worst_trade_r'))}R",
        f"- Total fees / funding: ${_fmt_num(m.get('total_fees'), 2)} / ${_fmt_num(m.get('total_funding'), 2)}",
        f"- Net P&L: ${_fmt_num(m.get('net_pnl_usd'), 2)}",
        f"- No single trade dominates profit: {m.get('no_single_trade_dominates_profit')}",
    ]


def save_report(markdown: str, run_id: str) -> Path:
    storage.ensure_data_dirs()
    path = storage.REPORTS_ROOT / f"report_{run_id}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
