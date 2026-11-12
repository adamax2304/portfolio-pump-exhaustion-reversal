"""Backtest metrics (spec section 22.7).

Operates on the TradeRecord/EventRecord lists simulator.py produces - one
call per slice you want a report for (all trades, just class A, just OOS
trades...). Callers filter the input lists before calling; this module has
no notion of dates, classes, or regimes on its own, which keeps it reusable
for event_study.py, walk_forward.py, and reports.py alike.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .simulator import EventRecord, TradeRecord

CONTINUATION_THRESHOLDS = (0.20, 0.50, 1.00)


def _r_values(trades: list[TradeRecord]) -> np.ndarray:
    return np.array([t.final_r for t in trades], dtype="float64")


def _max_drawdown_r(trades: list[TradeRecord]) -> float:
    if not trades:
        return 0.0
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    equity = np.cumsum([t.final_r for t in ordered])
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def _max_consecutive_losses(trades: list[TradeRecord]) -> int:
    if not trades:
        return 0
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    longest = current = 0
    for t in ordered:
        if t.final_r <= 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _profit_factor(r_values: np.ndarray) -> float:
    gains = r_values[r_values > 0].sum()
    losses = -r_values[r_values < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _no_single_trade_dominates(r_values: np.ndarray, threshold_fraction: float = 0.5) -> bool:
    """spec section 23's "no_single_trade_dominates_profit": the single best
    winning trade must not account for more than threshold_fraction of total
    gross profit - True (passes) when there are no winners at all (nothing to dominate)."""
    gains = r_values[r_values > 0]
    if gains.sum() <= 0:
        return True
    return bool(gains.max() / gains.sum() <= threshold_fraction)


def compute_trade_metrics(trades: list[TradeRecord]) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": np.nan, "avg_r": np.nan, "median_r": np.nan,
            "expectancy_r": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0,
            "max_consecutive_losses": 0, "avg_hold_minutes": np.nan, "avg_mae_r": np.nan,
            "avg_mfe_r": np.nan, "avg_slippage_cost": np.nan, "avg_slippage_bps_of_notional": np.nan,
            "stop_overrun_bps": np.nan, "tail_loss_r_p05": np.nan, "best_trade_r": np.nan,
            "worst_trade_r": np.nan, "total_fees": 0.0, "total_funding": 0.0, "net_pnl_usd": 0.0,
            "no_single_trade_dominates_profit": True,
        }

    r = _r_values(trades)
    wins = r > 0
    slippage_bps = [
        (t.slippage_cost / t.notional * 10_000) if t.notional else np.nan for t in trades
    ]
    stop_trades = [t for t in trades if t.exit_reason == "STOP"]
    stop_overrun_bps = (
        float(np.mean([abs(t.exit_price - t.stop_price_initial) / t.stop_price_initial * 10_000 for t in stop_trades]))
        if stop_trades
        else np.nan
    )

    return {
        "n_trades": n,
        "win_rate": float(wins.mean()),
        "avg_r": float(r.mean()),
        "median_r": float(np.median(r)),
        "expectancy_r": float(r.mean()),
        "profit_factor": _profit_factor(r),
        "max_drawdown_r": _max_drawdown_r(trades),
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "avg_hold_minutes": float(np.mean([t.hold_minutes for t in trades])),
        "avg_mae_r": float(np.mean([t.mae_r for t in trades])),
        "avg_mfe_r": float(np.mean([t.mfe_r for t in trades])),
        "avg_slippage_cost": float(np.mean([t.slippage_cost for t in trades])),
        "avg_slippage_bps_of_notional": float(np.nanmean(slippage_bps)),
        "stop_overrun_bps": stop_overrun_bps,
        "tail_loss_r_p05": float(np.percentile(r, 5)),
        "best_trade_r": float(r.max()),
        "worst_trade_r": float(r.min()),
        "total_fees": float(sum(t.fees_paid for t in trades)),
        "total_funding": float(sum(t.funding_cost for t in trades)),
        "net_pnl_usd": float(sum(t.net_pnl_usd for t in trades)),
        "no_single_trade_dominates_profit": _no_single_trade_dominates(r),
    }


def compute_continuation_probabilities(events: list[EventRecord]) -> dict:
    """P(price continues +20%/+50%/+100% against the short thesis after
    detection) - spec section 22.3/22.7's "probability and size of
    catastrophic continuation" - arguably the single most important risk
    metric here, more so than plain win rate."""
    if not events:
        return {f"prob_continuation_{int(t*100)}pct": np.nan for t in CONTINUATION_THRESHOLDS}
    adverse = np.array([e.max_adverse_after_detection for e in events if not np.isnan(e.max_adverse_after_detection)])
    if adverse.size == 0:
        return {f"prob_continuation_{int(t*100)}pct": np.nan for t in CONTINUATION_THRESHOLDS}
    return {
        f"prob_continuation_{int(t*100)}pct": float((adverse >= t).mean()) for t in CONTINUATION_THRESHOLDS
    }


def compute_event_metrics(events: list[EventRecord]) -> dict:
    n_events = len(events)
    n_triggered = sum(1 for e in events if e.trade is not None)
    by_class: dict[str, int] = defaultdict(int)
    for e in events:
        by_class[e.pump_class] += 1
    return {
        "n_pump_events": n_events,
        "n_events_triggered_trade": n_triggered,
        "trigger_rate": (n_triggered / n_events) if n_events else np.nan,
        "events_by_class": dict(by_class),
        **compute_continuation_probabilities(events),
    }


def _group_by(items: list, key_fn) -> dict:
    groups: dict = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)
    return dict(groups)


def breakdown_by(trades: list[TradeRecord], key_fn, label: str) -> dict[str, dict]:
    """Generic "performance by X" breakdown - X can be pump_class, entry_mode,
    or any caller-supplied key derived from a TradeRecord (e.g. a liquidity
    bucket or market-regime label the caller attaches via a lookup keyed on
    event_id/symbol/entry_ts, since TradeRecord itself doesn't carry those)."""
    groups = _group_by(trades, key_fn)
    return {str(k): compute_trade_metrics(v) for k, v in groups.items()}


def breakdown_by_pump_class(trades: list[TradeRecord]) -> dict[str, dict]:
    return breakdown_by(trades, lambda t: t.pump_class, "pump_class")


def breakdown_by_entry_mode(trades: list[TradeRecord]) -> dict[str, dict]:
    return breakdown_by(trades, lambda t: t.entry_mode, "entry_mode")


def breakdown_by_exit_reason(trades: list[TradeRecord]) -> dict[str, dict]:
    return breakdown_by(trades, lambda t: t.exit_reason.split(" ")[0], "exit_reason")


def compute_full_report(trades: list[TradeRecord], events: list[EventRecord]) -> dict:
    return {
        "overall": compute_trade_metrics(trades),
        "events": compute_event_metrics(events),
        "by_pump_class": breakdown_by_pump_class(trades),
        "by_entry_mode": breakdown_by_entry_mode(trades),
        "by_exit_reason": breakdown_by_exit_reason(trades),
    }
