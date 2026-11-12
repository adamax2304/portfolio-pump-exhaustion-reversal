"""Stage 2A Phase 2B: Entry Edge Validation. Converts first_passage.py's
binary profit/adverse/timeout outcomes into R-denominated trade economics
(fees, slippage, funding, stop overrun, net expectancy) - answering whether
the favorable first-passage shift found in Phase 2A/2B's first pass survives
becoming an actual tradeable edge, not just a shift in a binary probability.

Everything here is diagnostic-only: no production entry/risk/exit logic is
touched, and none of this is read by backtest/simulator.py or strategy/*
(see research/__init__.py's lookahead-safety framing) - "real entry price"
here means picking among ALREADY-KNOWN historical bars (confirmation close,
next bar's open), never a price that wasn't actually knowable at decision
time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.simulator import EventRecord
from ..config import ExecutionModelConfig
from . import first_passage as fp
from . import stats as research_stats

# Phase 2B's own ATR-multiple stop grid (sensitivity, not optimization -
# report every value, never silently pick a "best" one).
ATR_STOP_MULTIPLES: tuple[float, ...] = (0.15, 0.25, 0.40)

# Phase 2B's continuation-risk thresholds (user's exact list).
CONTINUATION_UPSIDE_PCTS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
CONTINUATION_DOWNSIDE_R_MULTIPLES: tuple[float, ...] = (1.0, 2.0, 3.0)


def resolve_entry_price(ohlcv: pd.DataFrame, signal_bar_idx: int, mode: str) -> float | None:
    """Two, and only two, real (already-knowable-at-decision-time) entry
    price choices - never a theoretical intra-candle fill:
      - "confirmation_close": the signal bar's own close (the price at which
        the failed-reclaim/structure condition was itself confirmed true -
        only knowable once that bar has closed).
      - "next_bar_open": the following bar's open (lookahead-safe in the
        strictest sense - matches backtest/simulator.py's _simulate_trade
        fill convention).
    Returns None if the requested bar is out of range."""
    if mode == "confirmation_close":
        if signal_bar_idx < 0 or signal_bar_idx >= len(ohlcv):
            return None
        return float(ohlcv["close"].iloc[signal_bar_idx])
    if mode == "next_bar_open":
        next_idx = signal_bar_idx + 1
        if next_idx < 0 or next_idx >= len(ohlcv):
            return None
        return float(ohlcv["open"].iloc[next_idx])
    raise ValueError(f"unknown entry mode {mode!r}")


def compute_atr_stop_price(
    ohlcv: pd.DataFrame, signal_bar_idx: int, atr_multiple: float, atr_length: int = 14
) -> float | None:
    """Stop ABOVE the signal bar's own high by atr_multiple * ATR (short
    thesis) - a fixed-multiple structural buffer, not the full
    risk_engine.compute_structural_stop machinery (that also widens/caps
    against min/max_stop_bps, which is a production risk-sizing concern out
    of scope for this sensitivity pass)."""
    if signal_bar_idx < 0 or signal_bar_idx >= len(ohlcv):
        return None
    atr_series = _atr(ohlcv, atr_length)
    atr_value = atr_series.iloc[signal_bar_idx]
    if pd.isna(atr_value):
        atr_value = (ohlcv["high"] - ohlcv["low"]).iloc[: signal_bar_idx + 1].mean()
    if pd.isna(atr_value):
        return None
    signal_high = float(ohlcv["high"].iloc[signal_bar_idx])
    return signal_high + atr_multiple * float(atr_value)


def _atr(ohlcv: pd.DataFrame, length: int) -> pd.Series:
    # Local, dependency-light ATR (Wilder-style true range mean) - avoids
    # importing features/structure.py's atr() purely to keep this research
    # module's import surface minimal; same formula.
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


def first_passage_from_entry(
    ohlcv: pd.DataFrame, walk_start_idx: int, entry_price: float, profit_barrier_pct: float, adverse_barrier_pct: float,
    horizon_bars: int,
) -> dict:
    """Same short-thesis barrier walk as first_passage.first_passage_outcome,
    but parameterized by an EXPLICIT entry_price and an explicit first bar to
    check (inclusive) - needed because Phase 2B's two real entry modes don't
    always want "the signal bar's own close" as the reference price the way
    first_passage_outcome's anchor-comparison table does:
      - confirmation_close: walk_start_idx = signal_bar_idx + 1, entry_price
        = the signal bar's own close (reproduces first_passage_outcome(...,
        anchor_idx=signal_bar_idx, ...) exactly).
      - next_bar_open: walk_start_idx = signal_bar_idx + 1 (the SAME bar as
        the fill), entry_price = that bar's open - its own high/low after
        the open print still count toward the barrier walk, unlike
        confirmation_close where that bar has already closed."""
    n = len(ohlcv)
    if walk_start_idx < 0 or walk_start_idx >= n or not entry_price or pd.isna(profit_barrier_pct) or pd.isna(adverse_barrier_pct):
        return fp._empty_outcome("timeout", 0, True)
    profit_price = entry_price * (1 - profit_barrier_pct)
    adverse_price = entry_price * (1 + adverse_barrier_pct)
    end_idx = min(n - 1, walk_start_idx + horizon_bars - 1)
    right_censored = end_idx < walk_start_idx + horizon_bars - 1 and end_idx == n - 1

    mae_pct = 0.0
    mfe_pct = 0.0
    for i in range(walk_start_idx, end_idx + 1):
        low, high, close = float(ohlcv["low"].iloc[i]), float(ohlcv["high"].iloc[i]), float(ohlcv["close"].iloc[i])
        mae_pct = max(mae_pct, (high - entry_price) / entry_price)
        mfe_pct = max(mfe_pct, (entry_price - low) / entry_price)
        hit_adverse = high >= adverse_price
        hit_profit = low <= profit_price
        if hit_adverse:  # same-candle ambiguity resolved conservatively: adverse wins ties
            return {
                "outcome": "adverse", "bars": i - walk_start_idx + 1, "right_censored": False,
                "mae_pct": mae_pct, "mfe_pct": mfe_pct, "return_pct_at_exit": (entry_price - close) / entry_price,
            }
        if hit_profit:
            return {
                "outcome": "profit", "bars": i - walk_start_idx + 1, "right_censored": False,
                "mae_pct": mae_pct, "mfe_pct": mfe_pct, "return_pct_at_exit": (entry_price - close) / entry_price,
            }

    exit_close = float(ohlcv["close"].iloc[end_idx]) if end_idx >= walk_start_idx else entry_price
    return {
        "outcome": "timeout", "bars": end_idx - walk_start_idx + 1, "right_censored": right_censored,
        "mae_pct": mae_pct, "mfe_pct": mfe_pct, "return_pct_at_exit": (entry_price - exit_close) / entry_price,
    }


def resolve_exit_slippage_bps(outcome: str, return_pct_at_exit: float, execution_cfg: ExecutionModelConfig, high_vol: bool) -> float:
    """Phase 2C fix: matches backtest/execution_model.py's actual fill
    functions exactly, as called from _simulate_trade - a STOP-style exit
    (simulate_stop_fill) uses `stop_slippage_bps` ALONE; a TP-style exit
    (simulate_tp_fill) uses `default_slippage_bps` (x high_vol_slippage_multiplier
    if high_vol). These are MUTUALLY EXCLUSIVE per exit leg in the canonical
    model - never both. A "timeout" (neither barrier touched within the
    horizon) has no barrier-defined fill type of its own; this mirrors
    _simulate_trade's TIME_STOP branch, which fills TP-style if the bar
    closed favorably (close_j < entry_price) and STOP-style otherwise -
    i.e. directional, based on which side of entry the price ended up on.

    Phase 2B's original compute_r_trades charged a flat "2x default_slippage_bps"
    (implicitly one default-rate leg for entry AND one for exit) and THEN
    added stop_slippage_bps again on top for adverse trades - double-counting
    the exit leg's slippage on every stop-hit trade. This function is the fix:
    called once per trade for the EXIT leg only (the entry leg always simply
    uses default_slippage_bps x high_vol_multiplier, computed separately in
    compute_r_trades)."""
    if outcome == "adverse":
        return execution_cfg.stop_slippage_bps
    if outcome == "profit":
        return execution_cfg.default_slippage_bps * (execution_cfg.high_vol_slippage_multiplier if high_vol else 1.0)
    # timeout: directional fill, matches strategy/exit_engine.check_time_stop's own tp_fill-vs-stop_fill split
    if pd.notna(return_pct_at_exit) and return_pct_at_exit > 0:
        return execution_cfg.default_slippage_bps * (execution_cfg.high_vol_slippage_multiplier if high_vol else 1.0)
    return execution_cfg.stop_slippage_bps


def compute_r_trades(
    outcomes_with_stop: list[dict], profit_r_multiple: float, adverse_r_multiple: float, execution_cfg: ExecutionModelConfig,
    bar_interval_ms: int,
) -> list[dict]:
    """Converts a batch of (first_passage_outcome result + stop_distance_pct)
    dicts into R-denominated trade economics. `stop_distance_pct` must be
    the SAME basis first_passage_outcome's barriers were resolved from (see
    first_passage.resolve_r_based_barriers) - required per-trade since ATR-
    based stops vary by event/anchor, not a single project-wide constant.

    Each input dict may optionally carry "high_vol": bool (defaults False -
    matches _simulate_trade's `vol_ratio_at_entry > 2.0` test, decided once
    at entry and reused for every slippage leg of that trade) and
    "funding_r": float (defaults 0.0 - a precomputed REAL R-denominated
    funding contribution over the trade's actual holding period, built from
    real per-bar funding_rate data by the caller - see
    scripts/run_phase_2b_entry_edge.py's _compute_funding_r. This module
    stays feature_frame-agnostic on purpose; it only does the R arithmetic).

    Entry-leg and exit-leg slippage are tracked SEPARATELY (entry always at
    default_slippage_bps x high-vol multiplier; exit depends on how the
    trade concluded - see resolve_exit_slippage_bps) - Phase 2B's original
    version combined them into one flat "2x default" term and then added a
    separate stop-overrun charge on top for adverse trades, double-counting
    the exit leg. `stop_overrun_premium_r` here is purely INFORMATIONAL: the
    portion of exit_slippage_r attributable to stop_slippage_bps exceeding
    the default rate - already included in exit_slippage_r, not an
    additional subtraction in net_r.
    """
    results = []
    for o in outcomes_with_stop:
        stop_distance_pct = o.get("stop_distance_pct")
        if stop_distance_pct is None or pd.isna(stop_distance_pct) or stop_distance_pct <= 0:
            continue
        outcome = o["outcome"]
        if outcome == "profit":
            gross_r = profit_r_multiple
        elif outcome == "adverse":
            gross_r = -adverse_r_multiple
        else:  # timeout - the actual unrealized return at the horizon's end, converted to R
            gross_r = o["return_pct_at_exit"] / stop_distance_pct if not pd.isna(o.get("return_pct_at_exit")) else np.nan
        high_vol = bool(o.get("high_vol", False))
        return_pct_at_exit = o.get("return_pct_at_exit")
        entry_slippage_bps = execution_cfg.default_slippage_bps * (execution_cfg.high_vol_slippage_multiplier if high_vol else 1.0)
        exit_slippage_bps = resolve_exit_slippage_bps(outcome, return_pct_at_exit, execution_cfg, high_vol)
        entry_slippage_r = entry_slippage_bps / 10_000 / stop_distance_pct
        exit_slippage_r = exit_slippage_bps / 10_000 / stop_distance_pct
        fees_r = 2 * execution_cfg.taker_fee_bps / 10_000 / stop_distance_pct
        stop_overrun_premium_r = (
            max(0.0, execution_cfg.stop_slippage_bps - execution_cfg.default_slippage_bps) / 10_000 / stop_distance_pct
            if exit_slippage_bps == execution_cfg.stop_slippage_bps
            else 0.0
        )
        funding_r = float(o.get("funding_r", 0.0))
        net_r = gross_r - fees_r - entry_slippage_r - exit_slippage_r + funding_r
        results.append(
            {
                "outcome": outcome, "gross_r": gross_r, "fees_r": fees_r,
                "entry_slippage_r": entry_slippage_r, "exit_slippage_r": exit_slippage_r,
                "stop_overrun_premium_r": stop_overrun_premium_r, "funding_r": funding_r, "net_r": net_r,
                "mae_r": o["mae_pct"] / stop_distance_pct if not pd.isna(o.get("mae_pct")) else np.nan,
                "mfe_r": o["mfe_pct"] / stop_distance_pct if not pd.isna(o.get("mfe_pct")) else np.nan,
            }
        )
    return results


def summarize_net_expectancy(r_trades: list[dict]) -> dict:
    """Table 5's aggregate: gross/fees/entry-slippage/exit-slippage/funding/net
    R means, profit factor, win rate, avg win/loss R, avg MAE/MFE R."""
    if not r_trades:
        return {"n": 0}
    df = pd.DataFrame(r_trades)
    net = df["net_r"].dropna()
    wins = net[net > 0]
    losses = net[net <= 0]
    gains = wins.sum()
    loss_sum = -losses.sum()
    return {
        "n": len(df),
        "n_profit": int((df["outcome"] == "profit").sum()),
        "n_adverse": int((df["outcome"] == "adverse").sum()),
        "n_timeout": int((df["outcome"] == "timeout").sum()),
        "gross_expectancy_r": float(df["gross_r"].mean()),
        "fees_r": float(df["fees_r"].mean()),
        "entry_slippage_r": float(df["entry_slippage_r"].mean()),
        "exit_slippage_r": float(df["exit_slippage_r"].mean()),
        "stop_overrun_premium_r": float(df["stop_overrun_premium_r"].mean()),
        "funding_r": float(df["funding_r"].mean()),
        "net_expectancy_r": float(net.mean()) if len(net) else np.nan,
        "profit_factor": float(gains / loss_sum) if loss_sum > 0 else (np.inf if gains > 0 else np.nan),
        "win_rate": float((net > 0).mean()) if len(net) else np.nan,
        "avg_win_r": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_r": float(losses.mean()) if len(losses) else np.nan,
        "avg_mae_r": float(df["mae_r"].dropna().mean()) if df["mae_r"].notna().any() else np.nan,
        "avg_mfe_r": float(df["mfe_r"].dropna().mean()) if df["mfe_r"].notna().any() else np.nan,
    }


def bootstrap_net_expectancy_ci(r_trades: list[dict], n_resamples: int = 2000, ci: float = 0.95) -> dict:
    """Bootstrap 95% CI for win probability and mean net expectancy R -
    N=72-ish is too small to trust a point estimate alone (spec section 11).
    Profit factor's CI is intentionally NOT bootstrapped here - it is a
    ratio of two resampled sums and can blow up/flip sign under resampling
    in a way that a symmetric percentile CI badly misrepresents at this
    N; report the point estimate only and lean on the win-rate/expectancy
    CIs instead."""
    if not r_trades:
        return {"n": 0}
    df = pd.DataFrame(r_trades)
    net = df["net_r"].dropna()
    win_indicator = (net > 0).astype(float)
    return {
        "n": len(net),
        "win_rate_ci": research_stats.bootstrap_ci(win_indicator, statistic=np.mean, n_resamples=n_resamples, ci=ci),
        "net_expectancy_ci": research_stats.bootstrap_ci(net, statistic=np.mean, n_resamples=n_resamples, ci=ci),
    }


STOP_DISTANCE_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)


def stop_distance_audit(candidates_df: pd.DataFrame, execution_cfg: ExecutionModelConfig) -> pd.DataFrame:
    """Phase 2C point 2: distribution of the REAL structural stop distance
    actually used (stop_price vs entry_price, both real - not Table 1's
    anchor-agnostic ATR proxy), split by exchange and ATR-stop-multiple,
    P10/25/50/75/90, plus round_trip_cost_over_stop_distance_pct_median - the
    ratio the user flagged as the likely real culprit ("if median stop is
    only 40-60bps, a 10-30bps execution cost becomes huge in R"). Fixed to
    entry_mode='confirmation_close' and de-duplicated per (event, ATR
    multiple) - candidates_df has one row per barrier family too, which
    would otherwise inflate N sixfold without changing the stop-distance
    distribution itself (the barrier family choice doesn't affect the stop).
    round_trip_cost_bps here is a fixed BASE-cost proxy (entry+exit default
    slippage + 2 fee legs, NOT outcome-dependent) purely to compare against
    stop_distance_bps - not Table 5's actual per-outcome cost accounting."""
    if candidates_df.empty:
        return pd.DataFrame()
    df = candidates_df[candidates_df["entry_mode"] == "confirmation_close"].drop_duplicates(subset=["event_id", "atr_stop_multiple"]).copy()
    df["stop_distance_bps"] = df["stop_distance_pct"] * 10_000
    round_trip_cost_bps = 2 * execution_cfg.default_slippage_bps + 2 * execution_cfg.taker_fee_bps
    df["round_trip_cost_over_stop_distance_pct"] = round_trip_cost_bps / df["stop_distance_bps"] * 100

    rows = []
    for (exchange, atr_multiple), group in df.groupby(["exchange", "atr_stop_multiple"]):
        stop_bps = group["stop_distance_bps"].dropna()
        ratio = group["round_trip_cost_over_stop_distance_pct"].dropna()
        row = {"exchange": exchange, "atr_stop_multiple": atr_multiple, "n": len(group)}
        for p in STOP_DISTANCE_PERCENTILES:
            row[f"stop_distance_bps_p{p}"] = float(np.percentile(stop_bps, p)) if len(stop_bps) else np.nan
        row["round_trip_cost_over_stop_distance_pct_median"] = float(ratio.median()) if len(ratio) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["exchange", "atr_stop_multiple"]).reset_index(drop=True)


# Phase 2C point 4: ZERO/LOW/BASE/HIGH cost profiles - not picking a
# "prettiest" model, showing net expectancy's sensitivity to the cost
# assumption. BASE uses whatever execution_cfg the caller already has (the
# project's configured realistic defaults) unchanged.
COST_SENSITIVITY_PROFILES: dict[str, dict] = {
    "ZERO": {"taker_fee_bps": 0.0, "default_slippage_bps": 0.0, "stop_slippage_bps": 0.0},
    "LOW": {"taker_fee_bps": 2.0, "default_slippage_bps": 5.0, "stop_slippage_bps": 10.0},
    "BASE": {},
    "HIGH": {"taker_fee_bps": 7.5, "default_slippage_bps": 20.0, "stop_slippage_bps": 40.0},
}


def cost_sensitivity_sweep(
    outcomes_with_stop: list[dict], profit_r_multiple: float, adverse_r_multiple: float, base_execution_cfg: ExecutionModelConfig,
    bar_interval_ms: int,
) -> pd.DataFrame:
    """Point 4: re-runs the SAME trade outcomes (signals/entries/stops
    unchanged) through each cost profile in COST_SENSITIVITY_PROFILES."""
    rows = []
    for name, overrides in COST_SENSITIVITY_PROFILES.items():
        cfg = base_execution_cfg.model_copy(update=overrides) if overrides else base_execution_cfg
        r_trades = compute_r_trades(outcomes_with_stop, profit_r_multiple, adverse_r_multiple, cfg, bar_interval_ms)
        summary = summarize_net_expectancy(r_trades)
        summary["profile"] = name
        summary["taker_fee_bps"] = cfg.taker_fee_bps
        summary["default_slippage_bps"] = cfg.default_slippage_bps
        summary["stop_slippage_bps"] = cfg.stop_slippage_bps
        rows.append(summary)
    return pd.DataFrame(rows)


DEFAULT_COST_MULTIPLIERS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def cost_multiplier_curve(
    outcomes_with_stop: list[dict], profit_r_multiple: float, adverse_r_multiple: float, base_execution_cfg: ExecutionModelConfig,
    bar_interval_ms: int, multipliers: tuple[float, ...] = DEFAULT_COST_MULTIPLIERS,
) -> pd.DataFrame:
    """expectancy(cost_multiplier): scales fee/default-slippage/stop-slippage
    bps by a multiplier of base_execution_cfg (funding untouched - not the
    cost being stress-tested here) and reports net expectancy at each point."""
    rows = []
    for m in multipliers:
        cfg = base_execution_cfg.model_copy(
            update={
                "taker_fee_bps": base_execution_cfg.taker_fee_bps * m,
                "default_slippage_bps": base_execution_cfg.default_slippage_bps * m,
                "stop_slippage_bps": base_execution_cfg.stop_slippage_bps * m,
            }
        )
        r_trades = compute_r_trades(outcomes_with_stop, profit_r_multiple, adverse_r_multiple, cfg, bar_interval_ms)
        summary = summarize_net_expectancy(r_trades)
        rows.append({"cost_multiplier": m, "net_expectancy_r": summary.get("net_expectancy_r", np.nan), "n": summary.get("n", 0)})
    return pd.DataFrame(rows)


def estimate_breakeven_cost_multiplier(curve_df: pd.DataFrame) -> float | None:
    """Linear interpolation between the two adjacent cost_multiplier points
    in cost_multiplier_curve's output that bracket net_expectancy_r == 0.
    None if net expectancy never crosses zero across the given range (always
    negative even at multiplier 0.0 - not a cost problem at all; or always
    positive - costs would need to be even higher than the range tested)."""
    df = curve_df.dropna(subset=["net_expectancy_r"]).sort_values("cost_multiplier").reset_index(drop=True)
    for i in range(len(df) - 1):
        y0, y1 = df["net_expectancy_r"].iloc[i], df["net_expectancy_r"].iloc[i + 1]
        if y1 == y0:
            continue
        if (y0 >= 0 >= y1) or (y0 <= 0 <= y1):
            x0, x1 = df["cost_multiplier"].iloc[i], df["cost_multiplier"].iloc[i + 1]
            return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    return None


def simulate_staged_entry(
    ohlcv: pd.DataFrame, leg1_signal_idx: int, entry_mode: str, leg2_signal_idx: int | None, stop_price: float,
    profit_r_multiple: float, horizon_bars: int,
) -> dict | None:
    """ENTRY D (Phase 2C point 5, staged entry - an explicitly secondary,
    predefined hypothesis, not the decisive test): 0.5R at leg1_signal_idx
    (the earliest qualifying early-entry family for this event/attempt), a
    further 0.5R added ONLY if leg2_signal_idx (failed_reclaim - ENTRY C's
    own anchor) occurs AFTER leg1 and BEFORE leg1's own stop is hit - never
    averaging into a loser: if the stop is hit while only leg1 is open, leg1
    closes in full for -0.5R and leg2 never opens. Total planned risk across
    both legs is capped at 1R.

    Both legs share ONE stop price (climax_high + ATR buffer, fixed
    throughout - not moved after leg2 joins). Each leg is risk-normalized
    independently (leg1_risk_per_unit = stop_price - leg1_entry_price,
    likewise for leg2 - they generally differ, since leg2 enters at a
    different price) so each leg's own R-multiple is well-defined even
    though they share a stop PRICE, not a stop DISTANCE. The profit target
    is fixed once, off leg1's own risk unit only (profit_price = leg1_entry -
    profit_r_multiple * leg1_risk_per_unit) and does not move when leg2
    joins - a deliberate simplification (redefining "1R" mid-trade once a
    second, differently-sized risk unit joins has no single unambiguous
    reading) disclosed here rather than hidden.

    Returns None if leg1's own entry/stop geometry is invalid (e.g. stop at
    or below entry - can happen if leg1's signal bar's own high already
    exceeds the ATR-buffered stop)."""
    leg1_entry_price = resolve_entry_price(ohlcv, leg1_signal_idx, entry_mode)
    if leg1_entry_price is None or stop_price <= leg1_entry_price:
        return None
    leg1_risk_per_unit = stop_price - leg1_entry_price
    leg1_entry_idx = leg1_signal_idx + 1
    profit_price = leg1_entry_price - profit_r_multiple * leg1_risk_per_unit

    leg2_entry_price = leg2_risk_per_unit = None
    leg2_entry_idx = None
    if leg2_signal_idx is not None and leg2_signal_idx > leg1_signal_idx:
        candidate_price = resolve_entry_price(ohlcv, leg2_signal_idx, entry_mode)
        if candidate_price is not None and stop_price > candidate_price:
            leg2_entry_price = candidate_price
            leg2_risk_per_unit = stop_price - leg2_entry_price
            leg2_entry_idx = leg2_signal_idx + 1

    n = len(ohlcv)
    if leg1_entry_idx < 0 or leg1_entry_idx >= n:
        return None
    end_idx = min(n - 1, leg1_entry_idx + horizon_bars - 1)

    def r_at(price: float, leg2_active: bool) -> float:
        r = 0.5 * (leg1_entry_price - price) / leg1_risk_per_unit
        if leg2_active:
            r += 0.5 * (leg2_entry_price - price) / leg2_risk_per_unit
        return r

    leg2_active = False
    mae_r = mfe_r = 0.0
    for i in range(leg1_entry_idx, end_idx + 1):
        if not leg2_active and leg2_entry_idx is not None and i >= leg2_entry_idx:
            leg2_active = True
        high, low, close = float(ohlcv["high"].iloc[i]), float(ohlcv["low"].iloc[i]), float(ohlcv["close"].iloc[i])
        mae_r = min(mae_r, r_at(high, leg2_active))
        mfe_r = max(mfe_r, r_at(low, leg2_active))
        stop_hit = high >= stop_price
        profit_hit = low <= profit_price
        if stop_hit and profit_hit:
            profit_hit = False  # same-candle ambiguity resolved conservatively: stop wins
        if stop_hit:
            return {
                "outcome": "adverse", "bars": i - leg1_entry_idx + 1, "leg2_activated": leg2_active,
                "planned_r": (1.0 if leg2_active else 0.5), "realized_r": r_at(stop_price, leg2_active),
                "mae_r": mae_r, "mfe_r": mfe_r,
            }
        if profit_hit:
            return {
                "outcome": "profit", "bars": i - leg1_entry_idx + 1, "leg2_activated": leg2_active,
                "planned_r": (1.0 if leg2_active else 0.5), "realized_r": r_at(profit_price, leg2_active),
                "mae_r": mae_r, "mfe_r": mfe_r,
            }

    exit_close = float(ohlcv["close"].iloc[end_idx])
    return {
        "outcome": "timeout", "bars": end_idx - leg1_entry_idx + 1, "leg2_activated": leg2_active,
        "planned_r": (1.0 if leg2_active else 0.5), "realized_r": r_at(exit_close, leg2_active),
        "mae_r": mae_r, "mfe_r": mfe_r,
    }


def compare_attempts(events: list[EventRecord]) -> pd.DataFrame:
    """Table 8: attempt #1 vs #2 comparison - hit rate (did this attempt's
    own exhaustion score reach 'forming', i.e. clear a nonzero score) and how
    it concluded (which invalidation reason code, or None for
    trade/timeout). Only ever compares attempts that actually happened -
    events with attempt_count==1 contribute one row (their only attempt);
    events with attempt_count==2 contribute two."""
    rows = []
    for event in events:
        for i, score_peak in enumerate(event.attempt_exhaustion_score_peaks):
            reason = event.attempt_invalidation_reason_codes[i] if i < len(event.attempt_invalidation_reason_codes) else None
            structure_break_idx = (
                event.attempt_structure_break_local_idxs[i] if i < len(event.attempt_structure_break_local_idxs) else None
            )
            rows.append(
                {
                    "event_id": event.event_id, "attempt_number": i + 1, "exhaustion_score_peak": score_peak,
                    "had_structure_break": pd.notna(structure_break_idx),
                    "invalidation_reason_code": reason if pd.notna(reason) else None,
                    "concluded_via_trade_or_timeout": pd.isna(reason),
                    "is_final_attempt": (i + 1) == event.attempt_count,
                }
            )
    return pd.DataFrame(rows)


def summarize_attempts(attempts_df: pd.DataFrame) -> pd.DataFrame:
    """Per-attempt-number aggregate (attempt 1 vs attempt 2 side by side)."""
    if attempts_df.empty:
        return pd.DataFrame(columns=["attempt_number", "n", "mean_exhaustion_score_peak", "pct_had_structure_break", "pct_retried_further"])
    rows = []
    for attempt_number, group in attempts_df.groupby("attempt_number"):
        rows.append(
            {
                "attempt_number": attempt_number, "n": len(group),
                "mean_exhaustion_score_peak": float(group["exhaustion_score_peak"].mean()),
                "pct_had_structure_break": float(group["had_structure_break"].mean()),
                "pct_retried_further": float((~group["is_final_attempt"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("attempt_number").reset_index(drop=True)


def continuation_risk_after_failed_reclaim(
    ohlcv: pd.DataFrame, anchor_idx: int, stop_distance_pct: float, bar_interval_ms: int, horizon_hours: float = 24.0,
) -> dict:
    """Table 10: for each (upside %, downside R-multiple) pair, which comes
    first - a continuation move up by upside_pct (the short thesis failing
    further), or a favorable move down by downside_r * stop_distance_pct.
    Reuses first_passage_outcome directly: "profit" here means the DOWNSIDE
    (favorable-for-short) target was hit first, "adverse" means the UPSIDE
    continuation happened first - same same-candle-ambiguity convention
    (ties go to the adverse/upside-continuation side, the more conservative
    read for tail-risk purposes)."""
    horizon_bars = max(round(horizon_hours * 3_600_000 / bar_interval_ms), 1)
    results = {}
    for upside_pct in CONTINUATION_UPSIDE_PCTS:
        for downside_r in CONTINUATION_DOWNSIDE_R_MULTIPLES:
            downside_pct = downside_r * stop_distance_pct if not pd.isna(stop_distance_pct) else np.nan
            key = f"up{upside_pct:.0%}_before_down{downside_r:.0f}R"
            results[key] = fp.first_passage_outcome(ohlcv, anchor_idx, downside_pct, upside_pct, horizon_bars)
    return results


def summarize_continuation_risk(per_event_results: list[dict[str, dict]]) -> pd.DataFrame:
    """Aggregates continuation_risk_after_failed_reclaim() across events into
    one row per (upside, downside) pair: P(upside continuation happened
    first) = P(outcome == "adverse") in that function's framing."""
    if not per_event_results:
        return pd.DataFrame(columns=["threshold", "n", "p_continuation_first", "p_downside_first", "p_neither"])
    keys = per_event_results[0].keys()
    rows = []
    for key in keys:
        outcomes = [r[key] for r in per_event_results if key in r]
        summary = fp.summarize_outcomes(outcomes)
        rows.append(
            {
                "threshold": key, "n": summary["n"], "p_continuation_first": summary["p_adverse"],
                "p_downside_first": summary["p_profit"], "p_neither": summary["p_timeout"],
            }
        )
    return pd.DataFrame(rows)
