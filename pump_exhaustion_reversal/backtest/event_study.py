"""Event study (spec section 22.3): runs the full detector/classifier/
exhaustion/entry lifecycle across a universe and collects every pump event -
whether or not it ever triggered a trade - for the broader "what usually
happens after a pump" research questions. Trade-level records (the subset
that reached SHORT_TRIGGERED) come along for free, since simulator.py
produces both from one shared lifecycle per event - there is no separate,
simplified event-only codepath to drift out of sync with the real one.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from ..config import AppConfig
from ..data import storage
from .simulator import DEFAULT_MAX_EVENT_HORIZON_BARS, EventRecord, MarketBaseline, simulate_symbol


def run_event_study(
    universe: list[dict],
    timeframe: str,
    cfg: AppConfig,
    catalyst_df: pd.DataFrame | None = None,
    equity: float = 5000.0,
    horizon_bars: int = DEFAULT_MAX_EVENT_HORIZON_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """universe: one dict per symbol with keys exchange, symbol, ohlcv,
    btc_ohlcv, and optionally oi_df, funding_df, trades, liquidity_score,
    market_baseline. Returns (events_df, trades_df) across the whole universe.
    """
    all_events: list[EventRecord] = []
    for entry in universe:
        events = simulate_symbol(
            exchange=entry["exchange"],
            symbol=entry["symbol"],
            ohlcv=entry["ohlcv"],
            btc_ohlcv=entry["btc_ohlcv"],
            timeframe=timeframe,
            cfg=cfg,
            oi_df=entry.get("oi_df"),
            funding_df=entry.get("funding_df"),
            catalyst_df=catalyst_df,
            trades=entry.get("trades"),
            liquidity_score=entry.get("liquidity_score", 50.0),
            equity=equity,
            market_baseline=entry.get("market_baseline") or MarketBaseline(),
            horizon_bars=horizon_bars,
        )
        all_events.extend(events)
    return events_to_frames(all_events)


def events_to_frames(events: list[EventRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    trade_rows = []
    for e in events:
        row = asdict(e)
        trade = row.pop("trade")
        event_rows.append(row)
        if trade is not None:
            trade_rows.append(trade)
    events_df = pd.DataFrame(event_rows)
    trades_df = pd.DataFrame(trade_rows)
    return events_df, trades_df


def save_event_study(events_df: pd.DataFrame, trades_df: pd.DataFrame, run_id: str) -> dict:
    storage.ensure_data_dirs()
    events_path = storage.EVENTS_ROOT / f"events_{run_id}.parquet"
    trades_path = storage.BACKTESTS_ROOT / f"trades_{run_id}.parquet"
    events_df.to_parquet(events_path, index=False)
    trades_df.to_parquet(trades_path, index=False)
    return {"events_path": str(events_path), "trades_path": str(trades_path)}
