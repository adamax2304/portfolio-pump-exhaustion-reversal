"""Walk-forward / out-of-sample split + regime segmentation (spec 22.6) and
parameter sensitivity sweeps (spec 22.5).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig
from . import event_study, metrics
from .simulator import DEFAULT_MAX_EVENT_HORIZON_BARS, TradeRecord


def split_chronological(
    events_df: pd.DataFrame, train_pct: float, validation_pct: float, oos_pct: float, ts_col: str = "pump_detected_ts"
) -> dict[str, pd.DataFrame]:
    """spec section 22.6: first train_pct chronologically -> train, next
    validation_pct -> validation, remainder -> out-of-sample. No shuffling -
    a random split would leak future information into "training"."""
    if abs(train_pct + validation_pct + oos_pct - 1.0) > 1e-6:
        raise ValueError("train/validation/oos pct must sum to 1.0")
    ordered = events_df.sort_values(ts_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + validation_pct))
    return {
        "train": ordered.iloc[:train_end].reset_index(drop=True),
        "validation": ordered.iloc[train_end:val_end].reset_index(drop=True),
        "oos": ordered.iloc[val_end:].reset_index(drop=True),
    }


def label_market_regime(events_df: pd.DataFrame) -> pd.DataFrame:
    """Adds regime_trend (bull/sideways/bear, from the event's concurrent BTC
    return), regime_weekday (weekday/weekend), regime_session (asia/eu/us by
    UTC hour) - spec 22.6's "bull/bear/sideways ... weekend vs weekday ...
    Asia/EU/US sessions" segmentation."""
    if events_df.empty:
        return events_df.assign(regime_trend=[], regime_weekday=[], regime_session=[])
    df = events_df.copy()
    df["regime_trend"] = pd.cut(
        df["btc_return_concurrent"], bins=[-np.inf, -0.02, 0.02, np.inf], labels=["bear", "sideways", "bull"]
    ).astype(str)
    dt = pd.to_datetime(df["pump_detected_ts"], unit="ms", utc=True)
    df["regime_weekday"] = np.where(dt.dt.dayofweek >= 5, "weekend", "weekday")
    df["regime_session"] = pd.cut(
        dt.dt.hour, bins=[-1, 7, 15, 23], labels=["asia", "eu", "us"]
    ).astype(str)
    return df


def _trades_for_events(trades_df: pd.DataFrame, event_ids: set[str]) -> list[TradeRecord]:
    if trades_df.empty:
        return []
    subset = trades_df[trades_df["event_id"].isin(event_ids)]
    return [TradeRecord(**row) for row in subset.to_dict("records")]


def evaluate_split_metrics(events_df: pd.DataFrame, trades_df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Trade-level metrics computed separately for each chronological split."""
    return {
        name: metrics.compute_trade_metrics(_trades_for_events(trades_df, set(split_df["event_id"])))
        for name, split_df in splits.items()
    }


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def parameter_sweep(
    universe: list[dict],
    timeframe: str,
    base_cfg: AppConfig,
    param_grid: dict[str, list],
    catalyst_df: pd.DataFrame | None = None,
    equity: float = 5000.0,
    horizon_bars: int = DEFAULT_MAX_EVENT_HORIZON_BARS,
) -> pd.DataFrame:
    """spec section 22.5. param_grid: {"pump_detector.min_pump_score": [60,70,80], ...}
    (dotted paths into the config, applied on top of base_cfg for each
    combination). One full event study runs per combination - keep the grid
    modest (spec: "не оптимизировать вслепую", avoid an unconstrained
    Cartesian search); this function itself does not sample down for you.
    Returns one row per combination with its parameters + trade metrics.
    """
    keys = list(param_grid.keys())
    rows = []
    for combo in itertools.product(*param_grid.values()):
        overrides = dict(zip(keys, combo))
        cfg_dict = base_cfg.model_dump(mode="python")
        for dotted_key, value in overrides.items():
            _set_dotted(cfg_dict, dotted_key, value)
        cfg = AppConfig.model_validate(cfg_dict)

        _events_df, trades_df = event_study.run_event_study(universe, timeframe, cfg, catalyst_df, equity, horizon_bars)
        trades = [TradeRecord(**row) for row in trades_df.to_dict("records")] if not trades_df.empty else []
        rows.append({**overrides, **metrics.compute_trade_metrics(trades)})
    return pd.DataFrame(rows)
