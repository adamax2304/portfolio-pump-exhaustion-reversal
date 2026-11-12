"""Catalyst/news event loading (spec section 4.7).

Stage 1 uses manual CSV/DB import only - no live scraping of exchange
announcements (Binance/Bybit listings, Twitter, etc. - that's Stage 2
live-scanner scope, spec section 10). Historical catalyst coverage will be
sparse; that is a documented limitation (project plan / README), not a bug.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CATALYST_CSV_COLUMNS = ["timestamp", "symbol", "event_type", "sentiment", "source", "confidence"]
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def load_catalyst_csv(path: str | Path) -> pd.DataFrame:
    """CSV columns: timestamp, symbol, event_type, sentiment, source, confidence
    (spec section 4.7). timestamp is parsed as UTC and normalized to epoch ms;
    symbol is upper-cased canonical form (see data/normalizer.py)."""
    df = pd.read_csv(path)
    missing = set(CATALYST_CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"catalyst CSV {path} missing required columns: {sorted(missing)}")
    df = df[CATALYST_CSV_COLUMNS].copy()
    # Timedelta floor-division (not .astype("int64")) is resolution-agnostic:
    # pd.to_datetime can infer second- or nanosecond-resolution datetime64
    # depending on input precision, which would silently change what an
    # int64 cast means - dividing by an explicit 1ms Timedelta does not.
    parsed = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ((parsed - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype(
        "int64"
    )
    df["symbol"] = df["symbol"].str.upper().str.replace("/", "", regex=False)
    bad_sentiment = ~df["sentiment"].isin(VALID_SENTIMENTS)
    if bad_sentiment.any():
        raise ValueError(
            f"catalyst CSV {path} has rows with sentiment outside {VALID_SENTIMENTS}: "
            f"{df.loc[bad_sentiment, 'sentiment'].unique().tolist()}"
        )
    return df.sort_values("timestamp").reset_index(drop=True)


def recent_catalysts(
    catalyst_df: pd.DataFrame, symbol: str, as_of_ms: int, lookback_hours: float, sentiment: str | None = None
) -> pd.DataFrame:
    """Rows for `symbol` within lookback_hours before (and at) as_of_ms,
    optionally filtered to a specific sentiment (e.g. 'positive' for the
    catalyst-block checks in spec section 10)."""
    if catalyst_df.empty:
        return catalyst_df
    window_start = as_of_ms - lookback_hours * 3_600_000
    mask = (
        (catalyst_df["symbol"] == symbol)
        & (catalyst_df["timestamp"] >= window_start)
        & (catalyst_df["timestamp"] <= as_of_ms)
    )
    if sentiment is not None:
        mask &= catalyst_df["sentiment"] == sentiment
    return catalyst_df.loc[mask].reset_index(drop=True)
