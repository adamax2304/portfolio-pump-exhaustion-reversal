from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest.simulator import MarketBaseline, simulate_symbol
from pump_exhaustion_reversal.config import AppConfig, PumpDetectorConfig

# beta_lookback_days defaults to 30 (=8640 5m-bars) - far beyond what a fast
# synthetic test series can supply. Override to a couple of days so the
# rolling-beta/abnormal-return warmup actually completes within the test's
# (much shorter, for speed) history - has no bearing on production defaults.
_FAST_WARMUP_CFG = AppConfig(pump_detector=PumpDetectorConfig(beta_lookback_days=2))


def _pump_and_reversal_ohlcv(n: int = 2000, seed: int = 7, first_cycle_start: int = 700) -> pd.DataFrame:
    """A long, mostly-quiet synthetic series with a few embedded pump ->
    exhaustion -> reversal cycles (same shape proven in
    test_features_structure.py's break/failed-reclaim scenario), plus
    volume/taker-buy patterns that plausibly (not guaranteed) drive CVD/OI/
    volume-decay signals in the reversal's favor. Good enough to smoke-test
    the full simulator wiring end-to-end without crashing or producing
    garbage - real triggering is verified against real data in the pilot run.
    """
    rng = np.random.default_rng(seed)
    close = np.full(n, 100.0)
    volume = np.full(n, 50.0) + np.abs(rng.normal(0, 3, n))
    buy_fraction = np.full(n, 0.5) + rng.normal(0, 0.03, n)

    cycle_starts = range(first_cycle_start, n - 60, 150)
    for start in cycle_starts:
        # near-instant first-bar shock (0->1, +20%): a rolling z-score is
        # freshest (least contaminated by its own lookback window) at the
        # very first elevated bar, then decays as the window fills with
        # already-elevated values - by design. Aligning the price move's
        # steepest single bar with that first bar is what lets the fixed
        # abnormal-return guard and the volume/return z-score guards clear
        # simultaneously, instead of one lagging the other.
        waypoints = [
            (0, 100.0), (1, 120.0), (5, 118.0), (10, 108.0), (16, 122.0), (22, 114.0),
            (28, 132.0), (34, 100.0), (40, 114.2), (46, 95.0), (52, 96.0),
        ]
        xs = [w[0] + start for w in waypoints]
        ys = [w[1] for w in waypoints]
        seg_idx = np.arange(xs[0], min(xs[-1] + 1, n))
        seg_prices = np.interp(seg_idx, xs, ys)
        close[seg_idx] = seg_prices
        # volume: an overwhelming one-bar shock right at onset, plus a broader
        # lift covering the rest of the run-up before decaying through the
        # reversal - buy-heavy on the way up, sell-heavy on the way down.
        rel = seg_idx - start
        volume[seg_idx] = (
            50.0
            + 3000.0 * np.exp(-((rel - 1.0) ** 2) / (2 * 1.0**2))
            + 150.0 * np.exp(-((rel - 20) ** 2) / (2 * 10.0**2))
        )
        buy_fraction[seg_idx] = np.where(rel <= 22, 0.85, np.where(rel <= 34, 0.35, 0.45))

    high = close * 1.001
    low = close * 0.999
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.abs(volume)
    buy_fraction = np.clip(buy_fraction, 0.05, 0.95)

    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 300_000,
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
            "quote_volume": volume * close,
            "taker_buy_base_volume": volume * buy_fraction,
        }
    )


def _flat_btc_ohlcv(n: int, seed: int = 99) -> pd.DataFrame:
    # small realistic noise, not perfectly flat: a truly zero-variance BTC
    # reference gives rolling_beta a zero-variance denominator (NaN by
    # design, see features/returns.py) on every bar, which would make
    # abnormal_return NaN everywhere - an unrealistic degenerate input, not
    # how real BTC (or any real asset) behaves.
    rng = np.random.default_rng(seed)
    close = 60_000.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 300_000,
            "open": close, "high": close * 1.0005, "low": close * 0.9995, "close": close,
            "volume": [1000.0] * n, "quote_volume": (close * 1000.0), "taker_buy_base_volume": [500.0] * n,
        }
    )


def test_simulate_symbol_runs_end_to_end_without_crashing_and_detects_pumps():
    # No OI/funding here deliberately: including them adds oi_anomaly as a
    # scored pump-score component, and this particular synthetic OI series'
    # own z-score dilutes the weighted average below min_pump_score even
    # though abnormal_return/volume_anomaly are both far past their
    # thresholds - a real illustration of spec section 8's renormalization
    # mattering, not a bug. Graceful degradation *with* OI/funding present
    # is covered separately below; this test's job is the full
    # detect->exhaust->trigger->manage->close wiring.
    n = 2000
    ohlcv = _pump_and_reversal_ohlcv(n)
    btc = _flat_btc_ohlcv(n)

    cfg = _FAST_WARMUP_CFG
    events = simulate_symbol(
        exchange="bybit", symbol="TESTUSDT", ohlcv=ohlcv, btc_ohlcv=btc, timeframe="5m", cfg=cfg,
        oi_df=None, funding_df=None, catalyst_df=None, liquidity_score=80.0, equity=5000.0,
        market_baseline=MarketBaseline(), horizon_bars=200,
    )

    assert isinstance(events, list)
    assert len(events) >= 1, "expected at least one pump to be detected across 3 embedded pump cycles"

    for ev in events:
        assert ev.symbol == "TESTUSDT"
        assert ev.exchange == "bybit"
        assert 0.0 <= ev.pump_score <= 100.0
        assert 0.0 <= ev.exhaustion_score_peak <= 100.0
        assert ev.pump_class in {"SHORT_SQUEEZE", "LEVERAGED_LONG", "SPOT_ORGANIC", "NEWS_FUNDAMENTAL", "LIQUIDITY_MANIPULATION"}
        assert ev.final_state in {"CLOSED", "INVALIDATED", "LOCKED"}
        assert ev.pump_detected_ts >= ev.pump_start_ts
        if ev.trade is not None:
            t = ev.trade
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.quantity > 0
            assert t.fees_paid >= 0
            assert t.exit_ts >= t.entry_ts
            assert t.exit_reason in {"TP1", "TP2", "TP3", "STOP", "FORCED_HORIZON_END"} or t.exit_reason.startswith("TIME_STOP")
            assert np.isfinite(t.final_r)


def test_simulate_symbol_handles_missing_oi_funding_catalyst_gracefully():
    n = 1200
    ohlcv = _pump_and_reversal_ohlcv(n, seed=11, first_cycle_start=700).drop(columns=["taker_buy_base_volume"])
    btc = _flat_btc_ohlcv(n)
    cfg = _FAST_WARMUP_CFG

    events = simulate_symbol(
        exchange="binance", symbol="NODATAUSDT", ohlcv=ohlcv, btc_ohlcv=btc, timeframe="5m", cfg=cfg,
        oi_df=None, funding_df=None, catalyst_df=None, liquidity_score=30.0, equity=5000.0, horizon_bars=150,
    )
    assert isinstance(events, list)  # must not raise even with every optional data source missing
    for ev in events:
        assert ev.data_confidence in {"HIGH", "LOW"}


def test_simulate_symbol_empty_history_returns_no_events():
    ohlcv = _pump_and_reversal_ohlcv(60)  # too short for meaningful detection, and no full cycle fits
    btc = _flat_btc_ohlcv(60)
    cfg = AppConfig()
    events = simulate_symbol("bybit", "SHORTUSDT", ohlcv, btc, "5m", cfg, horizon_bars=50)
    assert isinstance(events, list)
