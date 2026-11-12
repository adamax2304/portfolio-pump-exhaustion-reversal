from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.config import AppConfig
from pump_exhaustion_reversal.data.feature_store import align_asof, build_feature_frame


def test_align_asof_backward_never_looks_ahead():
    target = pd.Series([0, 1000, 2000, 3000, 4000])
    sparse = pd.DataFrame({"timestamp": [500, 2500], "value": [10.0, 20.0]})
    out = align_asof(sparse, target, ["value"])
    # before the first sparse row -> NaN, not a future value
    assert pd.isna(out["value"].iloc[0])
    assert out["value"].iloc[1] == pytest.approx(10.0)  # ts=1000 -> most recent <= is ts=500
    assert out["value"].iloc[2] == pytest.approx(10.0)  # ts=2000 -> still ts=500 (2500 is in the future)
    assert out["value"].iloc[3] == pytest.approx(20.0)  # ts=3000 -> now ts=2500 applies
    assert out["value"].iloc[4] == pytest.approx(20.0)


def test_align_asof_empty_sparse_is_all_nan():
    target = pd.Series([0, 1000])
    out = align_asof(pd.DataFrame(columns=["timestamp", "value"]), target, ["value"])
    assert out["value"].isna().all()


def _synthetic_ohlcv(n: int, seed: int, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * 1.001
    low = close * 0.999
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.abs(rng.normal(100, 10, n))
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 300_000,
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
            "quote_volume": volume * close,
            "taker_buy_base_volume": volume * 0.5,
        }
    )


def test_build_feature_frame_runs_end_to_end_with_full_data():
    n = 200
    ohlcv = _synthetic_ohlcv(n, seed=1)
    btc = _synthetic_ohlcv(n, seed=2, start_price=60_000.0)
    oi_df = pd.DataFrame({"timestamp": ohlcv["timestamp"].to_numpy()[::5], "open_interest": np.linspace(1000, 1500, n // 5 + (n % 5 > 0))[: len(ohlcv["timestamp"].to_numpy()[::5])]})
    funding_df = pd.DataFrame({"timestamp": ohlcv["timestamp"].to_numpy()[::20], "funding_rate": np.full(len(ohlcv["timestamp"].to_numpy()[::20]), 0.0001)})

    cfg = AppConfig()
    out = build_feature_frame(ohlcv, btc, "5m", cfg, oi_df=oi_df, funding_df=funding_df)

    assert len(out) == n
    for w in cfg.pump_detector.windows:
        assert f"abnormal_return_{w}" in out.columns
        assert f"return_zscore_{w}" in out.columns
    for col in ("volume_zscore", "realized_vol_ratio", "cvd_delta", "cvd_confidence", "rsi", "open_interest", "oi_zscore", "funding_rate", "funding_percentile", "short_liq_zscore"):
        assert col in out.columns
    # taker_buy_base_volume was supplied -> CVD should use the high-confidence path
    assert (out["cvd_confidence"] == "high").all()
    # OI/funding were supplied -> not entirely NaN by the end of warmup
    assert out["open_interest"].notna().sum() > 0
    assert out["funding_rate"].notna().sum() > 0


def test_build_feature_frame_degrades_gracefully_without_oi_funding():
    n = 100
    ohlcv = _synthetic_ohlcv(n, seed=3).drop(columns=["taker_buy_base_volume"])
    btc = _synthetic_ohlcv(n, seed=4, start_price=60_000.0)
    cfg = AppConfig()

    out = build_feature_frame(ohlcv, btc, "5m", cfg, oi_df=None, funding_df=None)

    assert out["open_interest"].isna().all()
    assert out["oi_zscore"].isna().all()
    assert out["funding_rate"].isna().all()
    assert out["short_liq_zscore"].isna().all()  # no OI -> liquidation proxy can't be computed either
    assert (out["cvd_confidence"] == "low").all()  # no trades, no taker_buy_base_volume -> proxy
