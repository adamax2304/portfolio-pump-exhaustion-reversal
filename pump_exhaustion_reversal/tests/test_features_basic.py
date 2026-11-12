from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.features import funding as funding_feat
from pump_exhaustion_reversal.features import returns as returns_feat
from pump_exhaustion_reversal.features import rsi as rsi_feat
from pump_exhaustion_reversal.features import volume as volume_feat


# ---- returns.py -----------------------------------------------------------


def test_log_return_matches_manual_calc():
    close = pd.Series([1.0, 1.0, 2.0, 2.0, 4.0])
    lr = returns_feat.log_return(close, periods=1)
    assert np.isnan(lr.iloc[0])
    assert lr.iloc[1] == pytest.approx(0.0)
    assert lr.iloc[2] == pytest.approx(np.log(2.0))
    assert lr.iloc[4] == pytest.approx(np.log(2.0))


def test_return_zscore_never_divides_by_true_zero():
    window_ret = pd.Series([0.10] * 5)
    zero_vol = pd.Series([0.0] * 5)
    z = returns_feat.return_zscore(window_ret, zero_vol, window_bars=15)
    assert np.isfinite(z).all()
    assert (z > 0).all()


def test_rolling_beta_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    btc_ret = pd.Series(rng.normal(0, 0.01, 500))
    coin_ret = 2.0 * btc_ret  # exact, noiseless -> beta must converge to 2.0
    beta = returns_feat.rolling_beta(coin_ret, btc_ret, lookback=100)
    assert beta.iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_abnormal_return_equals_coin_return_when_beta_zero():
    coin_ret = pd.Series([0.05, -0.03, 0.10])
    btc_ret = pd.Series([0.02, 0.01, -0.02])
    beta = pd.Series([0.0, 0.0, 0.0])
    out = returns_feat.abnormal_return(coin_ret, btc_ret, beta)
    pd.testing.assert_series_equal(out, coin_ret, check_names=False)


def test_compute_return_features_smoke():
    n = 300
    idx = pd.RangeIndex(n)
    rng = np.random.default_rng(1)
    btc_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))), index=idx)
    coin_close = pd.Series(1.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=idx)
    windows_ms = {"15m": 900_000, "1h": 3_600_000}
    out = returns_feat.compute_return_features(
        coin_close, btc_close, bar_interval_ms=300_000, windows_ms=windows_ms, beta_lookback_days=1
    )
    assert list(out.index) == list(idx)
    for label in windows_ms:
        assert f"return_{label}" in out.columns
        assert f"abnormal_return_{label}" in out.columns
        assert f"return_zscore_{label}" in out.columns


# ---- volume.py --------------------------------------------------------


def test_volume_zscore_constant_series_is_nan_not_error():
    vol = pd.Series([100.0] * 30)
    z = volume_feat.volume_zscore(vol, lookback=10)
    assert z.iloc[-1] != z.iloc[-1] or pd.isna(z.iloc[-1])  # NaN, std=0 -> no crash


def test_volume_decay_ratio_and_flags():
    current = pd.Series([100.0, 80.0, 50.0, 30.0, 10.0])
    ratio = volume_feat.volume_decay_ratio(current, pump_peak_volume=100.0)
    assert ratio.tolist() == pytest.approx([1.0, 0.8, 0.5, 0.3, 0.1])
    flags = volume_feat.volume_decay_flags(ratio, threshold=0.50, strong_threshold=0.35)
    assert flags["volume_decayed"].tolist() == [False, False, True, True, True]
    assert flags["volume_strongly_decayed"].tolist() == [False, False, False, True, True]


def test_volume_decay_ratio_handles_missing_peak():
    current = pd.Series([1.0, 2.0])
    assert volume_feat.volume_decay_ratio(current, pump_peak_volume=None).isna().all()
    assert volume_feat.volume_decay_ratio(current, pump_peak_volume=0).isna().all()


def test_volume_zscore_excludes_current_bar_from_its_own_baseline():
    # a single huge spike must not pull its own baseline mean/std up enough
    # to dampen its own z-score - regression test for a real bug found
    # against live Bybit/Binance data (see features/volume.py docstring).
    # Small noise on the baseline (not a perfectly flat series) avoids the
    # unrelated zero-variance-denominator edge case.
    rng = np.random.default_rng(0)
    volume = pd.Series(list(100.0 + rng.normal(0, 2, 25)) + [3000.0])
    z_including_self = (volume - volume.rolling(20, min_periods=10).mean()) / volume.rolling(20, min_periods=10).std()
    z_correct = volume_feat.volume_zscore(volume, lookback=20)
    assert z_correct.iloc[-1] > z_including_self.iloc[-1] * 3  # excluding self gives a dramatically larger z-score
    assert z_correct.iloc[-1] > 15  # for a 30x spike against a near-flat baseline, z-score should be huge, not ~4


# ---- funding.py --------------------------------------------------------


def test_funding_percentile_ranks_latest_extreme_high():
    funding = pd.Series([0.0001] * 40 + [0.01])
    pct = funding_feat.funding_percentile(funding, lookback=40)
    assert pct.iloc[-1] == pytest.approx(100.0)


def test_funding_heat_flags_thresholds():
    pct = pd.Series([50.0, 91.0, 98.0])
    flags = funding_feat.funding_heat_flags(pct, overheated_pct=90, extreme_pct=97)
    assert flags["funding_overheated"].tolist() == [False, True, True]
    assert flags["funding_extreme"].tolist() == [False, False, True]


# ---- rsi.py --------------------------------------------------------


def test_rsi_pure_uptrend_is_near_100():
    close = pd.Series(np.arange(1, 40, dtype=float))
    r = rsi_feat.rsi(close, length=14)
    assert r.iloc[-1] == pytest.approx(100.0, abs=0.01)


def test_rsi_pure_downtrend_is_near_0():
    close = pd.Series(np.arange(40, 1, -1, dtype=float))
    r = rsi_feat.rsi(close, length=14)
    assert r.iloc[-1] == pytest.approx(0.0, abs=0.01)


def test_rsi_bearish_divergence_detects_price_hh_rsi_lh():
    # price: rising, makes a fresh high at the end while RSI has already rolled over
    high = pd.Series([1, 2, 3, 5, 5.5])
    rsi_values = pd.Series([50, 60, 90, 70, 65])  # RSI peaked at bar 2, price peaks at bar 4
    div = rsi_feat.rsi_bearish_divergence(high, rsi_values, lookback_bars=5)
    assert bool(div.iloc[-1]) is True


def test_rsi_bearish_divergence_false_when_both_make_new_highs_together():
    high = pd.Series([1, 2, 3, 4, 5])
    rsi_values = pd.Series([50, 60, 70, 80, 90])
    div = rsi_feat.rsi_bearish_divergence(high, rsi_values, lookback_bars=5)
    assert bool(div.iloc[-1]) is False
