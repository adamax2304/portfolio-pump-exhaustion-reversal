from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.backtest.simulator import _PostPumpInputs
from pump_exhaustion_reversal.config import AppConfig
from pump_exhaustion_reversal.features.structure import atr as compute_atr
from pump_exhaustion_reversal.features.structure import compute_structure
from pump_exhaustion_reversal.screener_v2 import features as feat_mod

BAR_MS = 300_000
_CFG = AppConfig()


def _ohlcv(n: int = 30, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.6, n))
    close[:5] = 100.0
    close[5:15] = np.linspace(100, 115, 10)  # a clean run-up
    close[15:] = np.linspace(115, 108, n - 15)  # then fade
    high = close + rng.uniform(0, 0.5, n)
    low = close - rng.uniform(0, 0.5, n)
    open_ = close.copy()
    volume = rng.uniform(80, 150, n)
    volume[9] = 400.0  # climax bar volume spike
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * BAR_MS, "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": volume * close,
        }
    )


def _cvd_slice(n: int, with_oi: bool = True, with_funding: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    out = pd.DataFrame({"volume_zscore": rng.normal(0, 1, n)})
    out["open_interest"] = np.linspace(1000, 1400, n) if with_oi else np.nan
    out["funding_rate"] = np.linspace(0.0001, 0.0003, n) if with_funding else np.nan
    out["funding_percentile"] = np.linspace(50, 80, n) if with_funding else np.nan
    return out


def _truncate_arg(arg, cutoff: int, full_len: int):
    if isinstance(arg, (pd.Series, pd.DataFrame)) and len(arg) == full_len:
        return arg.iloc[:cutoff].reset_index(drop=True)
    return arg


def _assert_prefix_matches(compute_fn, df_full: pd.DataFrame, cutoff: int, *args, **kwargs):
    full_len = len(df_full)
    truncated_args = [_truncate_arg(a, cutoff, full_len) for a in args]
    truncated_kwargs = {k: _truncate_arg(v, cutoff, full_len) for k, v in kwargs.items()}
    full_result = compute_fn(df_full, *args, **kwargs)
    truncated_result = compute_fn(df_full.iloc[:cutoff].reset_index(drop=True), *truncated_args, **truncated_kwargs)
    pd.testing.assert_frame_equal(full_result.iloc[:cutoff].reset_index(drop=True), truncated_result, check_dtype=False, check_names=True)
    return full_result


def test_volume_features_causal():
    ohlcv = _ohlcv()
    cvd = _cvd_slice(len(ohlcv))
    _assert_prefix_matches(feat_mod.compute_volume_features, ohlcv, 18, cvd, 9, BAR_MS)


def test_oi_features_causal_and_missing_stays_nan():
    ohlcv = _ohlcv()
    cvd_with_oi = _cvd_slice(len(ohlcv), with_oi=True)
    _assert_prefix_matches(feat_mod.compute_oi_features, cvd_with_oi, 18, BAR_MS)

    cvd_no_oi = _cvd_slice(len(ohlcv), with_oi=False)
    result = feat_mod.compute_oi_features(cvd_no_oi, BAR_MS)
    assert result["oi_change_5m"].isna().all()
    assert result["oi_drawdown_from_peak"].isna().all()


def test_oi_regime_features_buckets_and_missing_stays_none():
    ohlcv = _ohlcv()
    cvd_with_oi = _cvd_slice(len(ohlcv), with_oi=True)
    full = _assert_prefix_matches(feat_mod.compute_oi_regime_features, cvd_with_oi, 18, BAR_MS)
    assert set(full["oi_regime"].dropna().unique()) <= {"OI_STRONG_DOWN", "OI_MILD_DOWN", "OI_FLAT", "OI_MILD_UP", "OI_STRONG_UP"}

    cvd_no_oi = _cvd_slice(len(ohlcv), with_oi=False)
    result = feat_mod.compute_oi_regime_features(cvd_no_oi, BAR_MS)
    assert result["oi_regime"].isna().all()


def test_oi_regime_features_labels_strong_up_for_a_sharp_oi_climb():
    n = 20
    b60_bars = feat_mod._bars_for(60, BAR_MS)
    oi = pd.Series([1000.0] * (b60_bars + 1) + [2000.0] * (n - b60_bars - 1))  # +100% over 60m -> well past the strong-up threshold
    cvd = pd.DataFrame({"open_interest": oi.iloc[:n].reset_index(drop=True)})
    result = feat_mod.compute_oi_regime_features(cvd, BAR_MS)
    assert result["oi_regime"].iloc[-1] == "OI_STRONG_UP"


def test_price_oi_regime_labels_and_causal():
    ohlcv = _ohlcv()
    cvd = _cvd_slice(len(ohlcv))
    full = _assert_prefix_matches(feat_mod.compute_price_oi_regime_features, ohlcv, 20, cvd, BAR_MS)
    assert set(full["price_oi_regime"].dropna().unique()) <= {"price_up_oi_up", "price_up_oi_down", "price_down_oi_up", "price_down_oi_down"}


def test_funding_features_causal_and_missing_stays_nan():
    ohlcv = _ohlcv()
    cvd_with_funding = _cvd_slice(len(ohlcv), with_funding=True)
    _assert_prefix_matches(feat_mod.compute_funding_features, cvd_with_funding, 18, BAR_MS)
    cvd_no_funding = _cvd_slice(len(ohlcv), with_funding=False)
    result = feat_mod.compute_funding_features(cvd_no_funding, BAR_MS)
    assert result["funding_zscore"].isna().all()


def test_structure_deterioration_features_causal():
    ohlcv = _ohlcv()
    atr_series = compute_atr(ohlcv, _CFG.stops.atr_length).bfill()
    structure = compute_structure(ohlcv, atr_series, _CFG.structure)
    post_pump = _PostPumpInputs(
        climax_idx=9, structure_start_idx=0, post_pump_df=ohlcv, post_pump_atr=atr_series, structure=structure,
        failed_reclaim=pd.Series([False] * len(ohlcv)), sweep_rejection=pd.Series([False] * len(ohlcv)),
        post_break=pd.DataFrame(index=ohlcv.index), cvd_divergence=None, cvd_rollover=None, cvd_divergence_magnitude=None,
        oi_rollover=None, oi_drop_pct_from_peak=None, rsi_divergence=None,
    )

    def _wrapped(df, atr, pp):
        return feat_mod.compute_structure_deterioration_features(df, pp, atr)

    full_result = _wrapped(ohlcv, atr_series, post_pump)
    truncated_ohlcv = ohlcv.iloc[:20].reset_index(drop=True)
    truncated_atr = atr_series.iloc[:20].reset_index(drop=True)
    truncated_structure = compute_structure(truncated_ohlcv, truncated_atr, _CFG.structure)
    truncated_post_pump = _PostPumpInputs(
        climax_idx=9, structure_start_idx=0, post_pump_df=truncated_ohlcv, post_pump_atr=truncated_atr, structure=truncated_structure,
        failed_reclaim=pd.Series([False] * 20), sweep_rejection=pd.Series([False] * 20), post_break=pd.DataFrame(index=truncated_ohlcv.index),
        cvd_divergence=None, cvd_rollover=None, cvd_divergence_magnitude=None, oi_rollover=None, oi_drop_pct_from_peak=None, rsi_divergence=None,
    )
    truncated_result = _wrapped(truncated_ohlcv, truncated_atr, truncated_post_pump)
    pd.testing.assert_frame_equal(full_result.iloc[:20].reset_index(drop=True), truncated_result, check_dtype=False)


def test_wick_rejection_features_causal_and_bounded():
    ohlcv = _ohlcv()
    atr_series = compute_atr(ohlcv, _CFG.stops.atr_length).bfill()
    full = _assert_prefix_matches(feat_mod.compute_wick_rejection_features, ohlcv, 18, atr_series)
    valid = full["close_location_in_range"].dropna()
    assert (valid >= -1e-9).all() and (valid <= 1 + 1e-9).all()


def test_volatility_features_causal():
    ohlcv = _ohlcv()
    atr_series = compute_atr(ohlcv, _CFG.stops.atr_length).bfill()
    full = _assert_prefix_matches(feat_mod.compute_volatility_features, ohlcv, 20, atr_series, 9, BAR_MS)
    assert full["vol_after_climax_ratio"].iloc[:9].isna().all()  # NaN strictly before the climax bar


def test_pump_age_stage_features_causal_and_monotonic_age():
    ohlcv = _ohlcv()
    full = _assert_prefix_matches(feat_mod.compute_pump_age_stage_features, ohlcv, 18, 5, 9, BAR_MS)
    assert (full["minutes_since_pump_start"].diff().dropna() == 5.0).all()  # BAR_MS=5m -> +5 min per bar, always increasing


def test_market_context_features_causal():
    ohlcv = _ohlcv()
    rng = np.random.default_rng(9)
    btc_close = 60_000.0 * np.exp(np.cumsum(rng.normal(0, 0.001, len(ohlcv) + 50)))
    btc_ohlcv = pd.DataFrame({"timestamp": np.arange(len(ohlcv) + 50) * BAR_MS, "close": btc_close})
    _assert_prefix_matches(feat_mod.compute_market_context_features, ohlcv, 18, btc_ohlcv, BAR_MS)
