from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pump_exhaustion_reversal.features import avwap as avwap_feat
from pump_exhaustion_reversal.features import cvd as cvd_feat
from pump_exhaustion_reversal.features import liquidations as liq_feat
from pump_exhaustion_reversal.features import news as news_feat
from pump_exhaustion_reversal.features import oi as oi_feat


# ---- cvd.py -----------------------------------------------------------


def test_compute_delta_from_taker_buy_volume():
    volume = pd.Series([10.0])
    taker_buy = pd.Series([6.0])
    delta = cvd_feat.compute_delta_from_taker_buy_volume(volume, taker_buy)
    assert delta.iloc[0] == pytest.approx(2.0)  # buy=6, sell=4, delta=2


def test_compute_delta_proxy_from_ohlc_sign():
    open_ = pd.Series([10.0, 10.0, 10.0])
    close = pd.Series([11.0, 9.0, 10.0])
    volume = pd.Series([5.0, 5.0, 5.0])
    delta = cvd_feat.compute_delta_proxy_from_ohlc(open_, close, volume)
    assert delta.tolist() == [5.0, -5.0, 0.0]


def test_aggregate_trades_to_bar_delta_excludes_unknown_side():
    bar_ts = pd.Series([0, 1000])
    trades = pd.DataFrame(
        {
            "timestamp": [100, 200, 1100],
            "amount": [5.0, 2.0, 3.0],
            "side": ["buy", None, "sell"],
        }
    )
    delta = cvd_feat.aggregate_trades_to_bar_delta(trades, bar_ts, bar_interval_ms=1000)
    assert delta.tolist() == [5.0, -3.0]


def test_anchored_cvd_resets_at_anchor():
    delta = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = cvd_feat.anchored_cvd(delta, anchor_idx=2)
    assert result.isna().iloc[:2].all()
    assert result.iloc[2:].tolist() == [3.0, 7.0]


def test_cvd_bearish_divergence_true_when_cvd_fails_to_confirm():
    high = pd.Series([1, 2, 3, 5, 5.5])
    cvd = pd.Series([10, 20, 40, 30, 28])  # cvd peaked at bar2, price peaks at bar4
    div = cvd_feat.cvd_bearish_divergence(high, cvd, lookback_bars=5, tolerance=0.02)
    assert bool(div.iloc[-1]) is True


def test_cvd_bearish_divergence_false_when_both_confirm():
    high = pd.Series([1, 2, 3, 4, 5])
    cvd = pd.Series([10, 20, 30, 40, 50])
    div = cvd_feat.cvd_bearish_divergence(high, cvd, lookback_bars=5, tolerance=0.02)
    assert bool(div.iloc[-1]) is False


def test_cvd_divergence_magnitude_positive_when_diverging():
    high = pd.Series([1, 2, 3, 5, 5.5])
    cvd = pd.Series([10, 20, 40, 30, 28])  # cvd peaked at bar2, price peaks at bar4 - same fixture as the boolean test above
    mag = cvd_feat.cvd_divergence_magnitude(high, cvd, lookback_bars=5)
    assert mag.iloc[-1] == pytest.approx((40 - 28) / 40)


def test_cvd_divergence_magnitude_negative_when_confirming():
    high = pd.Series([1, 2, 3, 4, 5])
    cvd = pd.Series([10, 20, 30, 40, 50])
    mag = cvd_feat.cvd_divergence_magnitude(high, cvd, lookback_bars=5)
    assert mag.iloc[-1] == pytest.approx((40 - 50) / 40)


def test_cvd_divergence_magnitude_nan_when_price_not_at_new_high():
    high = pd.Series([1, 5, 3, 2, 2])  # last bar is well off the rolling high
    cvd = pd.Series([10, 20, 40, 30, 28])
    mag = cvd_feat.cvd_divergence_magnitude(high, cvd, lookback_bars=5)
    assert pd.isna(mag.iloc[-1])


def test_cvd_rollover_flags_transition_to_negative():
    slope = pd.Series([1.0, 2.0, 0.5, -0.5, -1.0])
    rollover = cvd_feat.cvd_rollover(slope)
    assert rollover.tolist() == [False, False, False, True, False]


def test_compute_cvd_features_source_priority():
    df = pd.DataFrame(
        {
            "open": [10.0] * 5,
            "close": [11.0] * 5,
            "volume": [10.0] * 5,
            "taker_buy_base_volume": [6.0] * 5,
        }
    )
    out = cvd_feat.compute_cvd_features(df, lookback_bars=3)
    assert (out["cvd_confidence"] == "high").all()
    assert out["cvd_delta"].iloc[0] == pytest.approx(2.0)

    df_no_taker = df.drop(columns=["taker_buy_base_volume"])
    out2 = cvd_feat.compute_cvd_features(df_no_taker, lookback_bars=3)
    assert (out2["cvd_confidence"] == "low").all()
    assert out2["cvd_delta"].iloc[0] == pytest.approx(10.0)  # proxy: close>open -> +volume


# ---- oi.py --------------------------------------------------------


def test_oi_zscore_constant_series_no_crash():
    oi = pd.Series([1000.0] * 20)
    z = oi_feat.oi_zscore(oi, lookback=10)
    assert pd.isna(z.iloc[-1])


def test_detect_oi_rollover_fires_after_peak_drop():
    price = pd.Series([100.0] * 10 + [101.0] * 10)
    oi = pd.Series(list(np.linspace(1000, 1500, 10)) + list(np.linspace(1500, 1300, 10)))
    out = oi_feat.detect_oi_rollover(price, oi, lookback_bars=10, min_oi_drop_pct_after_peak=5, danger_oi_rise_pct_on_drop=8)
    assert out["oi_rollover"].iloc[-1]  # dropped from 1500 peak to 1300 = 13.3% > 5%


def test_detect_oi_rollover_danger_flag_on_price_down_oi_up():
    price = pd.Series(list(np.linspace(100, 90, 12)))  # price falling
    oi = pd.Series(list(np.linspace(1000, 1200, 12)))  # OI aggressively rising
    out = oi_feat.detect_oi_rollover(price, oi, lookback_bars=10, min_oi_drop_pct_after_peak=5, danger_oi_rise_pct_on_drop=8)
    assert out["oi_danger_new_shorts"].iloc[-1]


def test_detect_oi_rollover_is_causal_no_lookahead():
    """Phase 2D point 6: oi_rollover at bar i must depend only on oi[<=i].
    Appends a FUTURE OI spike (which, if the rolling-peak computation ever
    looked ahead, would raise the peak and suppress oi_rollover at an
    EARLIER bar) and proves the earlier bar's value is unchanged by it -
    empirical proof, not just code inspection of the rolling().max() call."""
    price_future_visible = pd.Series(list(np.linspace(100, 101, 20)) + [105.0] * 10)
    oi_future_visible = pd.Series(list(np.linspace(1000, 1500, 10)) + list(np.linspace(1500, 1300, 10)) + [2000.0] * 10)
    out_full = oi_feat.detect_oi_rollover(price_future_visible, oi_future_visible, lookback_bars=10, min_oi_drop_pct_after_peak=5, danger_oi_rise_pct_on_drop=8)

    cutoff = 20  # bar 19 is the last bar BEFORE the future OI spike (index 20+)
    out_truncated = oi_feat.detect_oi_rollover(
        price_future_visible.iloc[:cutoff], oi_future_visible.iloc[:cutoff], lookback_bars=10,
        min_oi_drop_pct_after_peak=5, danger_oi_rise_pct_on_drop=8,
    )
    assert bool(out_full["oi_rollover"].iloc[cutoff - 1]) == bool(out_truncated["oi_rollover"].iloc[cutoff - 1])
    assert out_full["oi_drop_pct_from_peak"].iloc[cutoff - 1] == pytest.approx(out_truncated["oi_drop_pct_from_peak"].iloc[cutoff - 1])


# ---- avwap.py --------------------------------------------------------


def test_anchored_vwap_matches_manual_average_with_constant_volume():
    high = pd.Series([10.0, 12.0, 14.0])
    low = pd.Series([8.0, 10.0, 12.0])
    close = pd.Series([9.0, 11.0, 13.0])
    volume = pd.Series([1.0, 1.0, 1.0])
    vwap = avwap_feat.anchored_vwap(high, low, close, volume, anchor_idx=0)
    typical = (high + low + close) / 3
    expected_last = typical.mean()  # equal volume weights -> plain average
    assert vwap.iloc[-1] == pytest.approx(expected_last)


def test_avwap_loss_and_compute_features_skips_none_anchors():
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 8.0],
            "low": [9.0, 11.0, 7.0],
            "close": [9.5, 11.5, 7.5],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    out = avwap_feat.compute_avwap_features(df, {"pump_start": 0, "breakout_candle": None})
    assert "avwap_pump_start" in out.columns
    assert "avwap_loss_pump_start" in out.columns
    assert "avwap_breakout_candle" not in out.columns
    assert bool(out["avwap_loss_pump_start"].iloc[-1]) is True  # price dropped to 7.5, below the rising VWAP


# ---- liquidations.py --------------------------------------------------------


def test_estimate_liquidation_proxy_higher_on_oi_drop_price_up_volume_spike():
    # tiny realistic noise, not perfectly flat - a truly constant volume series
    # gives a zero-variance rolling std (NaN z-score), which is an unrealistic
    # degenerate input, not the "calm market" this test means to model.
    n = 30
    rng = np.random.default_rng(0)
    df_calm = pd.DataFrame(
        {
            "high": 100.0 + rng.normal(0, 0.01, n),
            "low": 99.0 + rng.normal(0, 0.01, n),
            "close": 99.5 + rng.normal(0, 0.01, n),
            "volume": 10.0 + rng.normal(0, 0.1, n),
        }
    )
    oi_calm = pd.Series([1000.0] * n)  # perfectly flat OI -> no drop anywhere
    calm = liq_feat.estimate_liquidation_proxy(df_calm, oi_calm, lookback_bars=10)

    df_spike = df_calm.copy()
    df_spike.loc[n - 1, ["high", "low", "close"]] = [110.0, 100.0, 109.0]
    df_spike.loc[n - 1, "volume"] = 200.0
    oi_spike = oi_calm.copy()
    oi_spike.iloc[-1] = 800.0  # sharp OI drop on the spike bar
    spiked = liq_feat.estimate_liquidation_proxy(df_spike, oi_spike, lookback_bars=10)

    # flat OI never triggers the proxy's precondition (a real drop) - this
    # must be NaN (no reading to offer), not a confirmed 0, so a 20%-weighted
    # pump-score component doesn't get dragged down by an inapplicable signal
    # (confirmed against real data: this exact bug made pump_score
    # essentially unreachable for any pump without a literal short squeeze -
    # see project notes / features/liquidations.py docstring).
    assert np.isnan(calm["liq_proxy_intensity"].iloc[-1])
    assert spiked["liq_proxy_intensity"].iloc[-1] > 0
    assert np.isfinite(spiked["liq_proxy_intensity"].iloc[-1])


def test_liquidation_depletion_ratio_handles_missing_peak():
    intensity = pd.Series([1.0, 2.0])
    assert liq_feat.liquidation_depletion_ratio(intensity, None).isna().all()
    assert liq_feat.liquidation_depletion_ratio(intensity, 0).isna().all()
    assert liq_feat.liquidation_depletion_ratio(intensity, 2.0).tolist() == [0.5, 1.0]


def test_proxy_liquidation_provider_filters_by_zscore_threshold():
    # a real z-score needs a real prior baseline of OI-drop events to compare
    # against (the proxy is NaN, by design, whenever OI isn't dropping - see
    # estimate_liquidation_proxy) - so the history alternates small drops
    # (with a small concurrent price/volume uptick) and recoveries, then ends
    # on one much bigger combined spike.
    n = 24
    rng = np.random.default_rng(1)
    close = 100.0 + rng.normal(0, 0.05, n)
    oi = np.full(n, 1000.0)
    for i in range(1, n - 1, 2):
        oi[i] = oi[i - 1] * 0.99  # a small, real 1% OI drop every other bar
        close[i] = close[i - 1] * 1.002  # a small concurrent price uptick
        oi[i + 1] = oi[i] / 0.99  # recovers (OI rises -> no drop -> NaN there)
    high = close + 0.3
    low = close - 0.3
    volume = np.full(n, 10.0) + rng.normal(0, 0.3, n)
    volume[1::2] = 15.0  # a mild volume tick on each small-drop bar

    # final bar: a much bigger combined spike
    close[-1] = close[-2] * 1.05
    high[-1] = close[-1] + 2
    low[-1] = close[-2]
    volume[-1] = 200.0
    oi[-1] = oi[-2] * 0.7  # a real 30% OI drop, dwarfing the 1% history

    df = pd.DataFrame({"timestamp": np.arange(n) * 300_000, "high": high, "low": low, "close": close, "volume": volume})
    oi = pd.Series(oi)

    def fake_loader(symbol, start_ms, end_ms):
        assert symbol == "BTCUSDT"
        return df, oi

    provider = liq_feat.ProxyLiquidationProvider(fake_loader, lookback_bars=10, zscore_threshold=1.0)
    assert provider.mode == "proxy_mode"
    result = provider.get_liquidations("BTCUSDT", 0, 1_000_000)
    assert not result.empty
    assert (result["confidence"] == "proxy").all()
    assert (result["side"] == "short").all()


# ---- news.py --------------------------------------------------------


def test_load_catalyst_csv_normalizes_and_validates(tmp_path):
    csv_path = tmp_path / "catalysts.csv"
    csv_path.write_text(
        "timestamp,symbol,event_type,sentiment,source,confidence\n"
        "2026-08-01T00:00:00Z,btc/usdt,listing,positive,manual,high\n",
        encoding="utf-8",
    )
    df = news_feat.load_catalyst_csv(csv_path)
    assert df.loc[0, "symbol"] == "BTCUSDT"
    expected_ms = pd.Timestamp("2026-08-01T00:00:00Z").value // 1_000_000
    assert df.loc[0, "timestamp"] == expected_ms


def test_load_catalyst_csv_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("timestamp,symbol\n2026-08-01,BTCUSDT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        news_feat.load_catalyst_csv(csv_path)


def test_load_catalyst_csv_rejects_bad_sentiment(tmp_path):
    csv_path = tmp_path / "bad_sentiment.csv"
    csv_path.write_text(
        "timestamp,symbol,event_type,sentiment,source,confidence\n"
        "2026-08-01T00:00:00Z,BTCUSDT,listing,super_bullish,manual,high\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sentiment"):
        news_feat.load_catalyst_csv(csv_path)


def test_recent_catalysts_filters_symbol_window_and_sentiment():
    df = pd.DataFrame(
        {
            "timestamp": [1000, 5000, 5000],
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "event_type": ["listing", "listing", "listing"],
            "sentiment": ["positive", "negative", "positive"],
            "source": ["x", "x", "x"],
            "confidence": ["high", "high", "high"],
        }
    )
    # as_of before the positive BTC event exists -> not visible yet (no lookahead)
    out_before = news_feat.recent_catalysts(df, "BTCUSDT", as_of_ms=500, lookback_hours=24, sentiment="positive")
    assert len(out_before) == 0

    # as_of after it, within the lookback window -> visible
    out_after = news_feat.recent_catalysts(df, "BTCUSDT", as_of_ms=5000, lookback_hours=24, sentiment="positive")
    assert len(out_after) == 1
    assert out_after.iloc[0]["timestamp"] == 1000

    # ETH rows must never leak into a BTC query, regardless of sentiment filter
    out_all = news_feat.recent_catalysts(df, "BTCUSDT", as_of_ms=5000, lookback_hours=24)
    assert (out_all["symbol"] == "BTCUSDT").all()
    assert len(out_all) == 2
