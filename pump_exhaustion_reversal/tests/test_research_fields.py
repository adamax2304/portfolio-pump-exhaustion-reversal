"""Stage 1.5 research-diagnostic fields on EventRecord: per-component
exhaustion timing/contribution, classifier reasons/notes, and continuous
signal-strength-at-peak snapshots. These are read-only observations of values
already computed inside _run_single_event's loop for other purposes - none of
them feed back into entry_engine/risk/exit, so this file only checks that the
new fields are populated sensibly, not that they change trading behavior."""

from __future__ import annotations

import numpy as np

from pump_exhaustion_reversal.backtest.simulator import MarketBaseline, simulate_symbol
from pump_exhaustion_reversal.strategy.exhaustion_detector import ExhaustionFlags
from pump_exhaustion_reversal.tests.test_simulator import _FAST_WARMUP_CFG, _flat_btc_ohlcv, _pump_and_reversal_ohlcv

_COMPONENT_NAMES = set(ExhaustionFlags().as_dict())


def _run_events():
    n = 2000
    ohlcv = _pump_and_reversal_ohlcv(n)
    btc = _flat_btc_ohlcv(n)
    return simulate_symbol(
        exchange="bybit", symbol="TESTUSDT", ohlcv=ohlcv, btc_ohlcv=btc, timeframe="5m", cfg=_FAST_WARMUP_CFG,
        oi_df=None, funding_df=None, catalyst_df=None, liquidity_score=80.0, equity=5000.0,
        market_baseline=MarketBaseline(), horizon_bars=200,
    )


def test_pump_class_reasons_and_notes_are_populated():
    events = _run_events()
    assert len(events) >= 1
    for ev in events:
        assert isinstance(ev.pump_class_reasons, list)
        assert len(ev.pump_class_reasons) >= 1, "classify_pump() always returns at least one reason, even for the SPOT_ORGANIC fallback"
        assert isinstance(ev.pump_class_data_confidence_notes, list)


def test_exhaustion_component_first_true_bar_has_all_ten_keys_and_only_advances_forward():
    events = _run_events()
    for ev in events:
        assert set(ev.exhaustion_component_first_true_bar) == _COMPONENT_NAMES
        for name, bar in ev.exhaustion_component_first_true_bar.items():
            assert bar is None or (isinstance(bar, int) and bar >= 0), f"{name}: {bar!r}"


def test_exhaustion_component_contribution_at_peak_present_when_score_peaked():
    events = _run_events()
    saw_nonzero_peak = False
    for ev in events:
        if ev.exhaustion_score_peak > 0:
            saw_nonzero_peak = True
            assert ev.exhaustion_component_contribution_at_peak, "a nonzero peak must have come from at least one contributing component"
            assert set(ev.exhaustion_component_contribution_at_peak) <= _COMPONENT_NAMES
    assert saw_nonzero_peak, "test fixture is expected to produce at least one event with signal beyond a flat 0 score"


def test_continuous_strength_at_peak_fields_are_finite_or_nan():
    events = _run_events()
    for ev in events:
        for value in (ev.oi_drop_pct_from_peak_at_peak, ev.volume_decay_ratio_at_peak, ev.cvd_divergence_magnitude_at_peak):
            assert isinstance(value, float)
            assert np.isnan(value) or np.isfinite(value)


def test_structure_break_and_dynamic_invalidation_bar_indices_are_ordered():
    events = _run_events()
    for ev in events:
        if ev.structure_break_bar_local_idx is not None:
            assert ev.structure_break_bar_local_idx >= 0
        if ev.dynamic_invalidation_bar_local_idx is not None:
            assert ev.dynamic_invalidation_bar_local_idx >= 0
            if ev.structure_break_bar_local_idx is not None:
                # dynamic invalidation (e.g. "new pump high") can only be evaluated
                # after a structure break has happened, so it can never precede it.
                assert ev.dynamic_invalidation_bar_local_idx >= ev.structure_break_bar_local_idx


def test_cvd_divergence_magnitude_populated_when_taker_buy_volume_present():
    # The fixture includes taker_buy_base_volume (real CVD, not the tick-rule
    # proxy) and a buy-heavy run-up followed by a sell-heavy reversal, so at
    # least one event should show a non-NaN divergence magnitude at its peak.
    events = _run_events()
    assert any(np.isfinite(ev.cvd_divergence_magnitude_at_peak) for ev in events)
