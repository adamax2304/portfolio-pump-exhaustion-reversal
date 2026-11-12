"""Phase 3: Pump Exhaustion Screener v2 / Research Engine (see project plan
"Phase 3"). A separate, physically isolated namespace from backtest/
simulator.py, strategy/*, and research/* (Stage 1.5/Phase 2A-2D) - v1's
`SCORE > X -> SHORT` entry-rule branches (failed_reclaim, OI_ROLLOVER +
VOLUME_DECAY) were both REJECTED after rigorous validation; v1 code is kept
unchanged as a reference, never imported by v2.

v2 asks a strictly prior, different question: does the CURRENT STATE of an
active pump event carry ranking information about P(REVERSAL) vs
P(CONTINUATION) - not "is there a profitable short rule". No PnL, no entry/
risk/exit logic, no live scanner anywhere in this package. Phase 3A (the only
phase built so far) is pure descriptive/diagnostic statistics: an event-
snapshot dataset (many causal snapshots per active pump lifecycle, not one
row per event), competing-outcome forward labels, a wide continuous feature
set organized around pump DYNAMICS (velocity/acceleration/deceleration, not
just levels), and per-feature decile/effect-size/stability analysis.

Every feature module here follows the same causal discipline already
established in research/*: a snapshot at bar i's features may read ohlcv/
feature_frame bars [0, i] only. Labels are the deliberate exception - they
look forward from each snapshot by construction (that is what makes them
labels, not features) and must never be treated as a feature.
"""
