"""Stage 1.5 research/diagnostic phase (see project plan). Pure analysis over
already-saved event studies and raw OHLCV - physically separate from
backtest/simulator.py and strategy/*. Nothing here is imported by the trading
codepath (entry_engine/risk_engine/exit_engine/simulator), and nothing here
should ever be - most of this package looks at *future* bars relative to a
pump's detection timestamp (eventual peak, forward drawdown), which is
research/labeling only and would be a lookahead bug if it ever leaked into a
live or backtested trading decision.
"""
