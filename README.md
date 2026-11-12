# Pump Exhaustion Reversal

Stage 1: research / data / backtest engine for a crypto short strategy. **No live trading, no order execution, anywhere in this codebase.** `strategy.live_trading_enabled` is hard-blocked at `False` by the config schema.

## Philosophy

> Do not short strength. Short failed strength.

A pump is only a reason to *watch* a symbol. The strategy never shorts because a coin is up 30/50/100%, and never shorts on RSI alone. It waits for the pump's underlying mechanism to visibly break (structure break, failed reclaim, CVD/OI rollover, volume decay, liquidation depletion) before considering a short - and even then, a hard gate requires structural confirmation regardless of how high the pump/exhaustion scores are. See `pump_exhaustion_reversal/strategy/` for the pump detector, classifier (A-E: short squeeze / leveraged long / spot-organic / news-fundamental / liquidity-manipulation), exhaustion detector, entry engine (aggressive/standard/conservative modes), risk/exit engines, and the event-level state machine.

## Order of operations

```
DATA -> EVENT STUDY -> BACKTEST -> OOS VALIDATION -> GO/NO-GO -> (Stage 2: live scanner) -> (Stage 3: paper trading) -> (real execution)
```

Only Stage 1 (data through OOS validation + an honest GO/NO-GO report) is built here. Stages 2/3 are explicitly out of scope until this stage's acceptance criteria (`config/default.yaml` -> `acceptance:`, spec section 23) are met on out-of-sample data.

## Setup

Python 3.11+. From the repo root:

```bash
pip install -e .
```

That installs the package in editable mode; the dependency list (pandas, numpy, scipy, ccxt, pydantic, pyyaml, typer, duckdb, pyarrow, matplotlib, plotly, statsmodels, requests, pytest) is in `pyproject.toml`.

Run the tests:

```bash
python -m pytest -q
```

## CLI

```bash
# Recommended: screen first, backfill only the candidates (see "Screening" below) -
# a full 5m/15m/1h+OI+funding backfill costs ~17+ API calls/symbol, so backfilling an
# entire liquidity-filtered universe (hundreds of symbols) just to find the handful
# that ever pumped is wasteful. screen_candidates.py costs ~1-2 calls/symbol (daily bars).
python scripts/screen_candidates.py --exchange bybit --start 2026-01-01 --end 2026-08-25 --min-move-pct 0.15
python scripts/ingest.py --exchange bybit --symbols-file data/events/candidates_bybit.txt --start 2026-01-01 --end 2026-08-25 --types ohlcv,oi,funding

# Or backfill everything that passes the liquidity filter directly, no screening:
python scripts/ingest.py --exchange bybit --symbols all --start 2024-09-01 --end 2026-08-25 --types ohlcv,oi,funding

# Run the full event study + trade simulation + walk-forward split + GO/NO-GO report
python scripts/run_backtest.py --config config/backtest.yaml --start 2024-09-01 --end 2026-08-25

# Reprint/regenerate the report for a saved run
python scripts/make_report.py --backtest-id latest
```

`ingest.py --symbols all` builds the universe from the liquidity/spread/age filter in `universe:` (config), sorted by 24h quote volume descending; `--max-symbols N` caps it (useful for a quick pilot before committing to a full backfill). `run_backtest.py` reads only what's already cached under `data/raw/` - it never calls an exchange itself.

## Repo structure

```
config/                    default.yaml (full effective config) + backtest.yaml (overlay)
pump_exhaustion_reversal/
  config.py                 pydantic schema + YAML loader (deep-merge over full defaults)
  connectors/                base.py (pagination + LiquidationDataProvider interface),
                              binance.py, bybit.py (native REST for OHLCV/funding/OI),
                              ccxt_fallback.py (trades, universe/ticker snapshot)
  data/                      storage.py (parquet, merge-on-write), normalizer.py,
                              universe.py (liquidity/spread/age filter), feature_store.py
                              (assembles one symbol's full feature frame)
  features/                  returns, volume, cvd, oi, funding, liquidations (proxy),
                              avwap, structure (pivots/HH-HL/break/sweep/failed-reclaim), rsi, news
  strategy/                  scoring, pump_detector, pump_classifier, catalyst_filter,
                              exhaustion_detector, entry_engine, risk_engine, exit_engine,
                              state_machine
  backtest/                  simulator (bar-walk event lifecycle), execution_model
                              (fees/slippage/funding/participation cap), metrics, event_study,
                              walk_forward (OOS split + regime labels + parameter sweep), reports
  tests/                     ~170 tests across every module above
scripts/                    ingest.py, run_backtest.py, make_report.py
data/                       raw/ (parquet cache), features/, events/, backtests/, reports/
```

`live/` (scanner, alerting, telegram) from a hypothetical Stage 2 does not exist here - out of scope until the GO/NO-GO gate is cleared.

## Screening (find pump candidates before paying for a full backfill)

`universe:` in `config/default.yaml` currently floors at `min_24h_quote_volume_usd: 2,000,000` / `preferred_min_24h_quote_volume_usd: 15,000,000` (lowered from an initial $20M/$50M) specifically to let smaller, more pump-prone alts into the universe - mega-caps like BTC/ETH/SOL essentially never move 40%/24h, which is exactly the kind of move spec's fixed abnormal-return guards are calibrated for. But that also means the filtered universe can run into the hundreds of symbols, and most of them never pump in any given window - backfilling full 5m/15m/1h+OI+funding depth for all of them is expensive for little payoff.

`scripts/screen_candidates.py` fixes this: it fetches only **daily** OHLCV (1-2 API calls/symbol) for the whole liquidity-filtered universe, finds the largest BTC-adjusted 1/3/7-day move each symbol showed anywhere in the window (`data/screening.py:compute_screening_score`), and writes a plain comma-separated candidate list. This is a coarse capacity filter only - it is not lookahead-safe the way `strategy/pump_detector.py` is, and is never used to make a trading decision, only to decide which symbols are worth the full backfill. Feed its output straight into `ingest.py --symbols-file`.

## Parameter sweep

Spec section 22.5's parameter sensitivity sweep is a library function, not a separate CLI command (matching spec section 25's own CLI list, which doesn't list one either):

```python
from pump_exhaustion_reversal.backtest import walk_forward
from pump_exhaustion_reversal.config import load_config

cfg = load_config("config/default.yaml", "config/backtest.yaml")
universe = [...]  # same shape run_backtest.py builds internally - see scripts/run_backtest.py:_load_universe
grid = {
    "pump_detector.min_pump_score": [60, 70, 80],
    "exhaustion_thresholds.trade": [65, 75, 85],
}
result_df = walk_forward.parameter_sweep(universe, "5m", cfg, grid)  # one full event study per combination
```

Keep the grid modest - each combination re-runs the full event study over the whole universe (spec: don't optimize blindly; validate out-of-sample after, don't just pick the in-sample-best cell).

## Config

`config/default.yaml` is the full effective config (every key documented inline, mirrors `config.py`'s pydantic schema 1:1 with spec-accurate defaults). `config/backtest.yaml` is a small overlay applied on top for `run_backtest.py` (date range, base timeframe). Any YAML file can be partial - `load_config()` deep-merges it onto the full schema defaults before validation, so you only need to list the keys you're changing. Unknown keys are rejected (`extra="forbid"` on every section) to catch typos early.

## Data depth - what's real, what's a proxy, and why

Historical data availability differs sharply by source; each is handled honestly rather than pretending a uniform depth:

- **OHLCV 5m/15m/1h/4h/1d**: full requested date range is realistic (proven pagination pattern, ~300 API calls/symbol for 24 months across those 5 timeframes). This is the backtest's base timeframe (`backtest.timeframe_base: "5m"`).
- **OHLCV 1m**: the connectors support full depth, but `ingest.py` only backfills a recent window by default (`--m1-lookback-days`, default 90) - a full-universe multi-year 1m archive is on the order of 1000+ API calls *per symbol* and isn't a good use of the ingestion budget for Stage 1's research question.
- **CVD**: high-confidence wherever Binance's native `taker_buy_base_volume` klines field is available (free, full history, no extra calls) or within event-anchored public-trades windows fetched around a detected pump. Bybit klines never carry a taker-side field, so Bybit CVD outside those event windows is a low-confidence close-vs-open tick-rule proxy - flagged via `cvd_confidence`, not silently trusted.
- **Open interest**: best-effort, and sharply *asymmetric* between exchanges (verified empirically, not just from docs) - Bybit's `/v5/market/open-interest` returned a full, gapless 60-day 5m history on the first real check (17,280 bars), while Binance's `/futures/data/openInterestHist` ignored an explicit 10-day-old `startTime` and only ever returned the most recent **~1.25 days** regardless, despite the endpoint's docs suggesting a 30-day window. Treat Binance OI-derived features (oi_zscore, oi_rollover, the liquidation proxy) as essentially unavailable beyond the last ~30 hours; Bybit's are usable much further back. Either way this is an exchange-side limit, not a code one - older bars simply have missing OI features, and the scoring layer renormalizes around missing components rather than scoring them as zero (spec section 8).
- **Liquidations**: `proxy_mode` only (`features/liquidations.py` - OI-drop + price-spike + volume-spike composite). No free historical liquidation feed exists on Binance or Bybit (only real-time WS streams). `connectors/base.py:LiquidationDataProvider` stubs `real_liquidation_data` / `paid_provider_placeholder` for a future paid feed.
- **News/catalyst**: manual CSV import only (`features/news.py`, format in spec section 4.7 - `timestamp,symbol,event_type,sentiment,source,confidence`; example at `config/example_catalysts.csv`, passed via `run_backtest.py --catalyst-csv`). No live announcement scraping. Historical catalyst coverage will be sparse; `news_flag=False` on an event does not mean no catalyst existed, only that none was imported.
- **Universe spread filter**: `ccxt`'s bulk `fetch_tickers()` does not reliably return bid/ask for Binance/Bybit USDT-perp markets, so `spread_bps` is usually `NaN` and the spread gate passes by default (missing data is never treated as "blocked" - see `data/universe.py`). In practice the spread filter is not currently load-bearing; a live orderbook feed would be needed to make it one.
- **`PumpClass.SPOT_ORGANIC`** is a residual/fallback classification, not a positive detection - Stage 1 collects no spot orderflow data (linear-perp-only universe) to confirm spot-led buying.
- **`PumpClass.LIQUIDITY_MANIPULATION`** has no cross-exchange confirmation - single-venue data only.

## Acceptance gate (spec section 23)

`scripts/run_backtest.py` computes an explicit GO/NO-GO verdict on out-of-sample trades against `config/default.yaml` -> `acceptance:` (min OOS trades, OOS profit factor, OOS expectancy, no single dominant trade, positive expectancy under a doubled-slippage stress re-run). **If it says NO-GO, the strategy stays research-only** - do not build a live scanner on top of it; go back and revise the hypothesis/features/scoring instead (see spec's own closing note on this).

## Testing

`python -m pytest -q` runs the full suite. Notable coverage: `test_features_structure.py` includes the spec's two named acceptance tests at the feature level - an intact HH/HL uptrend must never report a structure break regardless of how large the move is, and a full sweep -> volume-decay -> CVD-divergence -> structure-break -> failed-reclaim sequence must produce a detectable exhaustion signal. `test_simulator.py` and `test_backtest_orchestration.py` exercise the full bar-walk lifecycle end-to-end on synthetic data (and were what caught a real off-by-window bug during development - see git history / session notes).
