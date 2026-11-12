"""CLI: cheap pre-screen the universe for pump candidates before committing
to a full 5m/15m/1h + OI + funding backfill (see data/screening.py for why).

python scripts/screen_candidates.py --exchange bybit --start 2026-01-01 --end 2026-08-25 --min-move-pct 0.15

Writes a comma-separated symbol list to data/events/candidates_<exchange>.txt
by default - feed it straight into ingest.py:

python scripts/ingest.py --exchange bybit --symbols-file data/events/candidates_bybit.txt --start 2026-01-01 --end 2026-08-25
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Windows consoles often default to a non-UTF8 codepage (e.g. cp1252), which
# raises UnicodeEncodeError on any symbol/error text containing a character
# outside that codepage - replace rather than crash mid-run and lose output.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import typer

from pump_exhaustion_reversal.config import DEFAULT_CONFIG_PATH, load_config
from pump_exhaustion_reversal.connectors import binance, bybit, ccxt_fallback
from pump_exhaustion_reversal.data import screening
from pump_exhaustion_reversal.data import universe as universe_mod
from pump_exhaustion_reversal.data.normalizer import datetime_to_ms, utc_now_ms

app = typer.Typer(add_completion=False)

CONNECTORS = {"binance": binance, "bybit": bybit}


@app.command()
def main(
    exchange: str = typer.Option(..., help="binance | bybit"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(None, help="YYYY-MM-DD, defaults to now"),
    min_move_pct: float = typer.Option(0.15, help="min BTC-adjusted 1/3/7-day move (0.15 = 15%) to qualify as a candidate"),
    top_n: int = typer.Option(0, help="0 = all qualifying candidates; otherwise cap to the top-N by move size"),
    config: str = typer.Option(str(DEFAULT_CONFIG_PATH)),
    output: str = typer.Option(None, help="defaults to data/events/candidates_<exchange>.txt"),
) -> None:
    if exchange not in CONNECTORS:
        raise typer.BadParameter(f"exchange must be one of {list(CONNECTORS)}")
    connector = CONNECTORS[exchange]
    cfg = load_config(config)

    since_ms = datetime_to_ms(start)
    until_ms = datetime_to_ms(end) if end else utc_now_ms()

    ex = connector.make_exchange()
    snapshot = ccxt_fallback.fetch_universe_snapshot(ex, quote_asset=cfg.universe.quote_asset)
    filtered = universe_mod.apply_universe_filter(snapshot, cfg.universe)
    symbol_list = sorted(filtered.loc[filtered["passed_filter"], "symbol"].unique().tolist())
    typer.echo(f"[{exchange}] screening {len(symbol_list)} liquidity-filtered symbols, {start}..{end or 'now'} (daily bars only)")

    btc_1d = connector.fetch_ohlcv_paged("BTCUSDT", "1d", since_ms, until_ms)

    scores = []
    failures = 0
    for i, symbol in enumerate(symbol_list, 1):
        try:
            df_1d = connector.fetch_ohlcv_paged(symbol, "1d", since_ms, until_ms)
            score = screening.compute_screening_score(df_1d, btc_1d)
            scores.append({"symbol": symbol, **score})
        except Exception as exc:  # noqa: BLE001 - one symbol's failure must not abort the screen
            failures += 1
            typer.echo(f"  [FAIL] {symbol}: {exc}")
        if i % 25 == 0 or i == len(symbol_list):
            typer.echo(f"  screened {i}/{len(symbol_list)}")

    candidates = screening.rank_candidates(scores, min_move_pct=min_move_pct, top_n=top_n)

    # Save BEFORE printing: the actual work product must survive even if the
    # console can't display some character in the summary table below.
    output = output or f"data/events/candidates_{exchange}.txt"
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(",".join(candidates["symbol"].tolist()) if not candidates.empty else "", encoding="utf-8")

    typer.echo(f"\n{len(candidates)}/{len(symbol_list)} symbols qualify (move >= {min_move_pct:.0%}), {failures} fetch failures")
    typer.echo(f"Saved candidate list to {out_path}")
    if not candidates.empty:
        display = candidates.copy()
        display["max_move_pct"] = (display["max_move_pct"] * 100).round(1)
        display["max_move_date"] = pd.to_datetime(display["max_move_ts"], unit="ms", utc=True).dt.date
        typer.echo(display[["symbol", "max_move_pct", "max_move_window_days", "max_move_date"]].to_string(index=False))


if __name__ == "__main__":
    app()
