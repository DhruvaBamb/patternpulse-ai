"""Seed the market-data cache: fetch and persist candles + quotes for the
configured universe so the first UI load is fast and offline-friendly.

Usage: python -m scripts.seed_market_data [--intervals 1d,1h]
"""

from __future__ import annotations

import argparse

from ml.datasets.providers.base import MarketDataError
from ml.datasets.providers.yfinance_provider import build_provider
from ml.datasets.validation import validate_ohlcv
from backend.app.core.config import scanner_config
from backend.app.db.models import Candle, Symbol
from backend.app.db.session import SessionLocal, init_db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", default=",".join(scanner_config().get("timeframes", ["1d"])))
    ap.add_argument("--lookback", type=int, default=300)
    args = ap.parse_args()

    init_db()
    provider = build_provider()
    universe = scanner_config().get("universe", [])
    db = SessionLocal()
    ok = fail = 0
    try:
        for entry in universe:
            ticker = entry["symbol"]
            db.merge(Symbol(ticker=ticker, name=entry.get("name", ""), market=entry.get("market", "")))
            db.commit()
            for interval in args.intervals.split(","):
                try:
                    raw = provider.get_history(ticker, interval, args.lookback)
                    df, report = validate_ohlcv(raw, ticker)
                    if report.fatal:
                        print(f"[seed] {ticker} {interval}: validation failed — skipped")
                        fail += 1
                        continue
                    existing = set(
                        db.query(Candle.ts)
                        .filter(Candle.symbol_ticker == ticker, Candle.interval == interval)
                        .all()
                    )
                    existing_ts = {t for (t,) in existing}
                    rows = [
                        Candle(symbol_ticker=ticker, interval=interval, ts=ts.to_pydatetime(),
                               open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                               close=float(r["close"]), volume=float(r.get("volume") or 0))
                        for ts, r in df.iterrows() if ts not in existing_ts
                    ]
                    db.add_all(rows)
                    db.commit()
                    ok += 1
                    print(f"[seed] {ticker} {interval}: {len(df)} bars (+{len(rows)} new)")
                except MarketDataError as exc:
                    fail += 1
                    print(f"[seed] {ticker} {interval}: {exc}")
    finally:
        db.close()
    print(f"[seed] done: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
