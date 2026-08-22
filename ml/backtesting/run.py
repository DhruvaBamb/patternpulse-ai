"""Backtest CLI: run historical pattern evaluation and store results.

Usage: python -m ml.backtesting.run [--symbols RELIANCE.NS,AAPL --interval 1d]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..datasets.providers.yfinance_provider import build_provider
from ..datasets.validation import validate_ohlcv
from ..inference.pipeline import DetectionPipeline
from .engine import BacktestConfig, BacktestEngine

DEFAULT_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--lookback", type=int, default=1200)
    ap.add_argument("--out", default="artifacts/backtests")
    args = ap.parse_args()

    provider = build_provider()
    pipeline = DetectionPipeline()
    engine = BacktestEngine(pipeline, BacktestConfig())
    all_trades = []
    errors = []
    for sym in args.symbols.split(","):
        sym = sym.strip()
        if not sym:
            continue
        try:
            raw = provider.get_history(sym, args.interval, args.lookback)
            df, report = validate_ohlcv(raw, sym)
            if report.fatal:
                errors.append(f"{sym}: validation failed")
                continue
            trades = engine.run(df, sym, args.interval)
            all_trades.extend(trades)
            print(f"[backtest] {sym}: {len(trades)} trades")
        except Exception as exc:
            errors.append(f"{sym}: {exc}")
            print(f"[backtest] {sym} failed: {exc}")

    result = BacktestEngineResult(all_trades)
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": BacktestConfig().__dict__,
        "overall": result.overall(),
        "pattern_stats": result.summary_by_pattern(),
        "errors": errors,
        "n_trades": len(all_trades),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"backtest_{stamp}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[backtest] {len(all_trades)} trades | hit rate {out['overall']['hit_rate']}")
    print(f"[backtest] saved: {path}")


class BacktestEngineResult:
    """Small wrapper reusing BacktestResult aggregations."""

    def __init__(self, trades):
        from .engine import BacktestResult

        self._r = BacktestResult(trades=trades)

    def overall(self):
        return self._r.overall()

    def summary_by_pattern(self):
        return self._r.summary_by_pattern()


if __name__ == "__main__":
    main()
