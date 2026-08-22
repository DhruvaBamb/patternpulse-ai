"""Market data service: fetch, validate, cache candles + quote snapshots."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.datasets.providers.base import MarketDataError, MarketDataProvider
from ml.datasets.providers.yfinance_provider import build_provider
from ml.datasets.validation import validate_ohlcv
from ..core.logging import get_logger
from ..db.models import Candle, Symbol

logger = get_logger("market_data")

CACHE_TTL = 60.0  # seconds


class MarketDataService:
    def __init__(self, provider: MarketDataProvider | None = None) -> None:
        self.provider = provider or build_provider()
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}

    def get_candles(
        self,
        db: Session,
        ticker: str,
        interval: str,
        lookback: int = 250,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Validated OHLCV frame for a symbol/interval (fetch → validate → persist)."""
        key = (ticker, interval)
        now = time.time()
        if use_cache and key in self._cache and now - self._cache[key][0] < CACHE_TTL:
            return self._cache[key][1].tail(lookback).copy()

        df, report = validate_ohlcv(self.provider.get_history(ticker, interval, lookback * 2))
        if report.fatal or df.empty:
            raise MarketDataError(
                f"data validation failed for {ticker} {interval}: "
                + "; ".join(i["detail"] for i in report.issues[:3])
            )
        if report.issues:
            logger.info("ohlcv_validation", symbol=ticker, interval=interval,
                        issues=[i["code"] for i in report.issues])
        self._cache[key] = (now, df)
        self._persist(db, ticker, interval, df)
        return df.tail(lookback).copy()

    def _persist(self, db: Session, ticker: str, interval: str, df: pd.DataFrame) -> None:
        try:
            existing = set(
                db.execute(
                    select(Candle.ts).where(Candle.symbol_ticker == ticker, Candle.interval == interval)
                ).scalars()
            )
            rows = [
                Candle(
                    symbol_ticker=ticker,
                    interval=interval,
                    ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                    close=float(r["close"]), volume=float(r.get("volume", 0) or 0),
                )
                for ts, r in df.iterrows()
                if ts not in existing
            ]
            if not rows:
                return
            try:
                db.add_all(rows)
                db.commit()
            except Exception:
                # concurrent writer inserted first — fall back to per-row best effort
                db.rollback()
                for row in rows:
                    try:
                        db.add(row)
                        db.commit()
                    except Exception:
                        db.rollback()
        except Exception as exc:
            db.rollback()
            logger.warning("candle_persist_failed", symbol=ticker, error=str(exc))

    def upsert_symbol(self, db: Session, ticker: str, name: str = "", market: str = "") -> Symbol:
        sym = db.execute(select(Symbol).where(Symbol.ticker == ticker)).scalar_one_or_none()
        if sym is None:
            sym = Symbol(ticker=ticker, name=name, market=market)
            db.add(sym)
            db.commit()
        return sym

    def refresh_quote(self, db: Session, ticker: str) -> dict:
        try:
            q = self.provider.get_latest(ticker)
        except MarketDataError as exc:
            logger.info("quote_unavailable", symbol=ticker, error=str(exc))
            raise
        sym = db.execute(select(Symbol).where(Symbol.ticker == ticker)).scalar_one_or_none()
        if sym:
            prev = q.get("previous_close") or sym.last_price or q["last_price"]
            sym.last_price = q["last_price"]
            sym.last_change_pct = ((q["last_price"] - prev) / prev * 100) if prev else 0.0
            sym.updated_at = datetime.now(timezone.utc)
            db.commit()
        return q
