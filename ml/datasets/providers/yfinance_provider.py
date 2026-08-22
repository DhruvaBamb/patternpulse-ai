"""yfinance adapter — the default free data source for development."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from .base import MarketDataError, MarketDataProvider, ProviderInfo, TIMEFRAMES

logger = logging.getLogger(__name__)

_INTERVAL_TO_YF = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1wk",
}

# yfinance intraday fetch limits (approximate, per request)
_LOOKBACK_LIMITS = {
    "1m": timedelta(days=7),
    "5m": timedelta(days=60),
    "15m": timedelta(days=60),
    "30m": timedelta(days=60),
    "1h": timedelta(days=730),
    "4h": timedelta(days=730),
}


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def get_history(
        self,
        symbol: str,
        interval: str,
        lookback_bars: int = 250,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        if interval not in TIMEFRAMES:
            raise MarketDataError(f"unsupported interval {interval!r}")
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise MarketDataError("yfinance is not installed") from exc

        yf_interval = _INTERVAL_TO_YF[interval]
        limit = _LOOKBACK_LIMITS.get(interval)
        end = end or datetime.now(timezone.utc)
        start = None
        if limit is not None:
            start = end - limit
        elif interval == "1d":
            start = end - timedelta(days=int(lookback_bars * 365 / 252 * 1.6) + 10)
        elif interval == "1wk":
            start = end - timedelta(weeks=int(lookback_bars * 1.6) + 4)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                period="max" if limit is not None and interval in ("1m",) else "auto",
                interval=yf_interval,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
            )
            if df is None or (hasattr(df, "empty") and df.empty):
                # fall back to period-based fetch for daily/weekly
                df = ticker.history(
                    period={
                        "1d": "2y",
                        "1wk": "5y",
                        "4h": "max",
                        "1h": "max",
                    }.get(interval, "1mo"),
                    interval=yf_interval,
                    auto_adjust=False,
                    actions=False,
                )
        except Exception as exc:  # yfinance raises assorted errors
            raise MarketDataError(f"yfinance fetch failed for {symbol} {interval}: {exc}") from exc

        if df is None or df.empty:
            raise MarketDataError(f"no data returned for {symbol!r} at {interval}")

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        result = df.tail(lookback_bars)
        if len(result) < 2:
            raise MarketDataError(f"insufficient candles for {symbol!r} at {interval}")
        return result

    def get_latest(self, symbol: str) -> dict:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise MarketDataError("yfinance is not installed") from exc
        try:
            fast = yf.Ticker(symbol).fast_info
            return {
                "symbol": symbol,
                "last_price": float(fast["last_price"]),
                "previous_close": float(fast["previous_close"]),
            }
        except Exception as exc:
            raise MarketDataError(f"quote unavailable for {symbol}: {exc}") from exc

    def search_symbols(self, query: str) -> list[dict]:
        # yfinance has no reliable search API; treat configured universe as the
        # searchable set (the backend layers the app universe on top of this).
        return []

    @staticmethod
    def info() -> ProviderInfo:
        try:
            import yfinance  # noqa: F401

            return ProviderInfo(name="yfinance", configured=True)
        except ImportError:
            return ProviderInfo(name="yfinance", configured=False, note="pip install yfinance")


def build_provider() -> MarketDataProvider:
    """Factory used by the backend; extend here as more providers land."""
    return YFinanceProvider()
