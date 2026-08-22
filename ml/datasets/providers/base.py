"""Market data provider abstraction.

The application never imports yfinance directly outside the adapter —
everything talks to :class:`MarketDataProvider` so additional brokers
(Kite, MT5, Polygon, ...) can be dropped in without touching consumers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

TIMEFRAMES: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1wk",
}


@dataclass
class ProviderInfo:
    name: str
    configured: bool
    note: str = ""


class MarketDataError(RuntimeError):
    """Raised when a provider cannot serve the requested data."""


class MarketDataProvider(ABC):
    """Interface every data adapter must implement."""

    name: str = "abstract"

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        interval: str,
        lookback_bars: int = 250,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV candles indexed by tz-aware UTC timestamp.

        Columns: open, high, low, close, volume (lowercase).
        Raises :class:`MarketDataError` on failure.
        """

    def get_latest(self, symbol: str) -> dict:
        """Return the most recent quote snapshot for a symbol."""
        raise MarketDataError(f"{self.name}: get_latest not implemented")

    def stream(self, symbol: str, interval: str):
        """Yield candles as they complete. Optional; default unsupported."""
        raise MarketDataError(f"{self.name}: streaming not implemented")

    def search_symbols(self, query: str) -> list[dict]:
        """Search the provider's symbol universe."""
        raise MarketDataError(f"{self.name}: symbol search not implemented")

    @staticmethod
    def info() -> ProviderInfo:
        return ProviderInfo(name="abstract", configured=False)


class UnconfiguredProvider(MarketDataProvider):
    """Placeholder adapter for providers whose credentials are absent.

    It fails loudly (but with a clear message) instead of pretending to work.
    """

    def __init__(self, name: str, env_keys: list[str]) -> None:
        self.name = name
        self.env_keys = env_keys

    def get_history(self, symbol: str, interval: str, lookback_bars: int = 250, end=None) -> pd.DataFrame:
        raise MarketDataError(
            f"{self.name} is not configured. Set {', '.join(self.env_keys)} in .env to enable it."
        )
