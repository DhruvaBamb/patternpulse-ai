"""API test configuration: in-memory DB, no scanner, no network.

Env vars are set BEFORE importing any backend module so the SQLAlchemy
engine binds to a single in-memory database for the whole session.
"""

from __future__ import annotations

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from backend.app.main import create_app

    app = create_app()
    with TestClient(app) as tc:  # runs lifespan (init_db, services; scanner skipped in test env)
        _install_fake_market_data(tc.app)
        yield tc


def _install_fake_market_data(app) -> None:
    """Deterministic provider so tests never touch the network."""

    class FakeProvider:
        name = "fake"

        def get_history(self, symbol, interval, lookback_bars=250, end=None):
            n = min(lookback_bars, 250)
            up = 100 + np.arange(n) * 0.4 + np.random.default_rng(7).normal(0, 0.4, n)
            o = np.roll(up, 1)
            o[0] = up[0]
            return pd.DataFrame(
                {"open": o, "high": np.maximum(o, up) + 0.3, "low": np.minimum(o, up) - 0.3,
                 "close": up, "volume": np.full(n, 1e6)},
                index=pd.date_range("2023-01-01", periods=n, tz="UTC"),
            )

        def get_latest(self, symbol):
            df = self.get_history(symbol, "1d", 5)
            return {"symbol": symbol, "last_price": float(df["close"].iloc[-1]),
                    "previous_close": float(df["close"].iloc[-2])}

    app.state.market_data.provider = FakeProvider()
