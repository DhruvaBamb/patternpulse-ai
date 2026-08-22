"""Deterministic synthetic OHLC fixtures with known pattern content (spec §70)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _df(closes: np.ndarray, seed: int = 42, vol: float = 0.002, exact: list[dict] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    wick = np.maximum(np.abs(rng.normal(0, vol, n)), 0.0005) * closes * 0.5
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": rng.uniform(1e6, 5e6, n)}, index=idx,
    )
    if exact:
        for k, bar in enumerate(exact):
            i = n - len(exact) + k
            df.iloc[i, df.columns.get_loc("open")] = bar["open"]
            df.iloc[i, df.columns.get_loc("high")] = bar["high"]
            df.iloc[i, df.columns.get_loc("low")] = bar["low"]
            df.iloc[i, df.columns.get_loc("close")] = bar["close"]
    return df


def perfect_double_top(n_pre: int = 30, height: float = 20.0, valley: float = 8.0, n_post: int = 10) -> pd.DataFrame:
    closes = np.concatenate([
        np.linspace(100, 120, n_pre),                 # preceding uptrend
        [120, 118, 114, valley + 112 - 112 + 112, 114, 118, 120],  # placeholder replaced below
    ])
    # build explicitly for clarity
    pattern = np.array([120.0, 118.5, 115.0, 112.8, 112.8, 113.5, 116.0, 119.0, 120.0])
    closes = np.concatenate([
        np.linspace(100, 118, n_pre),
        pattern,
        np.linspace(120, 120, 1),
        np.linspace(119, 108, n_post),               # neckline break
    ])
    return _df(closes)


def invalid_double_top() -> pd.DataFrame:
    """Second peak far above the first — violates peak equality."""
    closes = np.concatenate([
        np.linspace(100, 118, 30),
        np.array([120.0, 118.5, 115.0, 112.8, 114.0, 118.0, 126.0]),
        np.linspace(126, 132, 10),                    # continues up
    ])
    return _df(closes)


def perfect_double_bottom() -> pd.DataFrame:
    closes = np.concatenate([
        np.linspace(130, 104, 30),                    # preceding downtrend
        np.array([100.0, 101.5, 104.0, 107.2, 107.2, 106.5, 104.0, 101.0, 100.0]),
        np.linspace(100, 100, 1),
        np.linspace(101, 112, 10),                    # neckline break upward
    ])
    return _df(closes)


def valid_head_and_shoulders() -> pd.DataFrame:
    closes = np.concatenate([
        np.linspace(95, 108, 25),                     # uptrend
        np.array([113.0, 110.0, 106.5]),              # left shoulder region
        np.array([104.0]),                            # trough 1
        np.array([108.0, 112.0, 115.0, 118.0, 120.0]),  # rise to head
        np.array([116.0, 112.0, 108.0, 104.5]),       # fall to trough 2
        np.array([108.0, 111.0, 112.5, 112.8]),       # right shoulder
        np.linspace(112, 103, 8),                     # neckline break
    ])
    return _df(closes)


def invalid_head_and_shoulders() -> pd.DataFrame:
    """Right shoulder far exceeds head."""
    closes = np.concatenate([
        np.linspace(95, 108, 25),
        np.array([113.0, 110.0, 106.5, 104.0, 108.0, 112.0, 115.0, 118.0, 120.0, 116.0, 112.0, 108.0, 104.5]),
        np.array([108.0, 114.0, 120.0, 126.0]),       # right shoulder above head
        np.linspace(126, 134, 8),
    ])
    return _df(closes)


def partial_cup_and_handle(reveal_frac: float = 0.6) -> tuple[pd.DataFrame, float]:
    """Rounded cup; window cut at `reveal_frac` of the formation."""
    n_cup = 80
    xs = np.arange(n_cup + 1)
    bowl = 120 - 22 * np.sin(np.pi * xs / n_cup)
    handle = 120 - 4 * np.sin(np.linspace(0, np.pi, 12))
    full = np.concatenate([np.linspace(102, 120, 25), bowl, handle, np.linspace(120, 130, 10)])
    total_pattern = n_cup + 12
    cut = int(25 + total_pattern * reveal_frac)
    closes = full[:cut]
    return _df(closes), reveal_frac * 100.0


def bullish_engulfing_bars() -> list[dict]:
    return [
        {"open": 110.0, "high": 110.8, "low": 107.2, "close": 108.0},   # bearish
        {"open": 107.5, "high": 111.6, "low": 107.0, "close": 111.2},   # bullish engulfing
    ]


def bearish_engulfing_bars() -> list[dict]:
    return [
        {"open": 108.0, "high": 110.8, "low": 107.2, "close": 110.0},   # bullish
        {"open": 110.5, "high": 111.0, "low": 106.4, "close": 106.8},   # bearish engulfing
    ]


def gartley_path(seed: int = 3) -> pd.DataFrame:
    """X=100, A=115 (XA up), B=105.7 (0.618 XA retrace), C=112.2 (0.70 AB
    retrace), D=103.2 (0.786 XA; CD = 1.38 x BC)."""
    closes = np.concatenate([
        np.linspace(104, 100, 12),          # decline into X (clear swing low)
        np.linspace(100, 115, 14),          # XA
        np.linspace(115, 105.7, 12),        # AB
        np.linspace(105.7, 112.2, 10),      # BC
        np.linspace(112.2, 103.2, 12),      # CD into PRZ
        np.linspace(103.2, 107.0, 6),       # reversal
    ])
    return _df(closes, seed=seed, vol=0.0015)


def failed_gartley_path() -> pd.DataFrame:
    """D overshoots far beyond X — invalid structure."""
    closes = np.concatenate([
        np.linspace(104, 100, 12),
        np.linspace(100, 115, 14),
        np.linspace(115, 105.7, 12),
        np.linspace(105.7, 112.2, 10),
        np.linspace(112.2, 92.0, 12),       # far beyond X
        np.linspace(92.0, 90.0, 6),
    ])
    return _df(closes, seed=3, vol=0.0015)
