"""Swing pivot extraction and trend primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pivot:
    index: int      # bar position within the analysed window
    price: float
    kind: str       # "high" | "low"

    def __post_init__(self) -> None:
        if self.kind not in ("high", "low"):
            raise ValueError(self.kind)


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Pivot]:
    """Fractal pivots: a bar is a swing high/low when it dominates `left`
    bars before and `right` bars after. Uses highs/lows, tolerant of ties.

    The most recent `right` bars cannot be pivots (confirmation lag) —
    that is inherent to non-repainting swing detection.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    pivots: list[Pivot] = []
    for i in range(left, n - right):
        window_h = high[i - left : i + right + 1]
        if high[i] >= window_h.max() and (window_h == high[i]).sum() <= (left + right + 1):
            # strict dominance vs neighbourhood except ties resolved toward earliest
            if high[i] > high[i - left : i].max() - 1e-12 and high[i] >= high[i + 1 : i + right + 1].max() - 1e-12:
                pivots.append(Pivot(i, float(high[i]), "high"))
        window_l = low[i - left : i + right + 1]
        if low[i] <= window_l.min():
            if low[i] < low[i - left : i].min() + 1e-12 and low[i] <= low[i + 1 : i + right + 1].min() + 1e-12:
                pivots.append(Pivot(i, float(low[i]), "low"))
    # deduplicate same-kind pivots on consecutive equal indices
    return sorted(pivots, key=lambda p: p.index)


def alternate_pivots(pivots: list[Pivot]) -> list[Pivot]:
    """Drop consecutive same-kind pivots keeping the more extreme one."""
    out: list[Pivot] = []
    for p in sorted(pivots, key=lambda q: q.index):
        if out and out[-1].kind == p.kind:
            keep_prev = (
                out[-1].price >= p.price if p.kind == "high" else out[-1].price <= p.price
            )
            if keep_prev:
                continue
            out[-1] = p
        else:
            out.append(p)
    return out


def trend_slope(series: np.ndarray) -> float:
    """OLS slope of a price series, normalized: slope / mean(price)."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=float)
    slope = float(np.polyfit(x, series, 1)[0])
    mean = float(np.mean(series)) or 1.0
    return slope / abs(mean)


def classify_trend(df: pd.DataFrame, lookback: int = 40) -> str:
    """Classify the recent trend: 'up', 'down' or 'sideways'."""
    closes = df["close"].tail(lookback).to_numpy(dtype=float)
    if len(closes) < 5:
        return "sideways"
    s = trend_slope(closes)
    # thresholds in normalized slope per bar
    if s > 0.0008:
        return "up"
    if s < -0.0008:
        return "down"
    return "sideways"


def fit_line(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Least-squares line through (index, price) points -> (slope, intercept, rmse)."""
    if not points:
        return 0.0, 0.0, float("inf")
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if len(points) == 1:
        return 0.0, float(ys[0]), 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    return float(slope), float(intercept), rmse


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range (simple mean of TR over the period)."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - prev_c), abs(l - prev_c)))
    window = tr[-period:] if len(tr) >= period else tr
    return float(np.mean(window)) if len(window) else 0.0
