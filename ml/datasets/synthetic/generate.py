"""Synthetic dataset generator.

Procedurally constructs OHLCV windows containing registry patterns at known
locations with controlled variation (scale, duration, noise, slope, gaps,
volume, preceding trend, partial completion). Each sample carries:
class label, bounding box, completion %, direction, keypoints.

Self-verification: after generation, the real geometry validator runs on the
sample; only samples the engine can detect are kept as positives. This makes
label noise measurable instead of assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...registry.registry import get_registry
from ...geometry.engine import GeometryEngine
from ..rendering.renderer import CandlestickRenderer, RenderConfig


# ── path synthesis primitives ────────────────────────────────────────────────

def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _candles_from_close(closes: np.ndarray, rng: np.random.Generator, volatility: float) -> pd.DataFrame:
    """Build OHLCV bars around a close path. No future leakage: each bar's
    high/low derives only from that bar's own open/close plus wick noise."""
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0] * (1 + rng.normal(0, volatility * 0.3))
    opens[1:] = closes[:-1] * (1 + rng.normal(0, volatility * 0.25, n - 1))
    wick = np.maximum(np.abs(rng.normal(0, volatility, n)), 0.0005)
    body = closes - opens
    highs = np.maximum(opens, closes) + wick * np.abs(rng.normal(1, 0.3, n))
    lows = np.minimum(opens, closes) - wick * np.abs(rng.normal(1, 0.3, n))
    base_vol = rng.lognormal(10, 0.4, n) * (1 + np.abs(body) / max(np.mean(np.abs(body)), 1e-9) * 0.3)
    idx = pd.date_range("2010-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": base_vol},
        index=idx,
    )


def _trend(n: int, start: float, drift: float, rng: np.random.Generator, volatility: float) -> np.ndarray:
    steps = rng.normal(drift, volatility, n)
    return start * np.cumprod(1 + steps)


def _traverse(points: list[tuple[int, float]], total: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Piecewise-linear path through (bar_offset, price) waypoints with noise."""
    path = np.empty(total)
    xs = np.arange(total, dtype=float)
    for (x0, p0), (x1, p1) in zip(points, points[1:]):
        if x1 == x0:
            continue
        seg = p0 + (p1 - p0) * (xs[x0 : x1 + 1] - x0) / (x1 - x0)
        path[x0 : x1 + 1] = seg
    path += rng.normal(0, noise * np.abs(np.mean(path)) * 0.05, total)
    return path


@dataclass
class SynthPattern:
    pattern_id: str
    closes: np.ndarray                       # full window close path
    span: tuple[int, int]                    # pattern bar range in window
    direction: str
    keypoints_price: list[tuple[int, float]]  # waypoints (for labels/meta)
    completion: float                        # 100 for complete generation (truncation applied later)
    volatility: float = 0.012               # noise level used when deriving candles


Generator = Callable[[dict, np.random.Generator], SynthPattern]


# ── classical path generators ────────────────────────────────────────────────

def gen_head_and_shoulders(v: dict, rng: np.random.Generator) -> SynthPattern:
    pre = int(v["pre_bars"])
    n_pat = int(rng.integers(*v["pattern_bars"]))
    post = int(rng.integers(3, 15))
    base = v["base_price"]
    inverse = v["inverse"]
    depth = base * rng.uniform(*v["hs_depth"])          # head prominence
    shoulder = base * rng.uniform(0.2, 0.9) * depth
    neck = base * rng.uniform(-0.05, 0.05)
    s = 1.0 if not inverse else -1.0
    pts = [
        (0, base + s * shoulder * 0.4),
        (n_pat // 5, base + s * shoulder),        # left shoulder
        (2 * n_pat // 5, neck + s * shoulder * rng.uniform(0.3, 0.6)),  # trough
        (n_pat // 2, base + s * depth),           # head
        (3 * n_pat // 5, neck - s * shoulder * rng.uniform(0.0, 0.2)),
        (4 * n_pat // 5, base + s * shoulder * rng.uniform(0.85, 1.1)),  # right shoulder
        (n_pat, neck),
        (n_pat + post, neck - s * depth * rng.uniform(0.4, 0.9)),        # measured move hint
    ]
    pre_drift = rng.uniform(0.0008, 0.003) * (1 if not inverse else 0.5)
    pre_path = _trend(pre, base - pre * pre_drift, pre_drift, rng, v["volatility"])
    pattern_path = _traverse(pts, n_pat + post + 1, rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern_path])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes,
        span=(pre, pre + n_pat), direction="bearish" if not inverse else "bullish",
        keypoints_price=[(pre + x, p) for x, p in pts], completion=100.0,
    )


def gen_multiples(v: dict, rng: np.random.Generator) -> SynthPattern:
    pre = int(v["pre_bars"])
    n = int(v["n_peaks"])
    n_pat = int(rng.integers(*v["pattern_bars"]))
    post = int(rng.integers(3, 12))
    base = v["base_price"]
    top = v["kind"] == "top"
    s = 1.0 if top else -1.0
    height = base * rng.uniform(*v["m_height"])
    valley = base * rng.uniform(*v["m_depth"])
    eq_tol = rng.uniform(0.0, v.get("peak_noise", 0.02))
    pts = [(0, base)]
    seg = n_pat // (2 * n - 1)
    for k in range(n):
        pts.append((seg * (2 * k + 1), base + s * height * (1 + rng.uniform(-eq_tol, eq_tol))))
        if k < n - 1:
            pts.append((seg * (2 * k + 2), base + s * valley))
    pts.append((n_pat, base + s * valley))
    pts.append((n_pat + post, base + s * (valley - height * rng.uniform(0.5, 1.0))))
    pre_drift = rng.uniform(0.001, 0.004) * (1 if top else -1)
    pre_path = _trend(pre, base - pre * pre_drift * (1 if top else 1), pre_drift, rng, v["volatility"])
    pattern_path = _traverse(pts, n_pat + post + 1, rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern_path])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes, span=(pre, pre + n_pat),
        direction="bearish" if top else "bullish",
        keypoints_price=[(pre + x, p) for x, p in pts], completion=100.0,
    )


def gen_cup(v: dict, rng: np.random.Generator) -> SynthPattern:
    pre = int(v["pre_bars"])
    n_cup = int(rng.integers(*v["pattern_bars"]))
    inverse = v["inverse"]
    with_handle = v["variant"] == "cup_and_handle"
    n_handle = int(n_cup * rng.uniform(0.10, 0.22)) if with_handle else 0
    post = int(rng.integers(4, 14))
    base = v["base_price"]
    s = -1.0 if not inverse else 1.0     # cup dips down
    depth = base * rng.uniform(*v["cup_depth"])
    rim_noise = rng.uniform(0.0, 0.02)
    xs = np.arange(n_cup + 1, dtype=float)
    # parabolic bowl: rim -> basin -> rim
    bowl = base + s * depth * np.sin(np.pi * xs / n_cup) ** 1.0
    bowl[0] = base * (1 + rim_noise * rng.uniform(-1, 1))
    bowl[-1] = base * (1 + rim_noise * rng.uniform(-1, 1))
    handle = base + s * depth * rng.uniform(0.08, 0.3) * np.sin(np.linspace(0, np.pi, n_handle + 1)) if n_handle else np.array([])
    breakout = np.linspace(base, base - s * depth * rng.uniform(0.5, 1.0), post + 1) * (
        1 + rng.normal(0, v["volatility"] * 0.2, post + 1)
    )
    pattern = np.concatenate([bowl, handle, breakout])
    pre_drift = rng.uniform(0.0008, 0.0025)
    pre_path = _trend(pre, base - n_cup * pre_drift, pre_drift, rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern])
    end_span = pre + n_cup + n_handle if with_handle else pre + n_cup
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes, span=(pre, end_span),
        direction="bullish" if not inverse else "bearish",
        keypoints_price=[(pre, base), (pre + n_cup // 2, base + s * depth), (pre + n_cup, base)],
        completion=100.0,
    )


def gen_trendline(v: dict, rng: np.random.Generator) -> SynthPattern:
    """Triangles & wedges: oscillation between two boundary lines."""
    pre = int(v["pre_bars"])
    n_pat = int(rng.integers(*v["pattern_bars"]))
    post = int(rng.integers(3, 10))
    base = v["base_price"]
    w0 = base * rng.uniform(*v["width"])          # initial range width
    upper_slope = v["upper_slope"] * w0 / n_pat   # slopes as fractions of width
    lower_slope = v["lower_slope"] * w0 / n_pat
    cycles = int(rng.integers(3, 7))
    xs = np.arange(n_pat + 1, dtype=float)
    upper = base + w0 / 2 + upper_slope * xs
    lower = base - w0 / 2 + lower_slope * xs
    phase = rng.uniform(0, np.pi * 2)
    osc = np.sin(np.linspace(0, np.pi * 2 * cycles, n_pat + 1) + phase)
    path = (upper + lower) / 2 + osc * (upper - lower) / 2 * rng.uniform(0.75, 0.95)
    path += rng.normal(0, v["volatility"] * base * 0.03, n_pat + 1)
    break_dir = v.get("break_dir", rng.choice([-1, 1]))
    brk = np.linspace(path[-1], path[-1] + break_dir * w0 * rng.uniform(0.6, 1.1), post + 1)
    pattern = np.concatenate([path, brk])
    pre_drift = upper_slope / max(upper_slope, 1e-9) * rng.uniform(0.001, 0.003)
    pre_path = _trend(pre, base - pre * pre_drift, pre_drift, rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes, span=(pre, pre + n_pat),
        direction=v.get("direction", "neutral"),
        keypoints_price=[(pre, upper[0]), (pre + n_pat // 2, (upper[n_pat // 2] + lower[n_pat // 2]) / 2),
                         (pre + n_pat, (upper[-1] + lower[-1]) / 2)],
        completion=100.0,
    )


def gen_flag(v: dict, rng: np.random.Generator) -> SynthPattern:
    pre = int(rng.integers(5, 20))
    n_pole = int(rng.integers(*v["pole_bars"]))
    n_flag = int(rng.integers(*v["flag_bars"]))
    post = int(rng.integers(3, 9))
    base = v["base_price"]
    bullish = v["direction"] == "bullish"
    s = 1.0 if bullish else -1.0
    pole_h = base * rng.uniform(*v["pole_height"])
    pole = base + s * pole_h * (np.arange(n_pole + 1) / n_pole) ** 1.1
    pole = pole * (1 + rng.normal(0, v["volatility"] * 0.5, n_pole + 1))
    flag_top = base + s * pole_h
    flag_depth = pole_h * rng.uniform(0.1, 0.4)
    xs = np.arange(n_flag + 1, dtype=float)
    if v["pennant"]:
        mid = flag_top - s * flag_depth / 2
        fpath = mid + (xs / n_flag - 0.5) * s * flag_depth + np.sin(np.linspace(0, 3 * np.pi, n_flag + 1)) * flag_depth * 0.18
    else:
        # slight counter-slope channel
        fpath = flag_top - s * flag_depth * (xs / n_flag) + np.sin(np.linspace(0, 2.5 * np.pi, n_flag + 1)) * flag_depth * 0.12
    breakout = np.linspace(fpath[-1], fpath[-1] + s * pole_h * rng.uniform(0.7, 1.2), post + 1)
    pattern = np.concatenate([pole, fpath, breakout])
    pre_path = _trend(pre, base * (1 - rng.uniform(0.0, 0.05)), 0.0, rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes,
        span=(pre + n_pole, pre + n_pole + n_flag),
        direction=v["direction"],
        keypoints_price=[(pre, base), (pre + n_pole, float(flag_top)), (pre + n_pole + n_flag, float(fpath[-1]))],
        completion=100.0,
    )


def gen_rectangle(v: dict, rng: np.random.Generator) -> SynthPattern:
    pre = int(v["pre_bars"])
    n_pat = int(rng.integers(*v["pattern_bars"]))
    post = int(rng.integers(3, 9))
    base = v["base_price"]
    height = base * rng.uniform(*v["height"])
    cycles = int(rng.integers(3, 8))
    xs = np.arange(n_pat + 1, dtype=float)
    mid = base + height / 2 * v.get("offset", 0.5)
    path = mid + np.sin(np.linspace(0, np.pi * 2 * cycles, n_pat + 1) + rng.uniform(0, np.pi)) * height / 2 * rng.uniform(0.8, 0.98)
    path += rng.normal(0, v["volatility"] * base * 0.03, n_pat + 1)
    d = 1 if v.get("bias", "neutral") == "bullish" else (-1 if v.get("bias") == "bearish" else rng.choice([-1, 1]))
    brk = np.linspace(path[-1], path[-1] + d * height * rng.uniform(0.8, 1.4), post + 1)
    pattern = np.concatenate([path, brk])
    pre_path = _trend(pre, base, d * rng.uniform(0.0, 0.001), rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes, span=(pre, pre + n_pat),
        direction=v.get("direction", "neutral"),
        keypoints_price=[(pre, base), (pre + n_pat // 2, mid), (pre + n_pat, base)],
        completion=100.0,
    )


def gen_channel(v: dict, rng: np.random.Generator) -> SynthPattern:
    cfg = {
        **v,
        "upper_slope": v["slope_sign"] * rng.uniform(0.6, 1.0),
        "lower_slope": v["slope_sign"] * rng.uniform(0.6, 1.0),
        "width": (0.08, 0.25),
        "direction": "bullish" if v["slope_sign"] > 0 else "bearish",
    }
    return gen_trendline(cfg, rng)


def gen_broadening(v: dict, rng: np.random.Generator) -> SynthPattern:
    cfg = {
        **v,
        "upper_slope": rng.uniform(0.3, 0.9),
        "lower_slope": -rng.uniform(0.3, 0.9),
        "width": (0.05, 0.15),
    }
    return gen_trendline(cfg, rng)


def gen_diamond(v: dict, rng: np.random.Generator) -> SynthPattern:
    pre = int(v["pre_bars"])
    n_pat = int(rng.integers(*v["pattern_bars"]))
    half = n_pat // 2
    post = int(rng.integers(3, 8))
    base = v["base_price"]
    height = base * rng.uniform(0.1, 0.35)
    xs1 = np.arange(half + 1, dtype=float)
    xs2 = np.arange(n_pat - half + 1, dtype=float)
    expand = np.sin(np.linspace(0, np.pi * 2 * 2, half + 1)) * (height / 2) * (xs1 / half + 0.2)
    contract = np.sin(np.linspace(0, np.pi * 2 * 2, n_pat - half + 1)) * (height / 2) * (1 - xs2 / (n_pat - half) + 0.2)
    path = np.concatenate([base + expand, base + contract[1:]])
    d = -1 if v["position"] == "top" else 1
    brk = np.linspace(path[-1], path[-1] + d * height * rng.uniform(0.5, 1.1), post + 1)
    pattern = np.concatenate([path, brk])
    drift = (0.001 if v["position"] == "top" else -0.001)
    pre_path = _trend(pre, base - pre * drift, drift, rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes, span=(pre, pre + n_pat),
        direction="bearish" if v["position"] == "top" else "bullish",
        keypoints_price=[(pre, base), (pre + half, base), (pre + n_pat, base)], completion=100.0,
    )


# ── candlestick generators ──────────────────────────────────────────────────

def gen_candlestick(v: dict, rng: np.random.Generator) -> SynthPattern:
    """Construct bars whose final k bars satisfy the registry rule params."""
    pre = int(rng.integers(30, 90))
    k = v["k_bars"]
    n_total = pre + k
    volatility = v["volatility"]
    df_placeholder = _candles_from_close(_trend(pre, v["base_price"], 0.0, rng, volatility), rng, volatility)
    bars: list[dict] = []
    base = float(df_placeholder["close"].iloc[-1])
    kind = v["ckind"]

    def bar(o, c, up, lo):
        return {"open": o, "close": c, "high": max(o, c) + up, "low": min(o, c) - lo, "volume": float(abs(rng.lognormal(10, 0.3)))}

    rng01 = abs(rng.normal(1, 0.1)) * 0.01
    if kind == "doji":
        rng_span = base * rng.uniform(0.004, 0.02)
        o = base
        c = o + rng.uniform(-0.04, 0.04) * rng_span
        up = rng_span * v.get("upper_frac", 0.4)
        lo = rng_span * v.get("lower_frac", 0.4)
        bars.append(bar(o, c, up, lo))
    elif kind == "hammer":
        body = base * rng.uniform(0.004, 0.012)
        direction = v.get("hb_dir", 1)
        o = base
        c = o + direction * body
        if v.get("inverted"):
            bars.append(bar(o, c, body * rng.uniform(2.2, 4.5), body * rng.uniform(0.05, 0.2)))
        else:
            bars.append(bar(o, c, body * rng.uniform(0.05, 0.2), body * rng.uniform(2.2, 4.5)))
    elif kind == "engulfing":
        body1 = base * rng.uniform(0.006, 0.018)
        o1 = base
        c1 = o1 - body1 if v.get("bull") else o1 + body1
        body2 = body1 * rng.uniform(1.3, 2.5)
        o2 = c1
        c2 = o1 + body2 * rng.uniform(0.6, 1.0) if v.get("bull") else o1 - body2 * rng.uniform(0.6, 1.0)
        bars.append(bar(o1, c1, body1 * 0.2, body1 * 0.2))
        bars.append(bar(o2, c2, body2 * 0.15, body2 * 0.15))
    elif kind == "star":
        body1 = base * rng.uniform(0.012, 0.03)
        o1 = base
        c1 = o1 + body1 if not v.get("bull") else o1 - body1
        # star gaps beyond body1
        star_mid = c1 + (body1 * 0.3 if not v.get("bull") else -body1 * 0.3)
        sb = base * rng.uniform(0.001, 0.004)
        o2 = star_mid
        c2 = o2 + rng.uniform(-1, 1) * sb
        w2 = sb * 2.5
        o3 = star_mid
        c3 = (min(o1, c1) + body1 * rng.uniform(0.55, 0.9)) if v.get("bull") else (max(o1, c1) - body1 * rng.uniform(0.55, 0.9))
        body3 = abs(c3 - o3)
        bars.append(bar(o1, c1, body1 * 0.1, body1 * 0.1))
        bars.append(bar(o2, c2, w2, w2))
        bars.append(bar(o3, c3, body3 * 0.2, body3 * 0.2))
    else:  # generic: strong single candle
        body = base * rng.uniform(0.008, 0.02)
        o = base
        c = o + (body if v.get("bull", True) else -body)
        bars.append(bar(o, c, body * 0.05, body * 0.05))

    closes = np.concatenate([
        df_placeholder["close"].to_numpy(),
        np.array([b["close"] for b in bars]),
    ])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes, span=(pre, n_total - 1),
        direction=v.get("direction", "neutral"),
        keypoints_price=[(pre, base)], completion=100.0,
    )


def _candle_bars_from_syn(sp: SynthPattern, v: dict, rng: np.random.Generator) -> pd.DataFrame:
    """For candlestick patterns we must control exact OHLC of final bars:
    rebuild the frame using the stored closes then override last bars."""
    df = _candles_from_close(sp.closes, rng, v["volatility"])
    return df


# ── harmonic generators ─────────────────────────────────────────────────────

def gen_harmonic(v: dict, rng: np.random.Generator) -> SynthPattern:
    pre = int(rng.integers(15, 40))
    n_xa = int(rng.integers(10, 30))
    var = v["hvariant"]
    x = v["base_price"]
    up = rng.random() < 0.5
    s = 1.0 if up else -1.0
    xa = x * rng.uniform(0.1, 0.35)
    a = x + s * xa
    r = v["ratios"]
    b = a - s * xa * rng.uniform(*r["xab"])
    # C retraces AB (classic 0.382–0.886 window) — the registry's [1.13,1.618]
    # window governs the later CD extension, which D placement below approximates.
    bc = abs(a - b) * rng.uniform(0.382, 0.886)
    c = b + s * bc
    d = a - s * xa * rng.uniform(*r["ad"])
    legs = [(x, a), (a, b), (b, c), (c, d)]
    post = int(rng.integers(4, 12))
    total = pre + sum(rng.integers(8, 18) for _ in legs) + post
    pts = [(0, x)]
    xi = 0
    for (p0, p1) in legs:
        xi += int(rng.integers(8, 18))
        pts.append((xi, p1))
    n_pat = xi
    # reversal after D
    pts.append((n_pat + post, d - s * abs(xa) * 0.4))
    pattern = _traverse(pts, n_pat + post + 1, rng, v["volatility"] * 0.5)
    pre_path = _trend(pre, x, 0.0, rng, v["volatility"])
    closes = np.concatenate([pre_path, pattern])
    return SynthPattern(
        pattern_id=v["pattern_id"], closes=closes, span=(pre, pre + n_pat),
        direction="neutral",
        keypoints_price=[(pre, x), (pre + pts[1][0], a), (pre + pts[2][0], b), (pre + pts[3][0], c), (pre + pts[4][0], d)],
        completion=100.0,
    )


# ── negative generators ─────────────────────────────────────────────────────

def gen_negative(v: dict, rng: np.random.Generator) -> SynthPattern:
    n = int(rng.integers(60, 160))
    drift = rng.uniform(-0.0015, 0.0015)
    closes = _trend(n, v["base_price"], drift, rng, v["volatility"])
    return SynthPattern(pattern_id="__negative__", closes=closes, span=(0, 0), direction="neutral",
                        keypoints_price=[], completion=0.0)


def gen_hard_negative(v: dict, rng: np.random.Generator) -> SynthPattern:
    """Pattern-like but structurally invalid (e.g. unequal double top)."""
    sp = gen_multiples({**v, "pattern_id": "__hard_negative__", "peak_noise": 0.09,
                        "m_depth": (0.005, 0.02), "kind": "top", "n_peaks": 2}, rng)
    sp.pattern_id = "__hard_negative__"
    return sp


# ── per-pattern generation configs ─────────────────────────────────────────

def build_generator_map(registry) -> dict[str, tuple[Generator, dict]]:
    """Map pattern id -> (generator, base variation params)."""
    m: dict[str, tuple[Generator, dict]] = {}
    common = {"volatility": 0.012, "base_price": 100.0}
    for d in registry.implemented():
        pid = d.id
        if pid.startswith(("inverse_", )):
            continue
        if pid == "head_and_shoulders":
            m[pid] = (gen_head_and_shoulders, {**common, "pattern_id": pid, "inverse": False,
                                               "pattern_bars": (40, 120), "hs_depth": (0.08, 0.3), "pre_bars": 30})
            m["inverse_head_and_shoulders"] = (gen_head_and_shoulders, {
                **common, "pattern_id": "inverse_head_and_shoulders", "inverse": True,
                "pattern_bars": (40, 120), "hs_depth": (0.08, 0.3), "pre_bars": 30})
        elif pid in ("double_top", "double_bottom"):
            m[pid] = (gen_multiples, {**common, "pattern_id": pid, "kind": "top" if pid == "double_top" else "bottom",
                                      "n_peaks": 2, "pattern_bars": (30, 90), "m_height": (0.06, 0.25),
                                      "m_depth": (0.02, 0.10), "pre_bars": 25, "peak_noise": 0.015})
        elif pid in ("triple_top", "triple_bottom"):
            m[pid] = (gen_multiples, {**common, "pattern_id": pid, "kind": "top" if pid == "triple_top" else "bottom",
                                      "n_peaks": 3, "pattern_bars": (50, 140), "m_height": (0.06, 0.2),
                                      "m_depth": (0.02, 0.08), "pre_bars": 25, "peak_noise": 0.015})
        elif pid == "cup_and_handle":
            m[pid] = (gen_cup, {**common, "pattern_id": pid, "variant": "cup_and_handle", "inverse": False,
                                "pattern_bars": (50, 140), "cup_depth": (0.08, 0.3), "pre_bars": 25})
            m["inverse_cup_and_handle"] = (gen_cup, {**common, "pattern_id": "inverse_cup_and_handle",
                                                     "variant": "cup_and_handle", "inverse": True,
                                                     "pattern_bars": (50, 140), "cup_depth": (0.08, 0.3), "pre_bars": 25})
            m["rounded_bottom"] = (gen_cup, {**common, "pattern_id": "rounded_bottom", "variant": "rounded",
                                             "inverse": False, "pattern_bars": (40, 120), "cup_depth": (0.06, 0.25), "pre_bars": 25})
            m["rounded_top"] = (gen_cup, {**common, "pattern_id": "rounded_top", "variant": "rounded",
                                          "inverse": True, "pattern_bars": (40, 120), "cup_depth": (0.06, 0.25), "pre_bars": 25})
        elif pid in ("ascending_triangle", "descending_triangle", "symmetrical_triangle",
                     "rising_wedge", "falling_wedge"):
            slopes = {
                "ascending_triangle": (0.0, 1.0),
                "descending_triangle": (-1.0, 0.0),
                "symmetrical_triangle": (-1.0, 1.0),
                "rising_wedge": (0.8, 0.4),
                "falling_wedge": (-0.4, -0.8),
            }[pid]
            m[pid] = (gen_trendline, {**common, "pattern_id": pid, "upper_slope": slopes[0],
                                      "lower_slope": slopes[1], "width": (0.08, 0.25), "pattern_bars": (30, 100),
                                      "pre_bars": 25})
        elif pid in ("bull_flag", "bear_flag", "bull_pennant", "bear_pennant"):
            bull = pid.startswith("bull")
            m[pid] = (gen_flag, {**common, "pattern_id": pid, "direction": "bullish" if bull else "bearish",
                                 "pennant": "pennant" in pid, "pole_bars": (5, 18), "flag_bars": (8, 25),
                                 "pole_height": (0.08, 0.3)})
        elif pid in ("rectangle", "bullish_rectangle", "bearish_rectangle", "horizontal_channel"):
            bias = {"rectangle": "neutral", "horizontal_channel": "neutral",
                    "bullish_rectangle": "bullish", "bearish_rectangle": "bearish"}[pid]
            m[pid] = (gen_rectangle, {**common, "pattern_id": pid, "height": (0.06, 0.2),
                                      "pattern_bars": (30, 100), "pre_bars": 20, "bias": bias,
                                      "direction": bias if bias != "neutral" else "neutral"})
        elif pid in ("ascending_channel", "descending_channel"):
            m[pid] = (gen_channel, {**common, "pattern_id": pid, "slope_sign": 1 if "ascending" in pid else -1,
                                    "pattern_bars": (40, 140), "pre_bars": 20})
        elif pid == "broadening_formation":
            m[pid] = (gen_broadening, {**common, "pattern_id": pid, "pattern_bars": (30, 100), "pre_bars": 20})
        elif pid in ("diamond_top", "diamond_bottom"):
            m[pid] = (gen_diamond, {**common, "pattern_id": pid, "position": "top" if pid == "diamond_top" else "bottom",
                                    "pattern_bars": (40, 120), "pre_bars": 25})
        elif d.category == "candlestick":
            k = d.max_bars
            ckind = {
                "doji": "doji", "long_legged_doji": "doji", "dragonfly_doji": "doji",
                "gravestone_doji": "doji", "four_price_doji": "doji",
                "hammer": "hammer", "inverted_hammer": "hammer", "hanging_man": "hammer", "shooting_star": "hammer",
                "bullish_engulfing": "engulfing", "bearish_engulfing": "engulfing",
                "morning_star": "star", "evening_star": "star",
            }.get(d.geometry_validator if d.geometry_validator in ("doji", "hammer", "engulfing", "star") else None, "generic")
            fracs = {
                "long_legged_doji": {"upper_frac": 0.45, "lower_frac": 0.45},
                "dragonfly_doji": {"upper_frac": 0.02, "lower_frac": 0.85},
                "gravestone_doji": {"upper_frac": 0.85, "lower_frac": 0.02},
                "four_price_doji": {"upper_frac": 0.02, "lower_frac": 0.02},
            }.get(pid, {})
            hb = {"inverted": "inverted" in pid} if pid in ("inverted_hammer", "shooting_star", "hammer", "hanging_man") else {}
            m[pid] = (gen_candlestick, {**common, "pattern_id": pid, "k_bars": k, "ckind": ckind,
                                        "bull": d.direction == "bullish", **fracs, **hb})
        elif d.category == "harmonic":
            ratios = {
                "gartley": {"xab": (0.618, 0.65), "bc": (1.13, 1.618), "ad": (0.786, 0.82)},
                "bat": {"xab": (0.382, 0.50), "bc": (1.618, 2.618), "ad": (0.886, 0.90)},
                "butterfly": {"xab": (0.786, 0.82), "bc": (1.618, 2.24), "ad": (1.27, 1.618)},
                "crab": {"xab": (0.382, 0.618), "bc": (2.4, 3.618), "ad": (1.55, 1.65)},
                "deep_crab": {"xab": (0.86, 0.90), "bc": (2.0, 3.618), "ad": (1.58, 1.63)},
                "shark": {"xab": (1.13, 1.618), "bc": (1.13, 1.618), "ad": (0.886, 1.13)},
                "cypher": {"xab": (0.382, 0.618), "bc": (1.272, 1.414), "ad": (0.786, 0.786)},
                "ab_cd": {"xab": (0.618, 0.786), "bc": (1.0, 1.0), "ad": (0.618, 0.786)},
            }[pid]
            m[pid] = (gen_harmonic, {**common, "pattern_id": pid, "hvariant": pid, "ratios": ratios})
    return m


# ── dataset assembly ────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    samples_per_pattern: int = 60
    partial_frac: float = 0.35     # fraction of positives rendered partially formed
    negative_frac: float = 0.35    # negatives relative to total positives
    hard_negative_frac: float = 0.15
    verify: bool = True
    seed: int = 7
    out_dir: Path = Path("artifacts/datasets")
    version: str | None = None


def _truncate_for_completion(sp: SynthPattern, frac: float) -> tuple[np.ndarray, float]:
    """Cut the window so only `frac` of the pattern span has formed."""
    i0, i1 = sp.span
    cut = int(i0 + (i1 - i0) * frac)
    cut = max(cut, i0 + 5)
    return sp.closes[: cut + 1], frac * 100.0


def generate_dataset(spec: DatasetSpec) -> dict:
    registry = get_registry()
    gen_map = build_generator_map(registry)
    engine = GeometryEngine(registry)
    renderer = CandlestickRenderer(RenderConfig())
    rng = _seeded_rng(spec.seed)

    version = spec.version or f"synth_{registry.version[:8]}_{spec.seed}"
    root = spec.out_dir / version
    for split in ("train", "valid", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": version,
        "seed": spec.seed,
        "pattern_registry_version": registry.version,
        "render_config": RenderConfig().__dict__,
        "spec": {"samples_per_pattern": spec.samples_per_pattern,
                 "partial_frac": spec.partial_frac,
                 "negative_frac": spec.negative_frac},
        "samples": [],
        "splits": {},
    }

    stats = {"generated": 0, "kept": 0, "failed_verification": 0, "negatives": 0}
    spec_common_vol = 0.012

    def process_sample(sp: SynthPattern, completion: float, kind: str, sample_id: str):
        # assign chronological split by series hash (independent series => no leakage)
        h = int(hashlib.sha1(sample_id.encode()).hexdigest(), 16) % 100
        split = "train" if h < 70 else ("valid" if h < 85 else "test")
        df = _candles_from_close(sp.closes, rng, min(sp.volatility, 0.012))
        if kind == "positive":
            # window must end at/after the pattern (no future bars shown past break)
            if completion >= 99.0:
                df = df.iloc[: sp.span[1] + 1]
        lines = []
        if kind != "negative":
            p_low = float(df["low"].iloc[sp.span[0] : sp.span[1] + 1].min()) if sp.span[1] < len(df) else float(df["low"].min())
            p_high = float(df["high"].iloc[sp.span[0] : sp.span[1] + 1].max()) if sp.span[1] < len(df) else float(df["high"].max())
            bbox = renderer.bbox_px(df, sp.span[0], sp.span[1], p_low, p_high)
            x0, y0, x1, y1 = bbox
            W, H = renderer.cfg.image_width, renderer.cfg.image_height
            nx0, ny0 = x0 / W, y0 / H
            nw, nh = (x1 - x0) / W, (y1 - y0) / H
            cls = registry.get(sp.pattern_id).cv_class_id
            if cls is None:
                return None
            lines.append(f"{cls} {nx0 + nw / 2:.6f} {ny0 + nh / 2:.6f} {nw:.6f} {nh:.6f}")
        rendered = renderer.render(df)
        img_path = root / "images" / split / f"{sample_id}.png"
        rendered.save(img_path)
        (root / "labels" / split / f"{sample_id}.txt").write_text("\n".join(lines))
        entry = {
            "id": sample_id, "split": split, "kind": kind, "pattern_id": sp.pattern_id,
            "completion": round(completion, 1), "direction": sp.direction,
            "span": list(sp.span), "keypoints": sp.keypoints_price,
        }
        manifest["samples"].append(entry)
        return entry

    for pid, (gen, base_v) in sorted(gen_map.items()):
        defn = registry.get(pid)
        for k in range(spec.samples_per_pattern):
            v = {**base_v, "volatility": float(rng.uniform(0.004, 0.018)),
                 "base_price": float(rng.uniform(15, 800))}
            sp = gen(v, rng)
            sp.volatility = v["volatility"]
            completion = 100.0
            if defn.supports_partial_detection and rng.random() < spec.partial_frac:
                frac = float(rng.uniform(0.35, 0.95))
                closes, completion = _truncate_for_completion(sp, frac)
                sp = SynthPattern(sp.pattern_id, closes, (sp.span[0], min(sp.span[1], len(closes) - 1)),
                                  sp.direction, sp.keypoints_price, completion)
            stats["generated"] += 1
            if spec.verify and completion >= 99.0:
                check_df = _candles_from_close(sp.closes, rng, min(sp.volatility, 0.012))
                res = engine.validate(check_df.iloc[: sp.span[1] + 1], pattern_ids=[pid])
                if not (res and res[0].matched):
                    stats["failed_verification"] += 1
                    continue
            sid = f"{pid}_{k:05d}"
            if process_sample(sp, completion, "positive", sid):
                stats["kept"] += 1

    n_pos = stats["kept"]
    for i in range(int(n_pos * spec.negative_frac)):
        sp = gen_negative({"base_price": float(rng.uniform(15, 800)), "volatility": float(rng.uniform(0.005, 0.025))}, rng)
        if process_sample(sp, 0.0, "negative", f"neg_{i:05d}"):
            stats["negatives"] += 1
    for i in range(int(n_pos * spec.hard_negative_frac)):
        sp = gen_hard_negative({"base_price": float(rng.uniform(15, 800)), "volatility": float(rng.uniform(0.005, 0.02)),
                                "pattern_bars": (30, 90), "pre_bars": 25,
                                "m_height": (0.06, 0.2), "m_depth": (0.02, 0.08)}, rng)
        if process_sample(sp, 0.0, "negative", f"hardneg_{i:05d}"):
            stats["negatives"] += 1

    for split in ("train", "valid", "test"):
        manifest["splits"][split] = sum(1 for s in manifest["samples"] if s["split"] == split)

    # data.yaml for ultralytics
    class_names = registry.cv_class_names()
    data_yaml = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    import yaml as _yaml

    (root / "data.yaml").write_text(_yaml.safe_dump(data_yaml, sort_keys=False))
    manifest["stats"] = stats
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[dataset] {version}: kept={stats['kept']} positives, {stats['negatives']} negatives, "
          f"verify-fail={stats['failed_verification']}, splits={manifest['splits']}")
    print(f"[dataset] root: {root}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic pattern dataset")
    ap.add_argument("--samples", type=int, default=60, help="samples per pattern")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out", type=str, default="artifacts/datasets")
    args = ap.parse_args()
    spec = DatasetSpec(
        samples_per_pattern=args.samples, seed=args.seed,
        verify=not args.no_verify, out_dir=Path(args.out),
    )
    generate_dataset(spec)


if __name__ == "__main__":
    main()
