"""Shared result types and scoring helpers for geometry validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Result of checking one pattern definition against one OHLC window."""

    pattern_id: str
    matched: bool
    geometry_score: float                  # 0..1
    violations: list[str] = field(default_factory=list)
    keypoints: dict[str, Any] = field(default_factory=dict)   # {"left_shoulder": {"index": 12, "price": 103.2}, ...}
    lines: dict[str, Any] = field(default_factory=dict)       # {"neckline": {"slope":..,"intercept":..,"level":..}}
    levels: dict[str, float] = field(default_factory=dict)    # breakout/support/resistance/target hint
    span: tuple[int, int] | None = None     # bar index range of the formation within window
    stages_present: list[str] = field(default_factory=list)   # completed structural stages
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern_id,
            "matched": self.matched,
            "geometry_score": round(self.geometry_score, 4),
            "violations": self.violations,
            "keypoints": self.keypoints,
            "lines": self.lines,
            "levels": {k: round(v, 6) for k, v in self.levels.items() if v is not None},
            "span": list(self.span) if self.span else None,
            "stages_present": self.stages_present,
            "meta": self.meta,
        }


def soft_pass(value: float, lo: float, hi: float) -> float:
    """Smooth 0..1 membership: 1 inside [lo, hi], decaying linearly outside.

    Used so near-misses get partial geometry credit instead of hard rejection.
    """
    if lo <= value <= hi:
        return 1.0
    width = max(abs(hi - lo), 1e-9)
    if value < lo:
        d = lo - value
    else:
        d = value - hi
    return float(np.clip(1.0 - d / width, 0.0, 1.0))


def near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))
