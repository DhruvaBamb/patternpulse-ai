"""Pattern fusion engine.

Combines the CV detector's confidence with the independent geometry score,
trend context and volume confirmation into one calibrated confidence.
Weights live in ``configs/app.yaml`` (``fusion:``) and can be optimized
against validation data with ``ml.evaluation.optimize_fusion``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..geometry.common import ValidationResult
from ..geometry.pivots import classify_trend
from ..registry.definitions import PatternDefinition

DEFAULT_WEIGHTS = {"visual": 0.45, "geometry": 0.35, "trend": 0.10, "volume": 0.10}


@dataclass
class FusionComponents:
    visual_score: float | None      # raw detector confidence for THIS pattern (None => no model)
    geometry_score: float
    trend_score: float
    volume_score: float
    cv_class_confirmed: bool | None  # did the CV box class agree with this pattern id?
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    pattern_id: str
    fused_confidence: float
    components: FusionComponents
    source: str                      # "cv+geometry" | "geometry_only"
    agreement_penalty_applied: bool

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "fused_confidence": round(self.fused_confidence, 4),
            "source": self.source,
            "agreement_penalty_applied": self.agreement_penalty_applied,
            "components": {
                "visual_score": self.components.visual_score,
                "geometry_score": round(self.components.geometry_score, 4),
                "trend_score": round(self.components.trend_score, 4),
                "volume_score": round(self.components.volume_score, 4),
            },
        }


def trend_alignment_score(defn: PatternDefinition, df: pd.DataFrame) -> float:
    """How well the *pre-pattern* trend matches the pattern's requirement."""
    trend = classify_trend(df)
    direction = defn.direction
    if direction == "neutral":
        return 0.6  # neutral patterns neither need nor reject a trend
    # reversal patterns want the opposite trend before them; continuation
    # patterns (flags/pennants) want a trend aligned with their direction.
    continuation = defn.id.endswith(("flag", "pennant")) or "channel" in defn.id or "rectangle" in defn.id
    want = direction if continuation else ("down" if direction == "bullish" else "up")
    if trend == want:
        return 1.0
    if trend == "sideways":
        return 0.5
    return 0.1


def volume_confirmation_score(df: pd.DataFrame, lookback: int = 20) -> float:
    """Recent volume vs its mean — mild confirmation evidence only."""
    if "volume" not in df.columns or len(df) < lookback:
        return 0.5
    vol = df["volume"].tail(lookback).to_numpy(dtype=float)
    mean = float(np.mean(vol[:-3])) if len(vol) > 3 else float(np.mean(vol))
    if mean <= 0:
        return 0.5
    recent = float(np.mean(vol[-3:]))
    ratio = recent / mean
    return float(np.clip(0.5 + (ratio - 1.0) * 0.5, 0.0, 1.0))


class FusionEngine:
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        geometry_only_floor: float = 0.55,
        no_visual_penalty: float = 0.85,
    ) -> None:
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        total = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.geometry_only_floor = geometry_only_floor
        self.no_visual_penalty = no_visual_penalty

    def fuse(
        self,
        defn: PatternDefinition,
        geometry: ValidationResult,
        df: pd.DataFrame,
        visual_score: float | None = None,
        cv_class_confirmed: bool | None = None,
    ) -> FusionResult:
        comps = FusionComponents(
            visual_score=visual_score,
            geometry_score=geometry.geometry_score,
            trend_score=trend_alignment_score(defn, df),
            volume_score=volume_confirmation_score(df),
            cv_class_confirmed=cv_class_confirmed,
        )

        parts: list[tuple[float, float]] = []  # (weight, score)
        if visual_score is not None:
            parts.append((self.weights["visual"], visual_score))
        parts.append((self.weights["geometry"], comps.geometry_score))
        parts.append((self.weights["trend"], comps.trend_score))
        parts.append((self.weights["volume"], comps.volume_score))
        wsum = sum(w for w, _ in parts) or 1.0
        fused = sum(w * s for w, s in parts) / wsum

        # Evidence penalty when the CV branch is absent (geometry-only mode):
        # fewer independent witnesses => cap the claimable confidence.
        source = "cv+geometry" if visual_score is not None else "geometry_only"
        if visual_score is None:
            fused *= self.no_visual_penalty
        # Disagreement penalty: CV saw a different class than the geometry hit
        penalty = False
        if cv_class_confirmed is False:
            fused *= 0.55
            penalty = True
        # geometry hard rejection suppresses even a confident CV box
        if comps.geometry_score < 0.25 and visual_score is not None:
            fused *= 0.5
            penalty = True

        fused = float(np.clip(fused, 0.0, 1.0))
        return FusionResult(
            pattern_id=defn.id,
            fused_confidence=fused,
            components=comps,
            source=source,
            agreement_penalty_applied=penalty,
        )
