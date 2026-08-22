"""End-to-end detection pipeline.

OHLCV window -> render -> CV detections + geometry validation -> fusion ->
completion -> explainable signal. The pipeline is pure computation over the
window; persistence and scheduling live in the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..completion.estimator import CompletionEstimate, estimate_completion
from ..datasets.rendering.renderer import CandlestickRenderer, RenderConfig
from ..geometry.common import ValidationResult
from ..geometry.engine import GeometryEngine
from ..geometry.pivots import atr, classify_trend
from ..inference.detector import Detection, PatternDetector
from ..inference.fusion import FusionEngine, FusionResult
from ..registry.definitions import PatternDefinition
from ..registry.registry import PatternRegistry, get_registry


@dataclass
class PatternFinding:
    """One fused, completion-aware pattern observation for a window."""

    pattern_id: str
    pattern_name: str
    category: str
    direction: str
    fused_confidence: float
    geometry_score: float
    visual_score: float | None
    completion: CompletionEstimate
    levels: dict[str, float]
    keypoints: dict
    lines: dict
    span: tuple[int, int] | None
    violations: list[str]
    source: str
    detected_breakout: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "category": self.category,
            "direction": self.direction,
            "fused_confidence": round(self.fused_confidence, 4),
            "geometry_score": round(self.geometry_score, 4),
            "visual_score": self.visual_score if self.visual_score is None else round(self.visual_score, 4),
            "completion": self.completion.to_dict(),
            "levels": {k: round(v, 4) for k, v in self.levels.items() if v is not None},
            "keypoints": self.keypoints,
            "lines": self.lines,
            "span": list(self.span) if self.span else None,
            "violations": self.violations,
            "source": self.source,
            "detected_breakout": self.detected_breakout,
            "meta": self.meta,
        }
        return d


@dataclass
class PipelineConfig:
    min_confidence: float = 0.60
    min_completion: float = 0.30
    fusion_weights: dict | None = None
    geometry_only_floor: float = 0.55
    conf_threshold: float = 0.25
    image_size: int = 640
    device: str = "auto"


DEFAULT_PIPELINE_CONFIG = PipelineConfig()


class DetectionPipeline:
    def __init__(
        self,
        registry: PatternRegistry | None = None,
        renderer: CandlestickRenderer | None = None,
        detector: PatternDetector | None = None,
        geometry: GeometryEngine | None = None,
        fusion: FusionEngine | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.renderer = renderer or CandlestickRenderer(RenderConfig())
        self.geometry = geometry or GeometryEngine(self.registry)
        self.fusion = fusion or FusionEngine()
        self.detector = detector or PatternDetector(None, self.registry)
        self.config = config or DEFAULT_PIPELINE_CONFIG

    # ── main entry ───────────────────────────────────────────────────────────
    def analyze(self, df: pd.DataFrame, max_findings: int = 5) -> list[PatternFinding]:
        """Analyse one validated OHLCV window; returns findings sorted by fused confidence."""
        cv_detections = self._cv_detect(df)
        geometry_results = self.geometry.validate(df)

        findings: list[PatternFinding] = []
        used_geometry: set[str] = set()

        # 1) CV-seeded findings (visual -> geometry cross-check)
        for det in cv_detections:
            if det.pattern_id in used_geometry:
                continue
            geo = next((g for g in geometry_results if g.pattern_id == det.pattern_id), None)
            defn = self.registry.get(det.pattern_id)
            if geo is None or not geo.matched:
                # CV proposal without geometric support gets a synthetic reject result
                geo = ValidationResult(
                    pattern_id=det.pattern_id, matched=False, geometry_score=0.0,
                    violations=["geometry engine found no supporting structure"],
                )
            used_geometry.add(det.pattern_id)
            finding = self._build_finding(defn, geo, df, visual_score=det.confidence,
                                          cv_class_confirmed=True)
            if finding:
                findings.append(finding)

        # 2) geometry-only findings (independent witness; runs even without a model)
        for geo in geometry_results:
            if geo.pattern_id in used_geometry or not geo.matched:
                continue
            defn = self.registry.get(geo.pattern_id)
            cv_confirmed = None
            if cv_detections:
                cv_confirmed = any(d.pattern_id == geo.pattern_id for d in cv_detections)
            finding = self._build_finding(defn, geo, df, visual_score=None,
                                          cv_class_confirmed=cv_confirmed)
            if finding:
                findings.append(finding)

        findings.sort(key=lambda f: f.fused_confidence, reverse=True)
        return findings[:max_findings]

    # ── helpers ─────────────────────────────────────────────────────────────
    def _cv_detect(self, df: pd.DataFrame) -> list[Detection]:
        if not self.detector.available:
            return []
        rendered = self.renderer.render(df, config=None)
        return self.detector.detect(rendered.image)

    def _build_finding(
        self,
        defn: PatternDefinition,
        geo: ValidationResult,
        df: pd.DataFrame,
        visual_score: float | None,
        cv_class_confirmed: bool | None,
    ) -> PatternFinding | None:
        fused = self.fusion.fuse(defn, geo, df, visual_score, cv_class_confirmed)
        completion = estimate_completion(defn, geo)

        # gates
        if visual_score is None and geo.geometry_score < self.config.geometry_only_floor * 0.9:
            # structured partial formations may publish at reduced confidence
            partial_ok = (
                geo.stages_present
                and geo.geometry_score >= 0.35
                and completion.completion / 100.0 >= self.config.min_completion
                and (geo.span is not None)
            )
            if not partial_ok:
                return None
        if fused.fused_confidence < self.config.min_confidence * 0.75:
            # keep borderline CV+geometry agreements, drop weak geometry-only noise
            if not (visual_score is not None and geo.matched and fused.fused_confidence >= 0.45):
                return None
        if defn.supports_partial_detection and completion.completion / 100.0 < self.config.min_completion * 0.75:
            if visual_score is None:
                return None

        breakout = geo.stages_present and any(
            s in ("breakout", "breakdown", "neckline_break", "reversal") for s in geo.stages_present
        )
        return PatternFinding(
            pattern_id=defn.id,
            pattern_name=defn.name,
            category=defn.category,
            direction=defn.direction,
            fused_confidence=fused.fused_confidence,
            geometry_score=geo.geometry_score,
            visual_score=visual_score,
            completion=completion,
            levels=dict(geo.levels),
            keypoints=geo.keypoints,
            lines=geo.lines,
            span=geo.span,
            violations=geo.violations,
            source=fused.source,
            detected_breakout=bool(breakout),
            meta={"fusion": fused.to_dict(), "geometry_meta": geo.meta, "trend": classify_trend(df)},
        )


def quality_score(finding: PatternFinding, historical_hit_rate: float | None = None) -> dict:
    """Explainable Pattern Quality Score 0-100 (optional feature #77)."""
    w = {
        "confidence": 0.35,
        "geometry": 0.25,
        "completion": 0.20,
        "breakout": 0.10,
        "history": 0.10,
    }
    parts = {
        "confidence": finding.fused_confidence,
        "geometry": finding.geometry_score,
        "completion": finding.completion.completion / 100.0,
        "breakout": 1.0 if finding.detected_breakout else 0.0,
        "history": historical_hit_rate if historical_hit_rate is not None else 0.5,
    }
    total = sum(w[k] * parts[k] for k in w)
    return {
        "quality_score": int(round(total * 100)),
        "components": {k: round(v, 3) for k, v in parts.items()},
        "weights": w,
        "note": "history component defaults to 0.5 when no backtest statistics exist yet",
    }
