"""Pattern completion estimation.

Converts the structural stages a geometry validator observed into a
formation percentage. This is deliberately per-pattern: the stage list
comes from the registry (e.g. Cup & Handle has 7 stages), and the final
confirmation stage is weighted so that a pattern cannot read ~100%
before its decisive event (breakout/breakdown) happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..geometry.common import ValidationResult
from ..registry.definitions import PatternDefinition


@dataclass
class CompletionEstimate:
    pattern_id: str
    completion: float          # 0..100, never above 100
    stage: str                 # current stage label ("complete" when done)
    confidence: float          # confidence of the estimate itself (0..1)
    missing_components: list[str]

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern_id,
            "completion": round(self.completion, 1),
            "stage": self.stage,
            "confidence": round(self.confidence, 3),
            "missing_components": self.missing_components,
        }


FINAL_STAGE_HINTS = {"breakout", "breakdown", "neckline_break", "reversal", "confirmation"}


def estimate_completion(
    defn: PatternDefinition,
    result: ValidationResult,
) -> CompletionEstimate:
    """Estimate how much of ``defn`` has formed given validator ``result``."""

    # Candlestick patterns complete on their final candle close.
    if not defn.supports_partial_detection or not defn.stages:
        completion = 100.0 if result.matched else (result.geometry_score * 100.0)
        stage = "complete" if result.matched else "unconfirmed"
        return CompletionEstimate(
            pattern_id=defn.id,
            completion=min(completion, 100.0),
            stage=stage,
            confidence=max(result.geometry_score, 0.0),
            missing_components=[] if result.matched else ["candle confirmation"],
        )

    stages = defn.stages
    present = [s for s in result.stages_present if s in stages]
    missing = [s for s in stages if s not in present]

    if not present:
        return CompletionEstimate(
            pattern_id=defn.id, completion=0.0, stage="not_forming",
            confidence=result.geometry_score, missing_components=list(stages),
        )

    # Weighting: the final confirmation stage carries 25% of the total; the
    # remaining 75% is spread across the structural stages actually observed.
    final_stage = stages[-1] if len(stages) > 1 else None
    structural = [s for s in stages if s != final_stage]
    structural_present = [s for s in present if s != final_stage]
    if structural:
        base = 0.75 * len(structural_present) / len(structural)
    else:
        base = 0.0
    completion = (base + (0.25 if final_stage in present else 0.0)) * 100.0

    # current stage label: last structural stage reached, or confirmation
    stage = present[-1]
    if final_stage in present:
        stage = final_stage

    conf = 0.5 * result.geometry_score + 0.5 * min(len(present) / max(len(stages), 1), 1.0)
    return CompletionEstimate(
        pattern_id=defn.id,
        completion=min(completion, 100.0),
        stage=stage,
        confidence=min(conf, 1.0),
        missing_components=missing,
    )
