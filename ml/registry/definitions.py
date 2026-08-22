"""Pattern definition dataclasses for the central PatternPulse registry.

Every module in the platform (geometry validators, completion estimators, CV
dataset generator, signal engine, frontend explorer, chatbot) resolves pattern
metadata through :class:`PatternRegistry` — never through hard-coded lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PatternCategory = Literal["classical", "candlestick", "harmonic"]
PatternDirection = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class PatternDefinition:
    """One entry of the pattern registry (see ``ml/registry/patterns/*.yaml``)."""

    id: str
    name: str
    category: PatternCategory
    direction: PatternDirection
    description: str
    min_bars: int
    max_bars: int
    geometry_validator: str | None
    supports_partial_detection: bool
    implemented: bool
    confirmation_rules: list[str] = field(default_factory=list)
    invalidation_rules: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    confused_with: list[str] = field(default_factory=list)
    validator_params: dict[str, Any] = field(default_factory=dict)
    cv_class_id: int | None = None  # assigned by the loader over implemented patterns

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "direction": self.direction,
            "description": self.description,
            "min_bars": self.min_bars,
            "max_bars": self.max_bars,
            "geometry_validator": self.geometry_validator,
            "supports_partial_detection": self.supports_partial_detection,
            "implemented": self.implemented,
            "confirmation_rules": self.confirmation_rules,
            "invalidation_rules": self.invalidation_rules,
            "stages": self.stages,
            "confused_with": self.confused_with,
            "cv_class_id": self.cv_class_id,
        }
        return out


# Category-level defaults applied by the loader so YAML entries stay compact.
CATEGORY_DEFAULTS: dict[PatternCategory, dict[str, Any]] = {
    "classical": {
        "min_bars": 20,
        "max_bars": 250,
        "supports_partial_detection": True,
        "implemented": True,
    },
    "candlestick": {
        "min_bars": 1,
        "max_bars": 5,
        "supports_partial_detection": False,  # completes on the final candle close
        "implemented": True,
    },
    "harmonic": {
        "min_bars": 30,
        "max_bars": 200,
        "supports_partial_detection": True,
        "implemented": True,
    },
}
