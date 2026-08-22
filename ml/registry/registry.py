"""Central pattern registry: loads YAML definitions, versions them, serves lookups."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .definitions import CATEGORY_DEFAULTS, PatternDefinition

REGISTRY_DIR = Path(__file__).parent / "patterns"


def _build(defn: dict) -> PatternDefinition:
    merged = {**CATEGORY_DEFAULTS[defn["category"]], **defn}
    return PatternDefinition(
        id=merged["id"],
        name=merged["name"],
        category=merged["category"],
        direction=merged["direction"],
        description=merged.get("description", ""),
        min_bars=int(merged["min_bars"]),
        max_bars=int(merged["max_bars"]),
        geometry_validator=merged.get("geometry_validator"),
        supports_partial_detection=bool(merged["supports_partial_detection"]),
        implemented=bool(merged.get("implemented", True)),
        confirmation_rules=merged.get("confirmation_rules", []),
        invalidation_rules=merged.get("invalidation_rules", []),
        stages=merged.get("stages", []),
        confused_with=merged.get("confused_with", []),
        validator_params=merged.get("validator_params", {}),
    )


class PatternRegistry:
    """Immutable registry of all supported patterns."""

    def __init__(self, definitions: list[PatternDefinition]) -> None:
        self._by_id: dict[str, PatternDefinition] = {d.id: d for d in definitions}
        if len(self._by_id) != len(definitions):
            raise ValueError("duplicate pattern ids in registry")
        # cv_class_id: stable, order-of-definition ids over implemented patterns
        next_id = 0
        for d in definitions:
            if d.implemented:
                object.__setattr__(d, "cv_class_id", next_id)
                next_id += 1
        self._definitions = list(definitions)
        self._version = hashlib.sha1(
            "\n".join(
                f"{d.id}:{d.name}:{d.category}:{d.direction}:{d.implemented}:{d.geometry_validator}"
                for d in definitions
            ).encode()
        ).hexdigest()[:12]

    # ── accessors ────────────────────────────────────────────────────────────
    @property
    def version(self) -> str:
        return self._version

    def all(self) -> list[PatternDefinition]:
        return list(self._definitions)

    def implemented(self) -> list[PatternDefinition]:
        return [d for d in self._definitions if d.implemented]

    def by_category(self, category: str) -> list[PatternDefinition]:
        return [d for d in self._definitions if d.category == category]

    def get(self, pattern_id: str) -> PatternDefinition:
        try:
            return self._by_id[pattern_id]
        except KeyError:
            raise KeyError(f"unknown pattern id: {pattern_id!r}") from None

    def get_by_name(self, name: str) -> PatternDefinition | None:
        low = name.strip().lower()
        for d in self._definitions:
            if d.name.lower() == low or d.id.replace("_", " ") == low:
                return d
        return None

    def cv_id_to_pattern(self, cv_class_id: int) -> PatternDefinition | None:
        for d in self._definitions:
            if d.cv_class_id == cv_class_id:
                return d
        return None

    def cv_class_names(self) -> list[str]:
        """Ordered class names for YOLO ``data.yaml`` (index == cv_class_id)."""
        return [d.id for d in self._definitions if d.implemented]

    def __len__(self) -> int:
        return len(self._definitions)


def load_registry(directory: Path | None = None) -> PatternRegistry:
    """Load and validate the registry from YAML files."""
    directory = directory or REGISTRY_DIR
    definitions: list[PatternDefinition] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        for defn in raw["patterns"]:
            definitions.append(_build(defn))
    if not definitions:
        raise RuntimeError(f"no pattern definitions found under {directory}")
    return PatternRegistry(definitions)


_CACHE: PatternRegistry | None = None


def get_registry() -> PatternRegistry:
    global _CACHE
    if _CACHE is None:
        _CACHE = load_registry()
    return _CACHE
