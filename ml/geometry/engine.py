"""Geometry validation engine — dispatches registry validators over OHLC windows."""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from ..registry.definitions import PatternDefinition
from ..registry.registry import PatternRegistry, get_registry
from .classical.validators import CLASSICAL_VALIDATORS
from .candlestick.rules import CANDLESTICK_RULES
from .common import ValidationResult
from .harmonic.validators import validate_harmonic

HARMONIC = {"harmonic"}


def _run_one(defn: PatternDefinition, df: pd.DataFrame) -> ValidationResult:
    validator_name = defn.geometry_validator
    if not validator_name:
        return ValidationResult(
            pattern_id=defn.id, matched=False, geometry_score=0.0,
            violations=["pattern has no geometry validator"],
        )
    if defn.category == "harmonic":
        return validate_harmonic(df, defn.validator_params, defn.id)
    if defn.category == "candlestick":
        rule = CANDLESTICK_RULES.get(validator_name)
        if rule is None:
            return ValidationResult(defn.id, False, 0.0, violations=[f"unknown candlestick rule {validator_name}"])
        return rule(df, defn.validator_params, defn.id)
    fn = CLASSICAL_VALIDATORS.get(validator_name)
    if fn is None:
        return ValidationResult(defn.id, False, 0.0, violations=[f"unknown classical validator {validator_name}"])
    return fn(df, defn.validator_params, defn.id)


class GeometryEngine:
    """Runs one or many pattern validators against a validated OHLCV window."""

    def __init__(self, registry: PatternRegistry | None = None) -> None:
        self.registry = registry or get_registry()

    def validate(
        self,
        df: pd.DataFrame,
        pattern_ids: Iterable[str] | None = None,
        category: str | None = None,
        min_score: float = 0.5,
    ) -> list[ValidationResult]:
        """Validate the window; returns results sorted by geometry score desc.

        ``pattern_ids`` narrows the search; ``category`` filters by family.
        Results with ``matched == False`` are included (the fusion engine and
        completion estimator consume near-misses too) but sorted last.
        """
        ids = list(pattern_ids) if pattern_ids else None
        if ids is None:
            if category:
                ids = [d.id for d in self.registry.by_category(category)]
            else:
                ids = [d.id for d in self.registry.implemented()]
        results: list[ValidationResult] = []
        for pid in ids:
            defn = self.registry.get(pid)
            if not defn.implemented:
                continue
            if len(df) < defn.min_bars:
                continue
            try:
                r = _run_one(defn, df)
            except Exception as exc:  # validator bugs must not kill a scan
                r = ValidationResult(defn.id, False, 0.0, violations=[f"validator error: {exc}"])
            results.append(r)
        results.sort(key=lambda r: (r.matched, r.geometry_score), reverse=True)
        return results

    def best(self, df: pd.DataFrame, pattern_ids: Iterable[str] | None = None) -> ValidationResult | None:
        results = self.validate(df, pattern_ids=pattern_ids)
        return results[0] if results else None

    def scan_timed(self, df: pd.DataFrame) -> tuple[list[ValidationResult], float]:
        t0 = time.perf_counter()
        results = self.validate(df)
        return results, time.perf_counter() - t0
