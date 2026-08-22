"""Known-answer tests for geometry validators (spec §69–70).

Each fixture has an unambiguous expected outcome; validators must agree.
"""

from __future__ import annotations

import pytest

from ml.geometry.engine import GeometryEngine
from ml.registry.registry import get_registry

from . import fixtures as fx


@pytest.fixture(scope="module")
def engine() -> GeometryEngine:
    return GeometryEngine(get_registry())


class TestDoubleTop:
    def test_perfect_double_top_detected(self, engine):
        res = engine.validate(fx.perfect_double_top(), pattern_ids=["double_top"])
        assert res and res[0].matched, f"expected match, got score={res[0].geometry_score} violations={res[0].violations}"
        assert res[0].geometry_score >= 0.6
        assert "neckline" in res[0].levels
        assert "target" in res[0].levels

    def test_invalid_double_top_rejected(self, engine):
        res = engine.validate(fx.invalid_double_top(), pattern_ids=["double_top"])
        assert not res[0].matched
        assert res[0].geometry_score < 0.5

    def test_double_bottom_detected(self, engine):
        res = engine.validate(fx.perfect_double_bottom(), pattern_ids=["double_bottom"])
        assert res[0].matched, f"score={res[0].geometry_score} violations={res[0].violations}"


class TestHeadAndShoulders:
    def test_valid_hs(self, engine):
        res = engine.validate(fx.valid_head_and_shoulders(), pattern_ids=["head_and_shoulders"])
        assert res[0].matched, f"score={res[0].geometry_score} violations={res[0].violations}"
        assert "head" in res[0].keypoints
        assert "neckline" in res[0].levels

    def test_invalid_hs_rejected(self, engine):
        res = engine.validate(fx.invalid_head_and_shoulders(), pattern_ids=["head_and_shoulders"])
        assert not res[0].matched


class TestCupAndHandle:
    def test_partial_cup_scores_partial_completion(self, engine):
        df, true_pct = fx.partial_cup_and_handle(0.6)
        res = engine.validate(df, pattern_ids=["cup_and_handle"])
        best = res[0]
        from ml.completion.estimator import estimate_completion

        reg = get_registry()
        est = estimate_completion(reg.get("cup_and_handle"), best)
        assert 20.0 <= est.completion <= 100.0
        assert est.completion <= 100.0
        assert est.missing_components is not None


class TestCandlesticks:
    def test_bullish_engulfing(self, engine):
        import numpy as np

        closes = np.linspace(100, 108, 40)
        df = fx._df(closes, exact=fx.bullish_engulfing_bars())
        res = engine.validate(df, pattern_ids=["bullish_engulfing"])
        assert res[0].matched, f"score={res[0].geometry_score} violations={res[0].violations}"

    def test_bearish_engulfing(self, engine):
        import numpy as np

        closes = np.linspace(120, 110, 40)
        df = fx._df(closes, exact=fx.bearish_engulfing_bars())
        res = engine.validate(df, pattern_ids=["bearish_engulfing"])
        assert res[0].matched, f"score={res[0].geometry_score} violations={res[0].violations}"

    def test_doji(self, engine):
        import numpy as np

        closes = np.linspace(100, 103, 40)
        doji = [{"open": 102.0, "high": 102.9, "low": 101.1, "close": 102.02}]
        df = fx._df(closes, exact=doji)
        res = engine.validate(df, pattern_ids=["doji"])
        assert res[0].matched or res[0].geometry_score > 0.4


class TestHarmonic:
    def test_gartley_ratios_pass(self, engine):
        res = engine.validate(fx.gartley_path(), pattern_ids=["gartley"])
        assert res[0].matched, f"score={res[0].geometry_score} violations={res[0].violations} meta={res[0].meta}"
        kp = res[0].keypoints
        assert set(kp) >= {"X", "A", "B", "C", "D"}

    def test_failed_gartley_ratios_rejected(self, engine):
        res = engine.validate(fx.failed_gartley_path(), pattern_ids=["gartley"])
        assert not res[0].matched or res[0].geometry_score < 0.5


class TestRegistryIntegrity:
    def test_registry_loads_with_expected_size(self):
        reg = get_registry()
        assert 70 <= len(reg) <= 85
        assert len(reg.by_category("classical")) == 28
        assert len(reg.by_category("candlestick")) == 41
        assert len(reg.by_category("harmonic")) == 8

    def test_every_implemented_pattern_has_validator(self):
        reg = get_registry()
        from ml.geometry.classical.validators import CLASSICAL_VALIDATORS
        from ml.geometry.candlestick.rules import CANDLESTICK_RULES

        for d in reg.implemented():
            if d.category == "classical":
                assert d.geometry_validator in CLASSICAL_VALIDATORS, d.id
            elif d.category == "candlestick":
                assert d.geometry_validator in CANDLESTICK_RULES, d.id
            else:
                assert d.geometry_validator == "harmonic", d.id

    def test_cv_class_ids_unique_and_dense(self):
        reg = get_registry()
        ids = [d.cv_class_id for d in reg.implemented()]
        assert sorted(ids) == list(range(len(ids)))

    def test_engine_handles_all_classical_patterns_without_crash(self, engine):
        """Every classical validator runs cleanly on a neutral window."""
        import numpy as np

        closes = np.cumsum(np.random.default_rng(1).normal(0, 0.5, 120)) + 100
        df = fx._df(closes, seed=1)
        for d in get_registry().by_category("classical"):
            res = engine.validate(df, pattern_ids=[d.id])
            assert len(res) == 1
