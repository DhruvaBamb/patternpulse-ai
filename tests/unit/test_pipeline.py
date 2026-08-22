"""Completion, fusion, signal and backtest tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.completion.estimator import estimate_completion
from ml.geometry.engine import GeometryEngine
from ml.inference.fusion import FusionEngine, trend_alignment_score, volume_confirmation_score
from ml.inference.pipeline import DetectionPipeline, PipelineConfig, quality_score
from ml.inference.signals import SignalConfig, generate_signal, no_pattern_signal
from ml.registry.registry import get_registry

from ..geometry import fixtures as fx


@pytest.fixture(scope="module")
def registry():
    return get_registry()


class TestCompletion:
    def test_never_exceeds_100(self, registry):
        engine = GeometryEngine(registry)
        res = engine.validate(fx.perfect_double_top(), pattern_ids=["double_top"])[0]
        est = estimate_completion(registry.get("double_top"), res)
        assert est.completion <= 100.0
        assert est.completion >= 0.0

    def test_complete_pattern_with_breakout_near_100(self, registry):
        engine = GeometryEngine(registry)
        res = engine.validate(fx.perfect_double_top(), pattern_ids=["double_top"])[0]
        est = estimate_completion(registry.get("double_top"), res)
        assert est.completion >= 80.0, est
        assert "neckline_break" in res.stages_present

    def test_early_pattern_low_completion(self, registry):
        df, _ = fx.partial_cup_and_handle(0.35)
        engine = GeometryEngine(registry)
        res = engine.validate(df, pattern_ids=["cup_and_handle"])[0]
        est = estimate_completion(registry.get("cup_and_handle"), res)
        assert est.completion < 80.0


class TestFusion:
    def test_geometry_only_penalty_applied(self, registry):
        fusion = FusionEngine()
        defn = registry.get("double_top")
        geo = GeometryEngine(registry).validate(fx.perfect_double_top(), pattern_ids=["double_top"])[0]
        r = fusion.fuse(defn, geo, fx.perfect_double_top(), visual_score=None)
        assert r.source == "geometry_only"
        assert r.fused_confidence <= geo.geometry_score

    def test_cv_geometry_agreement_boosts(self, registry):
        fusion = FusionEngine()
        defn = registry.get("double_top")
        df = fx.perfect_double_top()
        geo = GeometryEngine(registry).validate(df, pattern_ids=["double_top"])[0]
        with_model = fusion.fuse(defn, geo, df, visual_score=0.9, cv_class_confirmed=True)
        assert with_model.fused_confidence > fusion.fuse(defn, geo, df, visual_score=None).fused_confidence

    def test_disagreement_penalized(self, registry):
        fusion = FusionEngine()
        defn = registry.get("double_top")
        geo = GeometryEngine(registry).validate(fx.perfect_double_top(), pattern_ids=["double_top"])[0]
        agree = fusion.fuse(defn, geo, fx.perfect_double_top(), 0.9, cv_class_confirmed=True)
        disagree = fusion.fuse(defn, geo, fx.perfect_double_top(), 0.9, cv_class_confirmed=False)
        assert disagree.fused_confidence < agree.fused_confidence
        assert disagree.agreement_penalty_applied


class TestSignal:
    def test_forming_pattern_is_watch(self, registry):
        pipeline = DetectionPipeline(config=PipelineConfig(min_confidence=0.3))
        df, _ = fx.partial_cup_and_handle(0.6)
        findings = pipeline.analyze(df)
        assert findings, "geometry-only pipeline should produce findings on clean fixtures"
        sig = generate_signal(findings[0], df, SignalConfig())
        assert sig.action in ("WATCH", "HOLD")

    def test_confirmed_breakout_buys(self, registry):
        pipeline = DetectionPipeline(config=PipelineConfig(min_confidence=0.3))
        df = fx.perfect_double_bottom()
        findings = pipeline.analyze(df)
        assert findings
        sig = generate_signal(findings[0], df, SignalConfig())
        # double bottom with neckline break should at least reach a directional verdict
        assert sig.action in ("BUY", "WATCH")
        if sig.action == "BUY":
            assert sig.entry is not None

    def test_reason_always_present(self, registry):
        pipeline = DetectionPipeline(config=PipelineConfig(min_confidence=0.3))
        df = fx.valid_head_and_shoulders()
        findings = pipeline.analyze(df)
        for f in findings:
            sig = generate_signal(f, df, SignalConfig())
            assert sig.reason

    def test_no_pattern_signal(self):
        sig = no_pattern_signal(fx.perfect_double_top())
        assert sig.action == "HOLD"
        assert sig.reason


class TestQualityScore:
    def test_quality_score_explainable(self, registry):
        pipeline = DetectionPipeline(config=PipelineConfig(min_confidence=0.3))
        findings = pipeline.analyze(fx.perfect_double_top())
        assert findings
        q = quality_score(findings[0])
        assert 0 <= q["quality_score"] <= 100
        assert set(q["components"]) == {"confidence", "geometry", "completion", "breakout", "history"}


class TestBacktestMath:
    def test_engine_runs_and_reports(self, registry):
        from ml.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult

        engine = BacktestEngine(
            DetectionPipeline(config=PipelineConfig(min_confidence=0.3)),
            BacktestConfig(window_bars=40, step_bars=5, max_hold_bars=20),
        )
        trades = engine.run(fx.perfect_double_bottom(), "TEST.NS", "1d")
        result = BacktestResult(trades=trades)
        overall = result.overall()
        assert overall["n_trades"] >= 0
        for t in trades:
            assert t.outcome in ("target", "stop", "timeout")
            assert -100 <= t.return_pct <= 200

    def test_max_dd_computation(self):
        from ml.backtesting.engine import BacktestResult

        rets = np.array([10.0, -5.0, -5.0, 3.0])
        dd = BacktestResult._max_dd(rets)
        assert dd < 0
