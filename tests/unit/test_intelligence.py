"""Tests for the optional intelligence features (§76–79) and drift."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select

from backend.app.services.analysis import AnalysisService
from backend.app.db.models import ActiveDetection, Signal
from backend.app.db.session import Base, SessionLocal, engine


@pytest.fixture()
def db():
    Base.metadata.create_all(engine)
    session = SessionLocal()
    yield session
    session.query(ActiveDetection).delete()
    session.query(Signal).delete()
    session.commit()
    session.close()


def _det(**kw):
    defaults = dict(
        id=f"PATTERN-TEST-{kw.get('interval', '1d')}-{kw.get('pattern_id', 'x')}",
        symbol_ticker="TEST.NS", interval="1d", pattern_id="cup_and_handle",
        pattern_name="Cup & Handle", category="classical", direction="bullish",
        status="FORMING", completion=60.0, confidence=0.7, geometry_score=0.8,
    )
    defaults.update(kw)
    return ActiveDetection(**defaults)


class TestSimilarHistorical:
    def test_reports_when_history_empty(self, db):
        db.add(_det(id="D1"))
        db.commit()
        svc = AnalysisService()
        out = svc.similar_historical(db, "D1")
        assert out["similar_found"] == 0
        assert "note" in out

    def test_counts_confirmed_history(self, db):
        db.add(_det(id="D1"))
        db.add(_det(id="D2", status="CONFIRMED", completion=58.0, geometry_score=0.78))
        db.add(_det(id="D3", status="INVALIDATED"))
        db.commit()
        out = AnalysisService().similar_historical(db, "D1")
        assert out["similar_found"] == 2
        assert out["confirmed"] == 1

    def test_unknown_detection(self, db):
        assert "error" in AnalysisService().similar_historical(db, "nope")


class TestConfluence:
    def test_no_detections(self, db):
        out = AnalysisService().confluence(db, "TEST.NS", "1d")
        assert out["score"] == 0

    def test_aligned_bullish_patterns_score(self, db):
        for i in range(3):
            db.add(_det(id=f"B{i}", pattern_id=f"p{i}", confidence=0.8, completion=85.0))
        db.add(_det(id="S1", pattern_id="s1", direction="bearish", confidence=0.5))
        db.commit()
        out = AnalysisService().confluence(db, "TEST.NS", "1d")
        assert out["direction"] == "bullish"
        assert 0 < out["score"] <= 100
        assert out["factors"]

    def test_confirmed_breakout_boosts(self, db):
        db.add(_det(id="C1", status="CONFIRMED", confidence=0.9, completion=100.0))
        db.commit()
        out = AnalysisService().confluence(db, "TEST.NS", "1d")
        assert any("confirmed breakout" in f["factor"] for f in out["factors"])


class TestCrossTimeframe:
    def test_empty_symbol(self, db):
        out = AnalysisService().cross_timeframe(db, "TEST.NS", ["15m", "1h", "1d"])
        assert out["context_score"] is None
        assert out["note"]

    def test_alignment_across_timeframes(self, db):
        db.add(_det(id="T1", interval="15m", pattern_id="a"))
        db.add(_det(id="T2", interval="1h", pattern_id="b"))
        db.add(_det(id="T3", interval="1d", pattern_id="c"))
        db.commit()
        out = AnalysisService().cross_timeframe(db, "TEST.NS", ["15m", "1h", "1d"])
        assert out["alignment"] == "bullish"
        assert out["context_score"] > 0
        assert set(out["timeframes"]) == {"15m", "1h", "1d"}


class TestDrift:
    def test_monitor_window_flags_resolution(self):
        from ml.inference.drift import ReferenceStats, monitor_window

        ref = ReferenceStats(image_mean=40.0, image_std=60.0, edge_density_mean=0.10,
                             resolution=(640, 640), n_images=10)
        # chart-like image: dark background with a few thin bright candles -> low edge density
        img = np.zeros((640, 640, 3), dtype=np.uint8) + 36
        for x in range(40, 600, 12):
            img[100:400, x : x + 2] = 200
        out = monitor_window(img, ref)
        assert out["drifted"] is False, out["warnings"]

        wrong_res = img[:320, :320]
        out2 = monitor_window(wrong_res, ref)
        assert out2["drifted"] is True
        assert any("resolution" in w for w in out2["warnings"])
