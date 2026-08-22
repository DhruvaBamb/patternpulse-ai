"""API integration tests (in-memory SQLite, scanner disabled, fake market data)."""

from __future__ import annotations

import pytest


class TestRegistryRoutes:
    def test_list_patterns(self, client):
        r = client.get("/api/registry/patterns")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 70
        assert "registry_version" in body

    def test_pattern_detail(self, client):
        r = client.get("/api/registry/patterns/cup_and_handle")
        assert r.status_code == 200
        assert r.json()["stages"], "cup_and_handle must define formation stages"

    def test_unknown_pattern_404(self, client):
        r = client.get("/api/registry/patterns/not_a_pattern")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "unknown_pattern"


class TestSymbolRoutes:
    def test_candles(self, client):
        r = client.get("/api/symbols/TEST.NS/candles?interval=1d&lookback=100")
        assert r.status_code == 200
        body = r.json()
        assert len(body["candles"]) == 100
        c = body["candles"][0]
        assert {"ts", "open", "high", "low", "close", "volume"} <= set(c)

    def test_markets(self, client):
        r = client.get("/api/markets")
        assert r.status_code == 200
        assert any(s["symbol"] == "RELIANCE.NS" for s in r.json()["symbols"])

    def test_symbol_patterns_scan(self, client):
        r = client.get("/api/symbols/TEST.NS/patterns?interval=1d&fresh=true")
        assert r.status_code == 200
        body = r.json()
        assert "cv_status" in body
        assert body["cv_status"]["available"] is False  # no trained weights in test env


class TestPatternRoutes:
    def test_radar_shape(self, client):
        client.get("/api/symbols/TEST.NS/patterns?interval=1d&fresh=true")
        r = client.get("/api/radar")
        assert r.status_code == 200
        rows = r.json()["rows"]
        for row in rows:
            assert {"symbol", "pattern", "completion", "confidence", "interval"} <= set(row)
            assert 0 <= row["completion"] <= 100
            assert 0 <= row["confidence"] <= 1

    def test_active_patterns_filters(self, client):
        client.get("/api/symbols/TEST.NS/patterns?interval=1d&fresh=true")
        r = client.get("/api/patterns/active?min_confidence=0.9")
        assert r.status_code == 200
        for p in r.json()["patterns"]:
            assert p["confidence"] >= 0.9


class TestWatchlistAlerts:
    def test_watchlist_roundtrip(self, client):
        assert client.post("/api/watchlist", json={"symbol": "TEST.NS"}).status_code == 200
        r = client.get("/api/watchlist")
        assert any(i["symbol"] == "TEST.NS" for i in r.json()["items"])
        assert client.delete("/api/watchlist/TEST.NS").status_code == 200
        assert not any(i["symbol"] == "TEST.NS" for i in client.get("/api/watchlist").json()["items"])

    def test_alert_creation(self, client):
        r = client.post("/api/alerts", json={
            "pattern_id": "cup_and_handle",
            "condition": {"type": "completion", "value": 70, "confidence_min": 0.85},
        })
        assert r.status_code == 200
        alerts = client.get("/api/alerts").json()["alerts"]
        assert any(a["pattern_id"] == "cup_and_handle" for a in alerts)


class TestChat:
    def test_chat_status_reports_missing_key(self, client):
        r = client.get("/api/chat/status")
        assert r.status_code == 200
        body = r.json()
        assert "available" in body

    def test_chat_stream_endpoint_shape(self, client):
        with client.stream("POST", "/api/chat", json={"message": "hello"}) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "cv_detector" in body
