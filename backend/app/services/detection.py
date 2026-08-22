"""Detection service: pipeline execution + formation lifecycle tracking.

Formations persist across scans with stable IDs, completion/confidence
history, and status transitions (EARLY/FORMING/NEAR_COMPLETION/CONFIRMED/
FAILED/INVALIDATED/EXPIRED) — spec §22–24.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.inference.detector import PatternDetector
from ml.inference.pipeline import DetectionPipeline, PatternFinding
from ml.inference.signals import SignalConfig, generate_signal, no_pattern_signal
from ml.registry.registry import PatternRegistry, get_registry
from ..core.config import fusion_config, model_config, scanner_config
from ..core.logging import get_logger
from ..db.models import ActiveDetection, PatternEvent, Signal
from .events import bus
from .market_data import MarketDataService

logger = get_logger("detection")

ACTIVE_STATUSES = {"EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"}


class DetectionService:
    def __init__(
        self,
        market_data: MarketDataService,
        registry: PatternRegistry | None = None,
        pipeline: DetectionPipeline | None = None,
    ) -> None:
        self.market_data = market_data
        self.registry = registry or get_registry()
        mcfg = model_config()
        fcfg = fusion_config()
        scfg = scanner_config()
        from ml.datasets.rendering.renderer import CandlestickRenderer, RenderConfig
        from ml.inference.fusion import FusionEngine
        from ml.inference.pipeline import PipelineConfig

        detector = PatternDetector(
            weights=mcfg.get("weights"),
            registry=self.registry,
            device=mcfg.get("device", "auto"),
            conf_threshold=mcfg.get("confidence_threshold", 0.25),
            image_size=mcfg.get("image_size", 640),
        )
        pcfg = PipelineConfig(
            min_confidence=scfg.get("minimum_confidence", 0.6),
            min_completion=scfg.get("minimum_completion", 0.3),
            fusion_weights=fcfg.get("weights"),
            geometry_only_floor=fcfg.get("geometry_only_floor", 0.55),
            conf_threshold=mcfg.get("confidence_threshold", 0.25),
            image_size=mcfg.get("image_size", 640),
            device=mcfg.get("device", "auto"),
        )
        self.pipeline = pipeline or DetectionPipeline(
            registry=self.registry,
            renderer=CandlestickRenderer(RenderConfig()),
            detector=detector,
            fusion=FusionEngine(weights=fcfg.get("weights"), geometry_only_floor=fcfg.get("geometry_only_floor", 0.55)),
            config=pcfg,
        )
        self.signal_cfg = SignalConfig()

    # ── scan one symbol/timeframe ───────────────────────────────────────────
    def scan(self, db: Session, ticker: str, interval: str, lookback: int = 250) -> list[dict]:
        df = self.market_data.get_candles(db, ticker, interval, lookback)
        findings = self.pipeline.analyze(df)
        out = []
        for f in findings:
            det = self._upsert_detection(db, ticker, interval, f, len(df))
            sig = generate_signal(f, df, self.signal_cfg)
            self._store_signal(db, det, ticker, interval, sig)
            out.append({"detection": self._det_to_dict(det), "signal": sig.to_dict()})
        if not findings:
            sig = no_pattern_signal(df, ticker)
            self._store_signal(db, None, ticker, interval, sig)
            out.append({"detection": None, "signal": sig.to_dict()})
        db.commit()
        return out

    # ── formation lifecycle ────────────────────────────────────────────────
    def _upsert_detection(
        self, db: Session, ticker: str, interval: str, f: PatternFinding, n_bars: int
    ) -> ActiveDetection:
        status = self._status_for(f)
        det = db.execute(
            select(ActiveDetection).where(
                ActiveDetection.symbol_ticker == ticker,
                ActiveDetection.interval == interval,
                ActiveDetection.pattern_id == f.pattern_id,
                ActiveDetection.status.in_(ACTIVE_STATUSES),
            )
        ).scalar_one_or_none()

        now = datetime.now(timezone.utc)
        if det is None:
            det = ActiveDetection(
                id=self._formation_id(ticker, interval),
                symbol_ticker=ticker,
                interval=interval,
                pattern_id=f.pattern_id,
                pattern_name=f.pattern_name,
                category=f.category,
                direction=f.direction,
                pattern_registry_version=self.registry.version,
            )
            db.add(det)
            self._event(db, det, "first_seen", {"status": status})
            bus.publish("detections", {"event": "first_seen", "symbol": ticker, "interval": interval,
                                       "pattern": f.pattern_name, "status": status,
                                       "completion": f.completion.completion,
                                       "confidence": f.fused_confidence})
        det.status = status
        det.completion = f.completion.completion
        det.confidence = f.fused_confidence
        det.geometry_score = f.geometry_score
        det.visual_score = f.visual_score
        det.source = f.source
        det.levels = f.levels
        det.keypoints = f.keypoints
        det.lines = f.lines
        det.violations = f.violations
        det.stage = f.completion.stage
        det.detected_breakout = f.detected_breakout
        det.meta = f.meta
        det.last_seen = now
        det.completion_history = (det.completion_history or []) + [
            {"ts": now.isoformat(), "value": f.completion.completion}
        ][-100:]
        det.confidence_history = (det.confidence_history or []) + [
            {"ts": now.isoformat(), "value": f.fused_confidence}
        ][-100:]
        if status == "CONFIRMED" and det.confirmed_at is None:
            det.confirmed_at = now
            self._event(db, det, "confirmed", {})
            bus.publish("detections", {"event": "confirmed", "symbol": ticker,
                                       "pattern": f.pattern_name})
        prev_stage = (det.meta or {}).get("_last_stage")
        if prev_stage and prev_stage != f.completion.stage:
            self._event(db, det, "stage", {"from": prev_stage, "to": f.completion.stage})
        if f.detected_breakout and not (det.meta or {}).get("_breakout_logged"):
            self._event(db, det, "breakout", {"levels": f.levels})
            meta = dict(det.meta or {})
            meta["_breakout_logged"] = True
            det.meta = meta
        meta = dict(det.meta or {})
        meta["_last_stage"] = f.completion.stage
        det.meta = meta
        return det

    def _status_for(self, f: PatternFinding) -> str:
        c = f.completion.completion
        if f.completion.stage in ("breakout", "breakdown", "neckline_break") or (
            f.detected_breakout and c >= 95
        ):
            return "CONFIRMED"
        if c >= 80:
            return "NEAR_COMPLETION"
        if c >= 25:
            return "FORMING"
        return "EARLY"

    def _formation_id(self, ticker: str, interval: str) -> str:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"PATTERN-{ticker.replace('.', '_')}-{interval.upper()}-{date}-{uid4()[:4]}"

    def _event(self, db: Session, det: ActiveDetection, event: str, detail: dict) -> None:
        db.add(PatternEvent(detection_id=det.id, event=event, detail=detail))

    def _store_signal(self, db: Session, det, ticker: str, interval: str, sig) -> None:
        db.add(Signal(
            detection_id=det.id if det else None,
            symbol_ticker=ticker, interval=interval, action=sig.action,
            reason=sig.reason, payload=sig.to_dict(),
        ))

    def _det_to_dict(self, det: ActiveDetection) -> dict:
        return {
            "id": det.id, "symbol": det.symbol_ticker, "interval": det.interval,
            "pattern_id": det.pattern_id, "pattern_name": det.pattern_name,
            "category": det.category, "direction": det.direction, "status": det.status,
            "completion": det.completion, "confidence": det.confidence,
            "geometry_score": det.geometry_score, "visual_score": det.visual_score,
            "source": det.source, "stage": det.stage,
            "detected_breakout": det.detected_breakout,
            "levels": det.levels, "keypoints": det.keypoints, "lines": det.lines,
            "violations": det.violations,
            "completion_history": det.completion_history,
            "confidence_history": det.confidence_history,
            "first_seen": det.first_seen.isoformat() if det.first_seen else None,
            "last_seen": det.last_seen.isoformat() if det.last_seen else None,
        }

    # ── expiry ─────────────────────────────────────────────────────────────
    def expire_stale(self, db: Session) -> int:
        hours = scanner_config().get("formation_expiry_hours", 48)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stale = db.execute(
            select(ActiveDetection).where(
                ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION"]),
                ActiveDetection.last_seen < cutoff,
            )
        ).scalars().all()
        for det in stale:
            det.status = "EXPIRED"
            det.invalidated_at = datetime.now(timezone.utc)
            self._event(db, det, "expired", {})
        if stale:
            db.commit()
        return len(stale)


def uid4() -> str:
    import uuid

    return uuid.uuid4().hex
