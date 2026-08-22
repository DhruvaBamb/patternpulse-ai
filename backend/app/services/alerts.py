"""Alert evaluation engine (in-app notifications now; adapters later)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..db.models import ActiveDetection, Alert, Notification
from .events import bus

logger = get_logger("alerts")


class AlertEngine:
    """Evaluates alert conditions against current detections.

    Condition schema (JSON):
      {"type": "first_detected" | "completion" | "breakout_confirmed" | "invalidated",
       "op": ">", "value": 70, "confidence_min": 0.85, "pattern": "cup_and_handle"}
    """

    COOLDOWN_SECONDS = 1800  # per-alert re-fire cooldown

    def evaluate(self, db: Session) -> int:
        alerts = db.execute(select(Alert).where(Alert.enabled)).scalars().all()
        detections = db.execute(
            select(ActiveDetection).where(ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"]))
        ).scalars().all()
        fired = 0
        now = datetime.now(timezone.utc)
        for alert in alerts:
            if alert.last_fired_at and (now - alert.last_fired_at).total_seconds() < self.COOLDOWN_SECONDS:
                continue
            for det in detections:
                if not self._matches(alert, det):
                    continue
                if self._condition_met(alert.condition, det):
                    self._fire(db, alert, det)
                    fired += 1
                    break
        if fired:
            db.commit()
        return fired

    def _matches(self, alert: Alert, det: ActiveDetection) -> bool:
        if alert.symbol_ticker and alert.symbol_ticker != det.symbol_ticker:
            return False
        if alert.pattern_id and alert.pattern_id != det.pattern_id:
            return False
        cond = alert.condition or {}
        if cond.get("pattern") and cond["pattern"] != det.pattern_id:
            return False
        if cond.get("confidence_min") and det.confidence < cond["confidence_min"]:
            return False
        return True

    def _condition_met(self, cond: dict, det: ActiveDetection) -> bool:
        ctype = cond.get("type", "completion")
        if ctype == "first_detected":
            return True
        if ctype == "completion":
            value = cond.get("value", 0)
            return det.completion >= value if cond.get("op", ">=") == ">=" else det.completion > value
        if ctype == "breakout_confirmed":
            return det.detected_breakout or det.status == "CONFIRMED"
        if ctype == "invalidated":
            return det.status == "INVALIDATED"
        return False

    def _fire(self, db: Session, alert: Alert, det: ActiveDetection) -> None:
        alert.last_fired_at = datetime.now(timezone.utc)
        title = f"{det.symbol_ticker}: {det.pattern_name}"
        body = (
            f"{det.pattern_name} on {det.interval} — {det.completion:.0f}% complete, "
            f"{det.confidence:.0%} confidence, status {det.status}."
        )
        db.add(Notification(user_id=alert.user_id, alert_id=alert.id, title=title, body=body))
        bus.publish("alerts", {"event": "alert_fired", "alert_id": alert.id,
                               "symbol": det.symbol_ticker, "pattern": det.pattern_name,
                               "body": body})
        logger.info("alert_fired", symbol=det.symbol_ticker, pattern_id=det.pattern_id)
