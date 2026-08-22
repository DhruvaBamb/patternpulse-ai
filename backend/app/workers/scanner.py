"""Background scanner worker.

Runs the pattern scan across the configured universe × timeframes outside
HTTP request handlers. Uses APScheduler in-process (no Redis dependency);
the ``TaskQueue`` interface keeps a Celery adapter possible later.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from ..core.config import scanner_config
from ..core.logging import get_logger
from ..db.session import SessionLocal
from ..services.alerts import AlertEngine
from ..services.detection import DetectionService
from ..services.events import bus
from ..services.market_data import MarketDataService

logger = get_logger("scanner")


class TaskQueue:
    """Minimal interface so Celery/Redis can replace the in-process executor."""

    def submit(self, fn, *args) -> None:
        threading.Thread(target=fn, args=args, daemon=True).start()


class ScannerWorker:
    def __init__(self, detection: DetectionService, alerts: AlertEngine | None = None) -> None:
        self.detection = detection
        self.alerts = alerts
        self.queue = TaskQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_scan_at: datetime | None = None
        self.last_scan_summary: dict = {}
        self._scanner_lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="scanner", daemon=True)
        self._thread.start()
        logger.info("scanner_started")

    def stop(self) -> None:
        self._stop.set()

    # ── main loop ───────────────────────────────────────────────────────────
    def _run(self) -> None:
        interval = scanner_config().get("scan_interval_seconds", 300)
        # first scan shortly after boot
        self._stop.wait(5)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.scan_once()
            except Exception as exc:
                logger.error("scan_failed", error=str(exc))
            elapsed = time.time() - t0
            self._stop.wait(max(5.0, interval - elapsed))

    def scan_once(self) -> dict:
        if not self._scanner_lock.acquire(blocking=False):
            logger.info("scan_skipped", reason="already running")
            return {"skipped": True}
        try:
            cfg = scanner_config()
            universe = cfg.get("universe", [])
            timeframes = cfg.get("timeframes", ["1d"])
            lookback = cfg.get("lookback_bars", 250)
            summary = {"scanned": 0, "findings": 0, "errors": []}
            db = SessionLocal()
            try:
                for entry in universe:
                    ticker = entry["symbol"]
                    self.detection.market_data.upsert_symbol(
                        db, ticker, entry.get("name", ""), entry.get("market", "")
                    )
                    for tf in timeframes:
                        try:
                            results = self.detection.scan(db, ticker, tf, lookback)
                            summary["scanned"] += 1
                            summary["findings"] += sum(1 for r in results if r["detection"])
                        except Exception as exc:
                            summary["errors"].append(f"{ticker}:{tf}: {exc}")
                            logger.warning("scan_symbol_failed", symbol=ticker, interval=tf, error=str(exc))
                expired = self.detection.expire_stale(db)
                summary["expired"] = expired
                if self.alerts:
                    fired = self.alerts.evaluate(db)
                    summary["alerts_fired"] = fired
            finally:
                db.close()
            self.last_scan_at = datetime.now(timezone.utc)
            self.last_scan_summary = summary
            bus.publish("scanner", {"event": "scan_complete", **{k: v for k, v in summary.items() if k != "errors"},
                                    "n_errors": len(summary.get("errors", []))})
            logger.info("scan_complete", **{k: summary[k] for k in ("scanned", "findings") if k in summary})
            return summary
        finally:
            self._scanner_lock.release()

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "last_scan_at": self.last_scan_at.isoformat() if self.last_scan_at else None,
            "last_scan_summary": self.last_scan_summary,
        }
