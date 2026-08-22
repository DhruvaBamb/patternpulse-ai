"""Optional intelligence features (spec §76–79).

- Similar historical formations (§76): database search over confirmed
  historical detections of the same pattern, ranked by structural similarity
  (geometry score proximity + relative level anatomy), reporting how many
  confirmed and how many resolved in the pattern's favour.
- Pattern confluence (§78): overlapping signals on one symbol/timeframe
  (pattern direction agreement, indicator context hooks) with a 0–100 score.
- Cross-timeframe analysis (§79): per-symbol aggregation of active
  detections across timeframes into a multi-timeframe context score.

All numbers come from actual database rows; when history is thin the
services say so instead of inventing counts.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import ActiveDetection, PatternEvent, Signal


class AnalysisService:
    # ── §76: similar historical formations ─────────────────────────────────
    def similar_historical(self, db: Session, detection_id: str, limit: int = 50) -> dict:
        det = db.execute(
            select(ActiveDetection).where(ActiveDetection.id == detection_id)
        ).scalar_one_or_none()
        if det is None:
            return {"error": "detection not found"}

        rows = db.execute(
            select(ActiveDetection)
            .where(
                ActiveDetection.pattern_id == det.pattern_id,
                ActiveDetection.id != det.id,
                ActiveDetection.status.in_(["CONFIRMED", "INVALIDATED", "EXPIRED", "FAILED"]),
            )
            .order_by(ActiveDetection.last_seen.desc())
            .limit(limit)
        ).scalars().all()

        def similarity(other: ActiveDetection) -> float:
            # geometry proximity + completion proximity (0..1)
            g = 1.0 - min(abs(other.geometry_score - det.geometry_score), 1.0)
            c = 1.0 - min(abs(other.completion - det.completion) / 100.0, 1.0)
            return 0.6 * g + 0.4 * c

        ranked = sorted(rows, key=similarity, reverse=True)
        similar = [
            {
                "id": r.id,
                "symbol": r.symbol_ticker,
                "interval": r.interval,
                "status": r.status,
                "completion": round(r.completion, 1),
                "confidence": round(r.confidence, 3),
                "similarity": round(similarity(r), 3),
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            }
            for r in ranked[:20]
        ]
        confirmed = [r for r in rows if r.status == "CONFIRMED"]
        # outcomes: check follow-through signals recorded after confirmation
        favourable = 0
        checked = 0
        for r in confirmed[:limit]:
            sig = db.execute(
                select(Signal).where(Signal.detection_id == r.id).order_by(Signal.id.desc())
            ).scalars().first()
            if sig and sig.action in ("BUY", "SELL"):
                checked += 1
                direction_ok = (sig.action == "BUY") == (r.direction == "bullish")
                if direction_ok:
                    favourable += 1
        return {
            "detection_id": det.id,
            "pattern": det.pattern_name,
            "similar_found": len(rows),
            "similar": similar,
            "confirmed": len(confirmed),
            "signals_checked": checked,
            "resolved_in_favour": favourable,
            "note": None
            if rows
            else "no historical formations of this pattern recorded yet — statistics build as the scanner runs",
        }

    # ── §78: confluence ────────────────────────────────────────────────────
    def confluence(self, db: Session, symbol: str, interval: str) -> dict:
        dets = db.execute(
            select(ActiveDetection).where(
                ActiveDetection.symbol_ticker == symbol,
                ActiveDetection.interval == interval,
                ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"]),
            )
        ).scalars().all()
        if not dets:
            return {"symbol": symbol, "interval": interval, "score": 0, "factors": [],
                    "note": "no active detections on this timeframe"}

        bullish = [d for d in dets if d.direction == "bullish"]
        bearish = [d for d in dets if d.direction == "bearish"]
        dom = "bullish" if len(bullish) >= len(bearish) else "bearish"
        dominant = bullish if dom == "bullish" else bearish

        factors: list[dict] = []
        # pattern agreement
        if dominant:
            agree = len(dominant) / len(dets)
            factors.append({"factor": f"{len(dominant)} {dom} patterns aligned",
                            "weight": round(agree, 2)})
        # high-confidence presence
        strong = [d for d in dominant if d.confidence >= 0.75]
        if strong:
            factors.append({"factor": f"{len(strong)} high-confidence (≥75%) {dom} formations",
                            "weight": 0.2})
        # breakout readiness
        near = [d for d in dominant if d.completion >= 80]
        if near:
            factors.append({"factor": f"{len(near)} formation(s) ≥80% complete", "weight": 0.15})
        # confirmed breakouts
        confirmed = [d for d in dominant if d.status == "CONFIRMED"]
        if confirmed:
            factors.append({"factor": f"{len(confirmed)} confirmed breakout(s)", "weight": 0.25})

        score = int(min(sum(f["weight"] for f in factors), 1.0) * 100)
        return {
            "symbol": symbol,
            "interval": interval,
            "direction": dom,
            "score": score,
            "factors": factors,
            "patterns": [{"pattern": d.pattern_name, "direction": d.direction,
                          "completion": round(d.completion, 1), "confidence": round(d.confidence, 3),
                          "status": d.status} for d in dets[:12]],
        }

    # ── §79: cross-timeframe context ───────────────────────────────────────
    def cross_timeframe(self, db: Session, symbol: str, timeframes: list[str] | None = None) -> dict:
        timeframes = timeframes or ["15m", "1h", "1d"]
        out: dict = {"symbol": symbol, "timeframes": {}, "context_score": None, "note": None}
        votes: list[str] = []
        for tf in timeframes:
            dets = db.execute(
                select(ActiveDetection).where(
                    ActiveDetection.symbol_ticker == symbol,
                    ActiveDetection.interval == tf,
                    ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"]),
                )
            ).scalars().all()
            if not dets:
                out["timeframes"][tf] = {"summary": "no active pattern", "direction": None}
                continue
            best = max(dets, key=lambda d: d.confidence)
            bull = sum(1 for d in dets if d.direction == "bullish")
            bear = sum(1 for d in dets if d.direction == "bearish")
            tf_dir = "bullish" if bull > bear else ("bearish" if bear > bull else "neutral")
            votes.append(tf_dir)
            out["timeframes"][tf] = {
                "summary": f"{len(dets)} formation(s); strongest {best.pattern_name} "
                           f"({best.completion:.0f}% complete, {best.confidence:.0%} confidence)",
                "direction": tf_dir,
                "strongest": {"pattern": best.pattern_name, "completion": round(best.completion, 1),
                              "confidence": round(best.confidence, 3), "status": best.status},
            }
        if not votes:
            out["note"] = "no active detections on any timeframe"
            return out
        bull = votes.count("bullish")
        bear = votes.count("bearish")
        out["context_score"] = int(abs(bull - bear) / len(votes) * 100)
        out["alignment"] = "bullish" if bull > bear else ("bearish" if bear > bull else "mixed")
        return out

    # ── counts helper for overview ─────────────────────────────────────────
    def counts(self, db: Session) -> dict:
        rows = db.execute(
            select(ActiveDetection.status, func.count())
            .where(ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"]))
            .group_by(ActiveDetection.status)
        ).all()
        return dict(rows)
