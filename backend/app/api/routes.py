"""API routes (spec §66) with typed responses and informative errors."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.datasets.providers.base import MarketDataError
from ml.inference.pipeline import quality_score
from ml.registry.registry import get_registry
from ..core.config import get_settings, scanner_config
from ..core.logging import get_logger, new_request_id
from ..db.models import (
    ActiveDetection,
    Alert,
    BacktestRun,
    Notification,
    PatternEvent,
    Signal,
    Symbol,
    WatchlistItem,
)
from ..db.session import get_db
from ..services.chatbot.assistant import PulseAssistant
from ..services.detection import ACTIVE_STATUSES, DetectionService
from ..services.events import bus
from ..services.market_data import MarketDataService

logger = get_logger("api")
router = APIRouter(prefix="/api")


# ── helpers ─────────────────────────────────────────────────────────────────

def get_market_data(request: Request) -> MarketDataService:
    return request.app.state.market_data


def get_detection(request: Request) -> DetectionService:
    return request.app.state.detection


def get_assistant(request: Request) -> PulseAssistant:
    return request.app.state.assistant


class ErrorOut(BaseModel):
    code: str
    message: str
    detail: str = ""


def api_error(status: int, code: str, message: str, detail: str = "") -> HTTPException:
    return HTTPException(status_code=status, detail=ErrorOut(code=code, message=message, detail=detail).model_dump())


# ── registry ────────────────────────────────────────────────────────────────

@router.get("/registry/patterns")
def list_patterns(category: str | None = None, implemented_only: bool = False):
    reg = get_registry()
    defs = reg.by_category(category) if category else reg.all()
    return {
        "registry_version": reg.version,
        "total": len(defs),
        "patterns": [d.to_dict() for d in defs if (d.implemented or not implemented_only)],
    }


@router.get("/registry/patterns/{pattern_id}")
def pattern_detail(pattern_id: str):
    reg = get_registry()
    try:
        return reg.get(pattern_id).to_dict()
    except KeyError:
        raise api_error(404, "unknown_pattern", f"pattern '{pattern_id}' not in registry")


# ── markets / symbols ──────────────────────────────────────────────────────

@router.get("/markets")
def markets(db: Session = Depends(get_db)):
    universe = scanner_config().get("universe", [])
    symbols = {s.ticker: s for s in db.execute(select(Symbol)).scalars().all()}
    out = []
    for entry in universe:
        s = symbols.get(entry["symbol"])
        out.append({
            "symbol": entry["symbol"], "name": entry.get("name", ""), "market": entry.get("market", ""),
            "last_price": s.last_price if s else None,
            "change_pct": s.last_change_pct if s else None,
            "updated_at": s.updated_at.isoformat() if s and s.updated_at else None,
        })
    return {"symbols": out}


@router.get("/symbols/{symbol}")
def symbol_detail(symbol: str, db: Session = Depends(get_db), md: MarketDataService = Depends(get_market_data)):
    try:
        quote = md.refresh_quote(db, symbol)
    except MarketDataError as exc:
        raise api_error(503, "market_data_unavailable", "market data unavailable", str(exc))
    dets = db.execute(
        select(ActiveDetection).where(ActiveDetection.symbol_ticker == symbol,
                                      ActiveDetection.status.in_(ACTIVE_STATUSES))
    ).scalars().all()
    return {
        "symbol": symbol, **quote,
        "active_patterns": [
            {"pattern": d.pattern_name, "status": d.status, "completion": d.completion,
             "confidence": d.confidence, "interval": d.interval}
            for d in dets
        ],
    }


@router.get("/symbols/{symbol}/candles")
def candles(
    symbol: str,
    interval: str = Query("1d"),
    lookback: int = Query(250, ge=20, le=1000),
    db: Session = Depends(get_db),
    md: MarketDataService = Depends(get_market_data),
):
    try:
        df = md.get_candles(db, symbol, interval, lookback)
    except MarketDataError as exc:
        raise api_error(503, "market_data_unavailable", "could not load candles", str(exc))
    if len(df) < 30:
        raise api_error(422, "insufficient_candles", f"only {len(df)} candles available (need >= 30)")
    return {
        "symbol": symbol, "interval": interval,
        "candles": [
            {"ts": str(ts), "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"]), "volume": float(r.get("volume") or 0)}
            for ts, r in df.iterrows()
        ],
    }


# ── patterns / radar ────────────────────────────────────────────────────────

@router.get("/patterns/active")
def active_patterns(
    category: str | None = None, direction: str | None = None, timeframe: str | None = None,
    min_confidence: float = 0.0, min_completion: float = 0.0,
    db: Session = Depends(get_db),
):
    q = select(ActiveDetection).where(ActiveDetection.status.in_(ACTIVE_STATUSES))
    dets = db.execute(q).scalars().all()
    out = []
    for d in dets:
        if d.confidence < min_confidence or d.completion < min_completion:
            continue
        if category and d.category != category:
            continue
        if direction and d.direction != direction:
            continue
        if timeframe and d.interval != timeframe:
            continue
        out.append(_det_public(d))
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return {"count": len(out), "patterns": out}


@router.get("/patterns/{detection_id}")
def pattern_detail_by_id(detection_id: str, db: Session = Depends(get_db)):
    det = db.execute(
        select(ActiveDetection).where(ActiveDetection.id == detection_id)
    ).scalar_one_or_none()
    if det is None:
        raise api_error(404, "not_found", f"detection '{detection_id}' not found")
    events = db.execute(
        select(PatternEvent).where(PatternEvent.detection_id == detection_id).order_by(PatternEvent.id)
    ).scalars().all()
    sig = db.execute(
        select(Signal).where(Signal.detection_id == detection_id).order_by(Signal.id.desc())
    ).scalars().first()
    out = _det_public(det)
    out["events"] = [{"event": e.event, "detail": e.detail, "at": e.created_at.isoformat()} for e in events]
    out["signal"] = sig.payload if sig else None
    return out


@router.get("/radar")
def radar(
    sort: str = Query("confidence", pattern="^(confidence|completion|pattern|symbol|category|timeframe|direction)$"),
    db: Session = Depends(get_db),
):
    dets = db.execute(
        select(ActiveDetection).where(ActiveDetection.status.in_(ACTIVE_STATUSES))
    ).scalars().all()
    rows = [_det_public(d) for d in dets]
    keymap = {"timeframe": "interval"}
    k = keymap.get(sort, sort)
    rows.sort(key=lambda r: (r.get(k) is None, r.get(k) if not isinstance(r.get(k), str) else -1)
              if k not in ("pattern", "symbol", "category", "interval", "direction")
              else str(r.get(k, "")), reverse=(k in ("confidence", "completion")))
    return {"rows": rows, "scanner_status": _scanner_status_public(None)}


@router.get("/symbols/{symbol}/patterns")
def symbol_patterns(
    symbol: str, interval: str = Query("1d"), fresh: bool = Query(False),
    db: Session = Depends(get_db), det_svc: DetectionService = Depends(get_detection),
):
    if fresh:
        try:
            det_svc.scan(db, symbol, interval)
        except MarketDataError as exc:
            raise api_error(503, "market_data_unavailable", "scan failed", str(exc))
    dets = db.execute(
        select(ActiveDetection).where(
            ActiveDetection.symbol_ticker == symbol, ActiveDetection.interval == interval,
            ActiveDetection.status.in_(ACTIVE_STATUSES),
        )
    ).scalars().all()
    sigs = db.execute(
        select(Signal).where(Signal.symbol_ticker == symbol, Signal.interval == interval)
        .order_by(Signal.id.desc()).limit(1 + len(dets))
    ).scalars().all()
    latest_signal = sigs[0].payload if sigs else None
    return {
        "symbol": symbol, "interval": interval,
        "patterns": [_det_public(d) for d in dets],
        "signal": latest_signal,
        "cv_status": det_svc.pipeline.detector.status(),
    }


def _det_public(d: ActiveDetection) -> dict:
    return {
        "id": d.id, "symbol": d.symbol_ticker, "interval": d.interval,
        "pattern_id": d.pattern_id, "pattern": d.pattern_name, "category": d.category,
        "direction": d.direction, "status": d.status,
        "completion": round(d.completion, 1), "confidence": round(d.confidence, 3),
        "geometry_score": round(d.geometry_score, 3),
        "visual_score": d.visual_score, "source": d.source, "stage": d.stage,
        "detected_breakout": d.detected_breakout,
        "levels": d.levels, "keypoints": d.keypoints, "lines": d.lines,
        "violations": d.violations,
        "completion_history": d.completion_history[-30:] if d.completion_history else [],
        "confidence_history": d.confidence_history[-30:] if d.confidence_history else [],
        "first_seen": d.first_seen.isoformat() if d.first_seen else None,
        "last_seen": d.last_seen.isoformat() if d.last_seen else None,
    }


def _scanner_status_public(worker) -> dict:
    if worker is None:
        return {"running": False}
    return worker.status()


@router.get("/heatmap")
def heatmap(db: Session = Depends(get_db)):
    universe = scanner_config().get("universe", [])
    timeframes = scanner_config().get("timeframes", ["1d"])
    dets = db.execute(
        select(ActiveDetection).where(ActiveDetection.status.in_(ACTIVE_STATUSES))
    ).scalars().all()
    cell: dict[tuple[str, str], dict] = {}
    for d in dets:
        key = (d.symbol_ticker, d.interval)
        cur = cell.get(key)
        if cur is None or d.confidence > cur["confidence"]:
            cell[key] = {
                "pattern": d.pattern_name, "direction": d.direction,
                "confidence": round(d.confidence, 2), "completion": round(d.completion, 1),
                "last_seen": d.last_seen.isoformat() if d.last_seen else None,
            }
    return {
        "timeframes": timeframes,
        "symbols": [u["symbol"] for u in universe],
        "cells": {f"{s}|{tf}": v for (s, tf), v in cell.items()},
    }


# ── backtests ───────────────────────────────────────────────────────────────

@router.get("/backtests")
def backtests(db: Session = Depends(get_db)):
    runs = db.execute(select(BacktestRun).order_by(BacktestRun.created_at.desc())).scalars().all()
    return {"runs": [
        {"id": r.id, "status": r.status, "params": r.params,
         "overall": (r.results or {}).get("overall"),
         "pattern_stats": r.pattern_stats,
         "created_at": r.created_at.isoformat()} for r in runs
    ]}


@router.post("/backtests/run")
async def run_backtest(request: Request, db: Session = Depends(get_db)):
    """Launch a background backtest run (long operation => queued)."""
    body = await request.json() if request.headers.get("content-length") else {}
    symbols = body.get("symbols") or [u["symbol"] for u in scanner_config().get("universe", [])][:6]
    interval = body.get("interval", "1d")

    async def _run():
        from ml.backtesting.engine import BacktestConfig, BacktestEngine
        from ml.inference.pipeline import DetectionPipeline
        from ..db.session import SessionLocal

        run = BacktestRun(params={"symbols": symbols, "interval": interval}, status="running")
        s = SessionLocal()
        try:
            s.add(run)
            s.commit()
            from ml.datasets.providers.yfinance_provider import build_provider
            from ml.datasets.validation import validate_ohlcv

            provider = build_provider()
            engine = BacktestEngine(DetectionPipeline(), BacktestConfig())
            trades = []
            for sym in symbols:
                try:
                    df, rep = validate_ohlcv(provider.get_history(sym, interval, 1200), sym)
                    if rep.fatal:
                        continue
                    trades.extend(engine.run(df, sym, interval))
                except Exception as exc:
                    logger.warning("backtest_symbol_failed", symbol=sym, error=str(exc))
            from ml.backtesting.engine import BacktestResult

            res = BacktestResult(trades=trades)
            run.results = {"overall": res.overall()}
            run.pattern_stats = res.summary_by_pattern()
            run.status = "completed"
            s.commit()
        except Exception as exc:
            run.status = "failed"
            run.results = {"error": str(exc)}
            s.commit()
        finally:
            s.close()

    request.app.state.task_queue.submit(lambda: asyncio.run(_run()))
    return {"status": "queued"}


# ── chat ────────────────────────────────────────────────────────────────────

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


@router.post("/chat")
async def chat(body: ChatIn, db: Session = Depends(get_db), assistant: PulseAssistant = Depends(get_assistant)):
    session_id = assistant.ensure_session(db, body.session_id)

    async def gen():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        try:
            async for delta in assistant.stream_reply(db, session_id, body.message):
                yield f"data: {json.dumps({'type': 'delta', 'text': delta})}\n\n"
        except Exception as exc:
            logger.error("chat_failed", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': 'assistant error, try again'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/chat/status")
def chat_status(assistant: PulseAssistant = Depends(get_assistant)):
    return assistant.status()


# ── models / training ──────────────────────────────────────────────────────

@router.get("/models")
def models():
    det_status = get_registry()
    from pathlib import Path
    runs_root = Path("artifacts/runs")
    runs = []
    if runs_root.exists():
        for run_dir in sorted(runs_root.iterdir()):
            meta = run_dir / "run_meta.json"
            hist = run_dir / "metrics" / "history.json"
            runs.append({
                "run": run_dir.name,
                "has_meta": meta.exists(),
                "history_epochs": len(json.loads(hist.read_text())) if hist.exists() else 0,
            })
    return {"registry_version": det_status.version, "runs": runs}


@router.get("/training/status")
def training_status(request: Request, db: Session = Depends(get_db)):
    worker = getattr(request.app.state, "training_monitor", None)
    out = {"status": "unknown", "note": "training monitor not attached"}
    if worker:
        out = worker
    return out


# ── watchlist / alerts / notifications ─────────────────────────────────────

class WatchIn(BaseModel):
    symbol: str


@router.get("/watchlist")
def watchlist(db: Session = Depends(get_db)):
    items = db.execute(select(WatchlistItem).where(WatchlistItem.user_id == "local")).scalars().all()
    out = []
    for it in items:
        sym = db.execute(select(Symbol).where(Symbol.ticker == it.symbol_ticker)).scalar_one_or_none()
        det = db.execute(
            select(ActiveDetection).where(
                ActiveDetection.symbol_ticker == it.symbol_ticker,
                ActiveDetection.status.in_(ACTIVE_STATUSES),
            ).order_by(ActiveDetection.confidence.desc())
        ).scalars().first()
        out.append({
            "symbol": it.symbol_ticker,
            "last_price": sym.last_price if sym else None,
            "change_pct": sym.last_change_pct if sym else None,
            "pattern": det.pattern_name if det else None,
            "completion": round(det.completion, 1) if det else None,
            "confidence": round(det.confidence, 3) if det else None,
            "interval": det.interval if det else None,
            "updated_at": sym.updated_at.isoformat() if sym and sym.updated_at else None,
        })
    return {"items": out}


@router.post("/watchlist")
def add_watch(body: WatchIn, db: Session = Depends(get_db)):
    existing = db.execute(
        select(WatchlistItem).where(WatchlistItem.user_id == "local", WatchlistItem.symbol_ticker == body.symbol)
    ).scalar_one_or_none()
    if existing:
        return {"status": "already_watching"}
    db.add(WatchlistItem(user_id="local", symbol_ticker=body.symbol))
    db.commit()
    return {"status": "watching"}


@router.delete("/watchlist/{symbol}")
def remove_watch(symbol: str, db: Session = Depends(get_db)):
    item = db.execute(
        select(WatchlistItem).where(WatchlistItem.user_id == "local", WatchlistItem.symbol_ticker == symbol)
    ).scalar_one_or_none()
    if item:
        db.delete(item)
        db.commit()
    return {"status": "removed"}


class AlertIn(BaseModel):
    symbol: str | None = None
    pattern_id: str | None = None
    condition: dict


@router.get("/alerts")
def list_alerts(db: Session = Depends(get_db)):
    alerts = db.execute(select(Alert).where(Alert.user_id == "local")).scalars().all()
    return {"alerts": [
        {"id": a.id, "symbol": a.symbol_ticker, "pattern_id": a.pattern_id,
         "condition": a.condition, "enabled": a.enabled,
         "last_fired_at": a.last_fired_at.isoformat() if a.last_fired_at else None}
        for a in alerts
    ]}


@router.post("/alerts")
def create_alert(body: AlertIn, db: Session = Depends(get_db)):
    a = Alert(user_id="local", symbol_ticker=body.symbol, pattern_id=body.pattern_id, condition=body.condition)
    db.add(a)
    db.commit()
    return {"id": a.id, "status": "created"}


@router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    a = db.get(Alert, alert_id)
    if a:
        db.delete(a)
        db.commit()
    return {"status": "deleted"}


@router.get("/notifications")
def notifications(only_unread: bool = True, db: Session = Depends(get_db)):
    q = select(Notification).where(Notification.user_id == "local")
    if only_unread:
        q = q.where(Notification.read == False)  # noqa: E712
    notes = db.execute(q.order_by(Notification.id.desc()).limit(50)).scalars().all()
    return {"notifications": [
        {"id": n.id, "title": n.title, "body": n.body,
         "created_at": n.created_at.isoformat()} for n in notes
    ]}


@router.post("/notifications/read")
def mark_read(db: Session = Depends(get_db)):
    for n in db.execute(select(Notification).where(Notification.read == False)).scalars().all():  # noqa: E712
        n.read = True
    db.commit()
    return {"status": "ok"}


# ── intelligence: similar / confluence / cross-timeframe / drift ────────────

def get_analysis(request: Request):
    return request.app.state.analysis


@router.get("/patterns/{detection_id}/similar")
def similar_historical(detection_id: str, db: Session = Depends(get_db), analysis=Depends(get_analysis)):
    return analysis.similar_historical(db, detection_id)


@router.get("/symbols/{symbol}/confluence")
def confluence(symbol: str, interval: str = Query("1d"), db: Session = Depends(get_db), analysis=Depends(get_analysis)):
    return analysis.confluence(db, symbol, interval)


@router.get("/symbols/{symbol}/timeframes")
def cross_timeframe(symbol: str, db: Session = Depends(get_db), analysis=Depends(get_analysis)):
    from ..core.config import scanner_config

    tfs = scanner_config().get("timeframes", ["1d"])
    return analysis.cross_timeframe(db, symbol, tfs)


@router.get("/drift/status")
def drift_status(request: Request):
    return getattr(request.app.state, "drift_status", {"configured": False,
                                                        "note": "drift reference not built yet"})


# ── SSE stream ──────────────────────────────────────────────────────────────

@router.get("/stream")
async def stream(request: Request):
    async def gen():
        q = bus.subscribe("detections")
        alerts_q = bus.subscribe("alerts")
        scanner_q = bus.subscribe("scanner")
        try:
            yield f"data: {json.dumps({'channel': 'hello'})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                for queue in (q, alerts_q, scanner_q):
                    try:
                        ev = queue.get_nowait()
                        yield f"data: {ev.encode()}\n\n"
                    except asyncio.QueueEmpty:
                        pass
                await asyncio.sleep(0.5)
        finally:
            bus.unsubscribe("detections", q)
            bus.unsubscribe("alerts", alerts_q)
            bus.unsubscribe("scanner", scanner_q)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── system ──────────────────────────────────────────────────────────────────

@router.get("/health")
def health(request: Request):
    settings = get_settings()
    worker = getattr(request.app.state, "scanner", None)
    det = getattr(request.app.state, "detection", None)
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "scanner": worker.status() if worker else {"running": False},
        "cv_detector": det.pipeline.detector.status() if det else {"available": False},
        "chatbot": request.app.state.assistant.status() if hasattr(request.app.state, "assistant") else {},
    }
