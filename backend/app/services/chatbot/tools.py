"""PulseAI internal tools — the only way the chatbot sees platform data.

Every tool queries real database state; when data does not exist the tool
says so explicitly so the assistant never invents market data.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.registry.registry import get_registry
from ...db.models import ActiveDetection, BacktestRun, Signal

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_snapshot",
            "description": "Latest tracked price/change and active pattern count for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Ticker, e.g. RELIANCE.NS or AAPL"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_patterns",
            "description": "Currently tracked (forming/confirmed) patterns for a symbol, optionally filtered by timeframe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": ["string", "null"], "description": "Optional timeframe: 15m, 1h, 1d ..."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_patterns",
            "description": "Scan current detections across all symbols with filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_confidence": {"type": ["number", "null"], "description": "0-1"},
                    "min_completion": {"type": ["number", "null"], "description": "0-100"},
                    "pattern": {"type": ["string", "null"], "description": "pattern id like cup_and_handle"},
                    "direction": {"type": ["string", "null"], "enum": ["bullish", "bearish", "neutral", None]},
                    "timeframe": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detection_details",
            "description": "Full detail for one tracked formation by its ID (levels, history, violations).",
            "parameters": {"type": "object", "properties": {"pattern_id_full": {"type": "string"}}, "required": ["pattern_id_full"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_signal_context",
            "description": "The most recent BUY/SELL/HOLD/WATCH signal and its reasoning for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}, "interval": {"type": ["string", "null"]}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_symbols",
            "description": "Compare active patterns between two or more symbols.",
            "parameters": {
                "type": "object",
                "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
                "required": ["symbols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_supported_patterns",
            "description": "List patterns the platform knows, with category/direction and implementation status.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": ["string", "null"], "enum": ["classical", "candlestick", "harmonic", None]}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_pattern",
            "description": "Educational explanation of a pattern: geometry, confirmation, invalidation.",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_backtest_results",
            "description": "Historical performance statistics produced by the backtesting engine (only if a backtest has run).",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": ["string", "null"], "description": "optional pattern filter"}},
            },
        },
    },
]


def _resolve_symbol(db: Session, symbol: str) -> str | None:
    """Match a user-friendly name or partial ticker to a tracked symbol."""
    from ...db.models import Symbol

    symbols = db.execute(select(Symbol)).scalars().all()
    low = symbol.strip().lower()
    exact = [s for s in symbols if s.ticker.lower() == low]
    if exact:
        return exact[0].ticker
    name_match = [s for s in symbols if low in (s.name or "").lower() or low in s.ticker.lower()]
    if name_match:
        return name_match[0].ticker
    return None


def _det_json(det: ActiveDetection) -> dict:
    return {
        "id": det.id,
        "symbol": det.symbol_ticker,
        "interval": det.interval,
        "pattern": det.pattern_name,
        "pattern_id": det.pattern_id,
        "direction": det.direction,
        "status": det.status,
        "completion_pct": round(det.completion, 1),
        "confidence_pct": round(det.confidence * 100, 1),
        "geometry_score_pct": round(det.geometry_score * 100, 1),
        "source": det.source,
        "stage": det.stage,
        "breakout_detected": det.detected_breakout,
        "levels": det.levels,
        "last_seen": det.last_seen.isoformat() if det.last_seen else None,
    }


def build_tool_map(db: Session) -> dict[str, Callable[[dict], str]]:
    def get_stock_snapshot(args: dict) -> str:
        sym = _resolve_symbol(db, args["symbol"])
        if not sym:
            return json.dumps({"error": f"symbol '{args['symbol']}' is not tracked on this platform"})
        from ...db.models import Symbol

        s = db.execute(select(Symbol).where(Symbol.ticker == sym)).scalar_one_or_none()
        dets = db.execute(
            select(ActiveDetection).where(
                ActiveDetection.symbol_ticker == sym,
                ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"]),
            )
        ).scalars().all()
        if s is None:
            return json.dumps({"error": "symbol record missing"})
        return json.dumps({
            "symbol": sym, "name": s.name, "market": s.market,
            "last_price": s.last_price, "change_pct": s.last_change_pct,
            "active_patterns": len(dets),
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "note": None if s.last_price else "price not yet scanned",
        })

    def get_active_patterns(args: dict) -> str:
        sym = _resolve_symbol(db, args["symbol"])
        if not sym:
            return json.dumps({"error": f"symbol '{args['symbol']}' is not tracked"})
        q = select(ActiveDetection).where(
            ActiveDetection.symbol_ticker == sym,
            ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"]),
        )
        if args.get("interval"):
            q = q.where(ActiveDetection.interval == args["interval"])
        dets = db.execute(q).scalars().all()
        if not dets:
            return json.dumps({"symbol": sym, "patterns": [], "note": "no active patterns tracked right now"})
        return json.dumps({"symbol": sym, "patterns": [_det_json(d) for d in dets]})

    def scan_patterns(args: dict) -> str:
        q = select(ActiveDetection).where(
            ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"])
        )
        dets = db.execute(q).scalars().all()
        if args.get("min_confidence") is not None:
            dets = [d for d in dets if d.confidence >= args["min_confidence"]]
        if args.get("min_completion") is not None:
            dets = [d for d in dets if d.completion >= args["min_completion"]]
        if args.get("pattern"):
            dets = [d for d in dets if d.pattern_id == args["pattern"] or d.pattern_name.lower() == args["pattern"].lower()]
        if args.get("direction"):
            dets = [d for d in dets if d.direction == args["direction"]]
        if args.get("timeframe"):
            dets = [d for d in dets if d.interval == args["timeframe"]]
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)[: args.get("limit", 20)]
        if not dets:
            return json.dumps({"results": [], "note": "no detections match these filters"})
        return json.dumps({"results": [_det_json(d) for d in dets]})

    def get_detection_details(args: dict) -> str:
        det = db.execute(
            select(ActiveDetection).where(ActiveDetection.id == args["pattern_id_full"])
        ).scalar_one_or_none()
        if det is None:
            return json.dumps({"error": f"detection '{args['pattern_id_full']}' not found"})
        out = _det_json(det)
        out["completion_history"] = det.completion_history[-10:]
        out["confidence_history"] = det.confidence_history[-10:]
        out["violations"] = det.violations
        out["keypoints"] = det.keypoints
        return json.dumps(out)

    def get_signal_context(args: dict) -> str:
        sym = _resolve_symbol(db, args["symbol"])
        if not sym:
            return json.dumps({"error": f"symbol '{args['symbol']}' is not tracked"})
        q = select(Signal).where(Signal.symbol_ticker == sym)
        if args.get("interval"):
            q = q.where(Signal.interval == args["interval"])
        sig = db.execute(q.order_by(Signal.created_at.desc())).scalars().first()
        if sig is None:
            return json.dumps({"symbol": sym, "note": "no signals generated yet — scanner may not have run"})
        return json.dumps({"symbol": sym, **sig.payload})

    def compare_symbols(args: dict) -> str:
        out = {}
        for raw in args["symbols"][:4]:
            sym = _resolve_symbol(db, raw)
            if not sym:
                out[raw] = {"error": "not tracked"}
                continue
            dets = db.execute(
                select(ActiveDetection).where(
                    ActiveDetection.symbol_ticker == sym,
                    ActiveDetection.status.in_(["EARLY", "FORMING", "NEAR_COMPLETION", "CONFIRMED"]),
                )
            ).scalars().all()
            out[sym] = [_det_json(d) for d in dets] or "no active patterns"
        return json.dumps(out)

    def get_supported_patterns(args: dict) -> str:
        reg = get_registry()
        defs = reg.by_category(args["category"]) if args.get("category") else reg.all()
        return json.dumps({
            "registry_version": reg.version,
            "total": len(defs),
            "patterns": [
                {"id": d.id, "name": d.name, "category": d.category, "direction": d.direction,
                 "implemented": d.implemented}
                for d in defs
            ],
        })

    def explain_pattern(args: dict) -> str:
        reg = get_registry()
        d = reg.get_by_name(args["pattern"]) or (reg.get(args["pattern"]) if args["pattern"] in reg._by_id else None)
        if d is None:
            return json.dumps({"error": f"unknown pattern '{args['pattern']}'"})
        return json.dumps({
            "id": d.id, "name": d.name, "category": d.category, "direction": d.direction,
            "description": d.description,
            "confirmation_rules": d.confirmation_rules,
            "invalidation_rules": d.invalidation_rules,
            "formation_stages": d.stages,
            "commonly_confused_with": d.confused_with,
        })

    def get_backtest_results(args: dict) -> str:
        run = db.execute(select(BacktestRun).order_by(BacktestRun.created_at.desc())).scalars().first()
        if run is None or run.status != "completed" or not run.pattern_stats:
            return json.dumps({"note": "no completed backtest yet — historical statistics unavailable"})
        stats = run.pattern_stats
        if args.get("pattern"):
            key = next((k for k in stats if k == args["pattern"] or k.lower() in args["pattern"].lower()), None)
            stats = {key: stats[key]} if key else {}
        return json.dumps({"run_id": run.id, "created_at": run.created_at.isoformat(), "pattern_stats": stats})

    return {
        "get_stock_snapshot": get_stock_snapshot,
        "get_active_patterns": get_active_patterns,
        "scan_patterns": scan_patterns,
        "get_detection_details": get_detection_details,
        "get_signal_context": get_signal_context,
        "compare_symbols": compare_symbols,
        "get_supported_patterns": get_supported_patterns,
        "explain_pattern": explain_pattern,
        "get_backtest_results": get_backtest_results,
    }
