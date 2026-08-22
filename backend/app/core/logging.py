"""Structured logging with request context. Never logs secrets."""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
symbol_ctx: ContextVar[str] = ContextVar("symbol", default="-")
pattern_ctx: ContextVar[str] = ContextVar("pattern", default="-")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False)
            if level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        cache_logger_on_first_use=True,
    )


def _inject_context(_, __, event_dict):
    event_dict.setdefault("request_id", request_id_ctx.get())
    event_dict.setdefault("symbol", symbol_ctx.get())
    event_dict.setdefault("pattern_id", pattern_ctx.get())
    return event_dict


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_ctx.set(rid)
    return rid


def get_logger(module: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(module)
