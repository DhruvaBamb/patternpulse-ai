"""Database models (spec §62)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def uid() -> str:
    return uuid.uuid4().hex[:16]


class Symbol(Base):
    __tablename__ = "symbols"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    market: Mapped[str] = mapped_column(String(32), default="")
    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    detections: Mapped[list["ActiveDetection"]] = relationship(back_populates="symbol")


class Candle(Base):
    __tablename__ = "candles"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_ticker: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0)
    __table_args__ = (UniqueConstraint("symbol_ticker", "interval", "ts", name="uq_candle"),)


class ActiveDetection(Base):
    """A tracked formation with a stable lifecycle (spec §22)."""

    __tablename__ = "active_detections"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    symbol_ticker: Mapped[str] = mapped_column(String(32), ForeignKey("symbols.ticker"), index=True)
    interval: Mapped[str] = mapped_column(String(8), index=True)
    pattern_id: Mapped[str] = mapped_column(String(48), index=True)
    pattern_name: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(12))

    status: Mapped[str] = mapped_column(String(20), default="FORMING", index=True)
    completion: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    geometry_score: Mapped[float] = mapped_column(Float, default=0.0)
    visual_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="geometry_only")

    completion_history: Mapped[list] = mapped_column(JSON, default=list)
    confidence_history: Mapped[list] = mapped_column(JSON, default=list)

    levels: Mapped[dict] = mapped_column(JSON, default=dict)
    keypoints: Mapped[dict] = mapped_column(JSON, default=dict)
    lines: Mapped[dict] = mapped_column(JSON, default=dict)
    violations: Mapped[list] = mapped_column(JSON, default=list)
    stage: Mapped[str] = mapped_column(String(48), default="")
    detected_breakout: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    model_version: Mapped[str] = mapped_column(String(48), default="")
    pattern_registry_version: Mapped[str] = mapped_column(String(24), default="")

    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    symbol: Mapped[Symbol] = relationship(back_populates="detections")
    events: Mapped[list["PatternEvent"]] = relationship(back_populates="detection")
    signals: Mapped[list["Signal"]] = relationship(back_populates="detection")

    __table_args__ = (
        UniqueConstraint("symbol_ticker", "interval", "pattern_id", "status", name="uq_formation"),
    )


class PatternEvent(Base):
    __tablename__ = "pattern_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    detection_id: Mapped[str] = mapped_column(String(48), ForeignKey("active_detections.id"), index=True)
    event: Mapped[str] = mapped_column(String(32))          # first_seen | stage | breakout | confirmed | invalidated | expired
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    detection: Mapped[ActiveDetection] = relationship(back_populates="events")


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(primary_key=True)
    detection_id: Mapped[str | None] = mapped_column(String(48), ForeignKey("active_detections.id"), nullable=True, index=True)
    symbol_ticker: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(8))
    action: Mapped[str] = mapped_column(String(8))
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    detection: Mapped[ActiveDetection | None] = relationship(back_populates="signals")


class WatchlistItem(Base):
    __tablename__ = "watchlists"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(48), default="local", index=True)
    symbol_ticker: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("user_id", "symbol_ticker", name="uq_watch"),)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(48), default="local", index=True)
    symbol_ticker: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pattern_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    condition: Mapped[dict] = mapped_column(JSON, default=dict)  # {"type": "completion", "op": ">", "value": 70, "confidence_min": 0.85}
    channel: Mapped[str] = mapped_column(String(16), default="in_app")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(48), default="local", index=True)
    alert_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("alerts.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(128))
    body: Mapped[str] = mapped_column(Text, default="")
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class BacktestRun(Base):
    __tablename__ = "backtests"
    id: Mapped[int] = mapped_column(String(48), primary_key=True, default=uid)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    results: Mapped[dict] = mapped_column(JSON, default=dict)
    pattern_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(48), unique=True)
    run_name: Mapped[str] = mapped_column(String(64), default="")
    dataset_version: Mapped[str] = mapped_column(String(64), default="")
    pattern_registry_version: Mapped[str] = mapped_column(String(24), default="")
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    weights_path: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="idle")
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    best_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_epoch: Mapped[int] = mapped_column(Integer, default=0)
    latest_checkpoint: Mapped[str] = mapped_column(String(255), default="")
    history: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(String(48), default="local", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(48), ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(12))
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
