"""SQLAlchemy engine/session management (SQLite dev default, PostgreSQL ready)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ..core.config import get_settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        if ":memory:" in url:
            # single shared connection so all threads see the same in-memory DB
            from sqlalchemy.pool import StaticPool

            return {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
        path = url.split("///")[-1]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


settings = get_settings()
engine = create_engine(settings.database_url, **_engine_kwargs(settings.database_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    from . import models  # noqa: F401  (register models)

    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
