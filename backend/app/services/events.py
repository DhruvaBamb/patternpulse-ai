"""In-process event bus for SSE fan-out (detection updates, alerts)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Event:
    channel: str
    payload: dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def encode(self) -> str:
        return json.dumps({"channel": self.channel, "ts": self.ts, "data": self.payload})


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.setdefault(channel, set()).add(q)
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        self._subscribers.get(channel, set()).discard(q)

    def publish(self, channel: str, payload: dict[str, Any]) -> None:
        ev = Event(channel, payload)
        dead = []
        for q in self._subscribers.get(channel, set()):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(channel, q)


bus = EventBus()
