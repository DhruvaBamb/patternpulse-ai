"""PulseAI — Groq-powered assistant with tool access and session memory."""

from __future__ import annotations

import json
from typing import AsyncIterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.config import get_settings
from ...core.logging import get_logger
from ...db.models import ChatMessage, ChatSession
from .tools import TOOL_SCHEMAS, build_tool_map

logger = get_logger("pulseai")

SYSTEM_PROMPT = """You are PulseAI, the assistant inside the PatternPulse AI market-pattern platform.

Rules:
1. You answer questions about THIS platform's live state using the provided tools.
   Never invent market data, detections, statistics or prices. If a tool reports
   that data does not exist, say so plainly.
2. Explain detections with the evidence the platform tracks: completion %,
   fused confidence, geometry score, breakout status and levels.
3. Frame everything as research-oriented technical analysis, never guaranteed
   predictions. Do not say a trade will make money.
4. Be concise and quantitative. Use the tool results verbatim for numbers.
5. Pattern education questions can use explain_pattern / get_supported_patterns.
"""

CHATBOT_UNAVAILABLE_MESSAGE = (
    "PulseAI is not configured: no GROQ_API_KEY found. Add one to .env "
    "(get a free key at https://console.groq.com) and restart the backend."
)


class ChatbotUnavailable(RuntimeError):
    pass


class PulseAssistant:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        if self.settings.groq_api_key:
            try:
                from groq import AsyncGroq

                self._client = AsyncGroq(api_key=self.settings.groq_api_key)
            except Exception as exc:
                logger.error("groq_init_failed", error=str(exc))

    @property
    def available(self) -> bool:
        return self._client is not None

    def status(self) -> dict:
        return {
            "available": self.available,
            "model": self.settings.groq_model if self.available else None,
            "reason": None if self.available else "GROQ_API_KEY not set",
        }

    def _history(self, db: Session, session_id: str) -> list[dict]:
        msgs = db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
        ).scalars().all()
        return [{"role": m.role, "content": m.content} for m in msgs][-20:]

    async def stream_reply(
        self, db: Session, session_id: str, user_message: str
    ) -> AsyncIterator[str]:
        """Yield text deltas; runs the tool loop, persists the exchange."""
        if not self.available:
            yield CHATBOT_UNAVAILABLE_MESSAGE
            return

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self._history(db, session_id)
        messages.append({"role": "user", "content": user_message})

        db.add(ChatMessage(session_id=session_id, role="user", content=user_message))
        db.commit()

        tools = build_tool_map(db)
        final_text_parts: list[str] = []
        tool_calls_used: list[dict] = []

        for _turn in range(4):  # bounded tool loop
            resp = await self._client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                stream=True,
            )
            tool_calls: list[dict] = []
            content_parts: list[str] = []
            async for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content_parts.append(delta.content)
                    yield delta.content
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls[tc.index]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls[tc.index]["arguments"] += tc.function.arguments

            assistant_content = "".join(content_parts)
            final_text_parts.append(assistant_content)

            if not tool_calls:
                break

            messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn = tools.get(tc["name"])
                result = fn(args) if fn else json.dumps({"error": f"unknown tool {tc['name']}"})
                tool_calls_used.append({"tool": tc["name"], "args": args})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        final = "".join(final_text_parts) or "(no response)"
        db.add(ChatMessage(session_id=session_id, role="assistant", content=final,
                           tool_calls=tool_calls_used))
        db.commit()

    def ensure_session(self, db: Session, session_id: str | None, user_id: str = "local") -> str:
        if session_id:
            exists = db.execute(select(ChatSession).where(ChatSession.id == session_id)).scalar_one_or_none()
            if exists:
                return session_id
        s = ChatSession(id=session_id or None, user_id=user_id)
        db.add(s)
        db.commit()
        return s.id
