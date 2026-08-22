"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Bot, Send, User } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { Spinner } from "@/components/ui";
import { cn } from "@/lib/utils";

interface Msg {
  role: "user" | "assistant";
  content: string;
}

const SUGGESTIONS = [
  "What patterns are forming right now?",
  "Show bullish patterns above 80% completion",
  "What does a Falling Wedge mean?",
  "Which formations are close to breakout?",
];

export default function AssistantPage() {
  return (
    <Suspense>
      <AssistantContent />
    </Suspense>
  );
}

function AssistantContent() {
  const search = useSearchParams();
  const status = useApi(() => api.chatStatus(), []);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const q = search.get("q");
    if (q) setInput(q);
  }, [search]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  const send = async (text?: string) => {
    const message = (text ?? input).trim();
    if (!message || streaming) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: message }]);
    setStreaming(true);
    let acc = "";
    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId }),
      });
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const data = line.replace(/^data: /, "").trim();
          if (!data || data === "[DONE]") continue;
          try {
            const ev = JSON.parse(data);
            if (ev.type === "session") setSessionId(ev.session_id);
            else if (ev.type === "delta") {
              acc += ev.text;
              setMessages((m) => {
                const copy = [...m];
                copy[copy.length - 1] = { role: "assistant", content: acc };
                return copy;
              });
            } else if (ev.type === "error") {
              acc += `\n\n⚠ ${ev.message}`;
              setMessages((m) => {
                const copy = [...m];
                copy[copy.length - 1] = { role: "assistant", content: acc };
                return copy;
              });
            }
          } catch {
            /* skip malformed */
          }
        }
      }
    } catch {
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", content: "⚠ backend unreachable — is it running on :8000?" };
        return copy;
      });
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div className="flex h-screen flex-col">
      <PageHeader
        title="PulseAI"
        subtitle={
          status.data?.available
            ? `Project-aware assistant · ${status.data.model}`
            : "Assistant idle — add GROQ_API_KEY to .env to enable"
        }
      />
      <div className="flex min-h-0 flex-1 flex-col px-6 pb-5">
        <div className="panel flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4 dense-scroll">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-3">
                <Bot size={20} className="muted" />
                <p className="max-w-md text-center text-xs muted">
                  I answer from this platform&apos;s live detections, signals and backtests — I never invent
                  market data. Ask about formations, completion, confidence or pattern education.
                </p>
                <div className="flex flex-wrap justify-center gap-1.5">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="rounded-full border px-3 py-1 text-2xs muted hover:bg-ink-100 dark:hover:bg-ink-800"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={cn("flex gap-2", m.role === "user" && "justify-end")}>
                  {m.role === "assistant" ? <Bot size={14} className="mt-1 shrink-0 muted" /> : null}
                  <div
                    className={cn(
                      "max-w-[75%] whitespace-pre-wrap rounded-lg px-3 py-2 text-xs leading-relaxed",
                      m.role === "user"
                        ? "bg-ink-200/50 dark:bg-ink-800"
                        : "border"
                    )}
                  >
                    {m.content || (streaming ? "…" : "")}
                  </div>
                  {m.role === "user" ? <User size={14} className="mt-1 shrink-0 muted" /> : null}
                </div>
              ))
            )}
            <div ref={endRef} />
          </div>
          <form
            className="flex items-center gap-2 border-t p-3"
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={status.data?.available ? "Ask about patterns, symbols, signals…" : "GROQ_API_KEY not configured — chat disabled"}
              disabled={!status.data?.available || streaming}
              className="flex-1 rounded-md border bg-transparent px-3 py-1.5 text-xs outline-none focus:ring-1 focus:ring-ink-400"
              aria-label="Message"
            />
            <button
              type="submit"
              disabled={!status.data?.available || streaming || !input.trim()}
              className="flex items-center gap-1 rounded-md border px-3 py-1.5 text-xs disabled:opacity-50"
            >
              <Send size={12} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
