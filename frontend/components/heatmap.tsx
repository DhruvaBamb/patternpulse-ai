"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { cn, timeAgo } from "@/lib/utils";
import { EmptyState, Spinner } from "./ui";

export function PatternHeatmap() {
  const { data, loading, error, reload } = useApi(() => api.heatmap(), [], 60000);

  if (loading && !data) return <Spinner label="Loading heatmap…" />;
  if (error)
    return (
      <EmptyState
        title="Heatmap unavailable"
        hint={error === "Failed to fetch" ? "Backend not reachable — is it running on :8000?" : error}
      />
    );
  if (!data || data.symbols.length === 0) return <EmptyState title="No symbols configured" />;

  const { symbols, timeframes, cells } = data;

  return (
    <div className="dense-scroll overflow-x-auto">
      <table className="w-full border-separate border-spacing-0 text-xs">
        <thead>
          <tr>
            <th className="sticky left-0 z-10 bg-[var(--panel)] pb-2 pr-3 text-left font-medium muted">Symbol</th>
            {timeframes.map((tf) => (
              <th key={tf} className="px-1 pb-2 text-center font-medium muted">
                {tf}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {symbols.map((sym) => (
            <tr key={sym}>
              <td className="sticky left-0 z-10 bg-[var(--panel)] py-0.5 pr-3 font-mono">
                <Link href={`/symbol/${encodeURIComponent(sym)}`} className="hover:underline">
                  {sym.replace(".NS", "")}
                </Link>
              </td>
              {timeframes.map((tf) => {
                const cell = cells[`${sym}|${tf}`];
                if (!cell)
                  return (
                    <td key={tf} className="px-0.5 py-0.5">
                      <div className="h-7 rounded border border-dashed opacity-40" title="No pattern detected" />
                    </td>
                  );
                const tone =
                  cell.direction === "bullish"
                    ? "bg-[#26a69a]"
                    : cell.direction === "bearish"
                      ? "bg-[#ef5350]"
                      : "bg-ink-400";
                return (
                  <td key={tf} className="px-0.5 py-0.5">
                    <div
                      className={cn("flex h-7 items-center justify-center rounded font-mono text-2xs text-white")}
                      style={{ opacity: 0.35 + Math.min(cell.confidence, 1) * 0.6 }}
                      title={`${cell.pattern} · ${cell.completion.toFixed(0)}% complete · ${(cell.confidence * 100).toFixed(0)}% confidence · ${timeAgo(cell.last_seen)}`}
                    >
                      {cell.completion.toFixed(0)}%
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
