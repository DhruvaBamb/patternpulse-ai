"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { DirectionBadge, EmptyState, Spinner, StatusBadge } from "@/components/ui";
import { cn } from "@/lib/utils";

type SortKey = "confidence" | "completion" | "pattern" | "symbol" | "category" | "interval" | "direction";

export default function RadarPage() {
  const { data, loading, error, reload } = useApi(() => api.radar(), [], 45000);
  const [sort, setSort] = useState<SortKey>("confidence");
  const [category, setCategory] = useState("");
  const [direction, setDirection] = useState("");
  const [tf, setTf] = useState("");
  const [minCompletion, setMinCompletion] = useState(0);
  const [minConfidence, setMinConfidence] = useState(0);

  const rows = useMemo(() => {
    let r = data?.rows ?? [];
    if (category) r = r.filter((x) => x.category === category);
    if (direction) r = r.filter((x) => x.direction === direction);
    if (tf) r = r.filter((x) => x.interval === tf);
    r = r.filter((x) => x.completion >= minCompletion && x.confidence * 100 >= minConfidence);
    const keymap: Record<SortKey, (x: (typeof r)[number]) => string | number> = {
      confidence: (x) => x.confidence,
      completion: (x) => x.completion,
      pattern: (x) => x.pattern,
      symbol: (x) => x.symbol,
      category: (x) => x.category,
      interval: (x) => x.interval,
      direction: (x) => x.direction,
    };
    const key = keymap[sort];
    return [...r].sort((a, b) => {
      const va = key(a);
      const vb = key(b);
      if (typeof va === "number" && typeof vb === "number") return vb - va;
      return String(va).localeCompare(String(vb));
    });
  }, [data, sort, category, direction, tf, minCompletion, minConfidence]);

  const th = (label: string, key: SortKey, align = "left") => (
    <th
      className={cn("cursor-pointer select-none px-3 py-2 font-medium", align === "right" && "text-right")}
      onClick={() => setSort(key)}
      title={`Sort by ${label}`}
    >
      {label}
      {sort === key ? " ▾" : ""}
    </th>
  );

  return (
    <div>
      <PageHeader
        title="Pattern Radar"
        subtitle="Formations currently tracked across the scanned universe"
        actions={
          <button
            onClick={reload}
            className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs hover:bg-ink-100 dark:hover:bg-ink-800"
          >
            <RefreshCw size={12} className={cn(loading && "animate-spin")} /> Refresh
          </button>
        }
      />
      <Page>
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="rounded-md border bg-[var(--panel)] px-2 py-1"
            aria-label="Category filter"
          >
            <option value="">All categories</option>
            <option value="classical">Classical</option>
            <option value="candlestick">Candlestick</option>
            <option value="harmonic">Harmonic</option>
          </select>
          <select
            value={direction}
            onChange={(e) => setDirection(e.target.value)}
            className="rounded-md border bg-[var(--panel)] px-2 py-1"
            aria-label="Direction filter"
          >
            <option value="">Any direction</option>
            <option value="bullish">Bullish</option>
            <option value="bearish">Bearish</option>
          </select>
          <select
            value={tf}
            onChange={(e) => setTf(e.target.value)}
            className="rounded-md border bg-[var(--panel)] px-2 py-1"
            aria-label="Timeframe filter"
          >
            <option value="">All timeframes</option>
            {["15m", "1h", "1d"].map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <label className="flex items-center gap-1.5 muted">
            Completion ≥ {minCompletion}%
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={minCompletion}
              onChange={(e) => setMinCompletion(Number(e.target.value))}
              className="w-24"
            />
          </label>
          <label className="flex items-center gap-1.5 muted">
            Confidence ≥ {minConfidence}%
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              className="w-24"
            />
          </label>
        </div>

        {loading && !data ? (
          <Spinner label="Scanning…" />
        ) : rows.length === 0 ? (
          <EmptyState
            title="No formations match these filters"
            hint={error ? error : "Relax the filters, or wait for the next scanner pass — new formations appear automatically."}
          />
        ) : (
          <div className="panel dense-scroll overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left muted">
                  {th("Stock", "symbol")}
                  {th("Pattern", "pattern")}
                  {th("Category", "category")}
                  {th("Direction", "direction")}
                  {th("Timeframe", "interval")}
                  {th("Completion", "completion", "right")}
                  {th("Confidence", "confidence", "right")}
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Source</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className="border-b last:border-0 hover:bg-ink-100/50 dark:hover:bg-ink-800/40">
                    <td className="px-3 py-2 font-mono">
                      <Link href={`/symbol/${encodeURIComponent(r.symbol)}`} className="hover:underline">
                        {r.symbol.replace(".NS", "")}
                      </Link>
                    </td>
                    <td className="px-3 py-2">
                      <Link href={`/active/${r.id}`} className="hover:underline">
                        {r.pattern}
                      </Link>
                    </td>
                    <td className="px-3 py-2 muted">{r.category}</td>
                    <td className="px-3 py-2">
                      <DirectionBadge direction={r.direction} />
                    </td>
                    <td className="px-3 py-2 font-mono muted">{r.interval}</td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">{r.completion.toFixed(0)}%</td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">{(r.confidence * 100).toFixed(0)}%</td>
                    <td className="px-3 py-2">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="px-3 py-2 muted">{r.source === "geometry_only" ? "geometry" : "CV+geometry"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-3 text-2xs muted">
          Confidence is fused from visual-model score, geometry validation, trend and volume evidence.
          Geometry-only rows (no trained CV weights) carry a reduced-confidence penalty.
        </p>
      </Page>
    </div>
  );
}
