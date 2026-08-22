"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { PatternHeatmap } from "@/components/heatmap";
import { DirectionBadge, EmptyState, Meter, Spinner, Stat, StatusBadge } from "@/components/ui";
import { fmtPct, timeAgo } from "@/lib/utils";

export default function OverviewPage() {
  const health = useApi(() => api.health(), [], 30000);
  const patterns = useApi(() => api.activePatterns(), [], 30000);
  const markets = useApi(() => api.markets(), [], 60000);

  const rows = patterns.data?.patterns ?? [];
  const near = rows.filter((p) => p.completion >= 75 && p.status !== "CONFIRMED");
  const high = rows.filter((p) => p.confidence >= 0.75);
  const bullish = rows.filter((p) => p.direction === "bullish");
  const bearish = rows.filter((p) => p.direction === "bearish");
  const confirmed = rows.filter((p) => p.status === "CONFIRMED");

  return (
    <div>
      <PageHeader
        title="Overview"
        subtitle={
          health.data?.scanner?.last_scan_at
            ? `Scanner last ran ${timeAgo(health.data.scanner.last_scan_at)} · updates stream live`
            : "Waiting for first scanner pass…"
        }
      />
      <Page>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Active patterns" value={rows.length} />
          <Stat label="Near breakout" value={near.length} sub="≥ 75% complete" />
          <Stat label="High confidence" value={high.length} sub="≥ 75% fused" />
          <Stat label="Bullish" value={bullish.length} />
          <Stat label="Bearish" value={bearish.length} />
          <Stat label="Confirmed" value={confirmed.length} sub="breakout registered" />
        </div>

        {health.data && !health.data.cv_detector?.available ? (
          <div className="mt-4 rounded-md border border-amber-500/30 bg-amber-500/5 px-4 py-2.5 text-xs text-amber-600 dark:text-amber-400">
            CV detector unavailable ({health.data.cv_detector?.reason ?? "no trained weights"}) — detections
            currently come from the geometry engine only, with reduced confidence. Train the model to enable
            the full CV + geometry pipeline.
          </div>
        ) : null}

        <div className="mt-4 grid gap-4 lg:grid-cols-5">
          <div className="panel lg:col-span-3">
            <div className="border-b px-4 py-2.5 text-xs font-medium">Market–pattern heatmap</div>
            <div className="p-4">
              <PatternHeatmap />
            </div>
          </div>

          <div className="panel lg:col-span-2">
            <div className="flex items-center justify-between border-b px-4 py-2.5">
              <span className="text-xs font-medium">Strongest formations</span>
              <Link href="/radar" className="text-2xs muted hover:underline">
                full radar →
              </Link>
            </div>
            <div className="divide-y">
              {patterns.loading && !patterns.data ? (
                <Spinner />
              ) : rows.length === 0 ? (
                <EmptyState
                  title="No active patterns yet"
                  hint="The background scanner evaluates the configured universe every few minutes. Patterns will appear here as they form."
                />
              ) : (
                rows.slice(0, 6).map((p) => (
                  <Link key={p.id} href={`/active/${p.id}`} className="block px-4 py-2.5 hover:bg-ink-100/60 dark:hover:bg-ink-800/40">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="font-mono text-xs">{p.symbol.replace(".NS", "")}</span>
                        <span className="truncate text-xs">{p.pattern}</span>
                      </div>
                      <div className="flex shrink-0 items-center gap-1.5">
                        <DirectionBadge direction={p.direction} />
                        <StatusBadge status={p.status} />
                      </div>
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-3">
                      <Meter value={p.completion} label="Completion" tone="bg-amber-500" />
                      <Meter value={p.confidence * 100} label="Confidence" tone="bg-bull" />
                    </div>
                  </Link>
                ))
              )}
            </div>
          </div>
        </div>

        <div className="mt-4 panel">
          <div className="border-b px-4 py-2.5 text-xs font-medium">Markets</div>
          <div className="dense-scroll overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left muted">
                  <th className="px-4 py-2 font-medium">Symbol</th>
                  <th className="px-4 py-2 font-medium">Name</th>
                  <th className="px-4 py-2 text-right font-medium">Last</th>
                  <th className="px-4 py-2 text-right font-medium">Change</th>
                  <th className="px-4 py-2 text-right font-medium">Updated</th>
                </tr>
              </thead>
              <tbody>
                {(markets.data?.symbols ?? []).map((s) => (
                  <tr key={s.symbol} className="border-b last:border-0 hover:bg-ink-100/50 dark:hover:bg-ink-800/40">
                    <td className="px-4 py-1.5 font-mono">
                      <Link href={`/symbol/${encodeURIComponent(s.symbol)}`} className="hover:underline">
                        {s.symbol}
                      </Link>
                    </td>
                    <td className="px-4 py-1.5 muted">{s.name}</td>
                    <td className="px-4 py-1.5 text-right font-mono tabular-nums">
                      {s.last_price?.toFixed(2) ?? "—"}
                    </td>
                    <td
                      className={`px-4 py-1.5 text-right font-mono tabular-nums ${
                        (s.change_pct ?? 0) >= 0 ? "text-bull" : "text-bear"
                      }`}
                    >
                      {s.change_pct !== null ? fmtPct(s.change_pct) : "—"}
                    </td>
                    <td className="px-4 py-1.5 text-right muted">{timeAgo(s.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p className="mt-4 text-2xs muted">
          Research-oriented technical analysis — not investment advice. Pattern intelligence indicates
          historical structure, never guaranteed outcomes.
        </p>
      </Page>
    </div>
  );
}
