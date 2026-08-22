"use client";

import Link from "next/link";
import { Star, X } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { EmptyState, Spinner } from "@/components/ui";
import { fmtPct, timeAgo } from "@/lib/utils";

export default function WatchlistPage() {
  const { data, loading, reload } = useApi(() => api.watchlist(), [], 30000);
  const markets = useApi(() => api.markets(), [], 60000);
  const items = data?.items ?? [];

  return (
    <div>
      <PageHeader title="Watchlist" subtitle="Saved symbols with their strongest current formation" />
      <Page>
        {loading && !data ? (
          <Spinner />
        ) : items.length === 0 ? (
          <EmptyState
            title="Watchlist is empty"
            hint="Open any stock page and click Watch, or add from Markets."
          />
        ) : (
          <div className="panel dense-scroll overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left muted">
                  <th className="px-4 py-2.5 font-medium">Symbol</th>
                  <th className="px-4 py-2.5 text-right font-medium">Price</th>
                  <th className="px-4 py-2.5 text-right font-medium">Change</th>
                  <th className="px-4 py-2.5 font-medium">Pattern</th>
                  <th className="px-4 py-2.5 text-right font-medium">Completion</th>
                  <th className="px-4 py-2.5 text-right font-medium">Confidence</th>
                  <th className="px-4 py-2.5 text-right font-medium">Updated</th>
                  <th className="px-4 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {items.map((w) => (
                  <tr key={w.symbol} className="border-b last:border-0 hover:bg-ink-100/50 dark:hover:bg-ink-800/40">
                    <td className="px-4 py-2 font-mono">
                      <Link href={`/symbol/${encodeURIComponent(w.symbol)}`} className="hover:underline">
                        {w.symbol}
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">
                      {w.last_price?.toFixed(2) ?? "—"}
                    </td>
                    <td
                      className={`px-4 py-2 text-right font-mono tabular-nums ${
                        (w.change_pct ?? 0) >= 0 ? "text-bull" : "text-bear"
                      }`}
                    >
                      {w.change_pct !== null ? fmtPct(w.change_pct) : "—"}
                    </td>
                    <td className="px-4 py-2">
                      {w.pattern ?? <span className="muted">no active pattern</span>}
                      {w.interval ? <span className="ml-1.5 text-2xs muted">{w.interval}</span> : null}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">
                      {w.completion !== null ? `${w.completion.toFixed(0)}%` : "—"}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">
                      {w.confidence !== null ? `${(w.confidence * 100).toFixed(0)}%` : "—"}
                    </td>
                    <td className="px-4 py-2 text-right muted">{timeAgo(w.updated_at)}</td>
                    <td className="px-4 py-2 text-right">
                      <button
                        onClick={() => api.removeWatch(w.symbol).then(reload)}
                        aria-label={`Remove ${w.symbol}`}
                        className="muted hover:text-bear"
                      >
                        <X size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs muted">
            <Star size={12} /> quick add from the scanned universe
          </div>
          <div className="flex flex-wrap gap-1.5">
            {(markets.data?.symbols ?? [])
              .filter((s) => !items.some((i) => i.symbol === s.symbol))
              .map((s) => (
                <button
                  key={s.symbol}
                  onClick={() => api.addWatch(s.symbol).then(reload)}
                  className="rounded-full border px-2.5 py-1 font-mono text-2xs muted hover:bg-ink-100 dark:hover:bg-ink-800"
                >
                  + {s.symbol}
                </button>
              ))}
          </div>
        </div>
      </Page>
    </div>
  );
}
