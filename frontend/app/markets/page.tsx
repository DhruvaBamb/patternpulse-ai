"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { ErrorState, Spinner } from "@/components/ui";
import { fmtPct, timeAgo } from "@/lib/utils";

export default function MarketsPage() {
  const { data, loading, error, reload } = useApi(() => api.markets(), [], 60000);

  return (
    <div>
      <PageHeader title="Markets" subtitle="Configured scanner universe with live quotes" />
      <Page>
        {loading && !data ? (
          <Spinner label="Loading markets…" />
        ) : error ? (
          <ErrorState
            title="Market data unavailable"
            hint={error === "Failed to fetch" ? "Backend not reachable on :8000." : error}
            onRetry={reload}
          />
        ) : (
          <div className="panel dense-scroll overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left muted">
                  <th className="px-4 py-2.5 font-medium">Symbol</th>
                  <th className="px-4 py-2.5 font-medium">Name</th>
                  <th className="px-4 py-2.5 font-medium">Market</th>
                  <th className="px-4 py-2.5 text-right font-medium">Last</th>
                  <th className="px-4 py-2.5 text-right font-medium">Change</th>
                  <th className="px-4 py-2.5 text-right font-medium">Updated</th>
                </tr>
              </thead>
              <tbody>
                {(data?.symbols ?? []).map((s) => (
                  <tr key={s.symbol} className="border-b last:border-0 hover:bg-ink-100/50 dark:hover:bg-ink-800/40">
                    <td className="px-4 py-2 font-mono">
                      <Link href={`/symbol/${encodeURIComponent(s.symbol)}`} className="font-medium hover:underline">
                        {s.symbol}
                      </Link>
                    </td>
                    <td className="px-4 py-2 muted">{s.name}</td>
                    <td className="px-4 py-2 muted">{s.market}</td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">{s.last_price?.toFixed(2) ?? "—"}</td>
                    <td
                      className={`px-4 py-2 text-right font-mono tabular-nums ${
                        (s.change_pct ?? 0) >= 0 ? "text-bull" : "text-bear"
                      }`}
                    >
                      {s.change_pct !== null ? fmtPct(s.change_pct) : "—"}
                    </td>
                    <td className="px-4 py-2 text-right muted">{timeAgo(s.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Page>
    </div>
  );
}
