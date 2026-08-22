"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { DirectionBadge, EmptyState, Spinner, StatusBadge } from "@/components/ui";
import { timeAgo } from "@/lib/utils";

export default function ActivePatternsPage() {
  const { data, loading, error } = useApi(() => api.activePatterns(), [], 30000);
  const rows = data?.patterns ?? [];

  return (
    <div>
      <PageHeader title="Active Patterns" subtitle="All tracked formations with lifecycle state" />
      <Page>
        {loading && !data ? (
          <Spinner />
        ) : rows.length === 0 ? (
          <EmptyState
            title="No active formations"
            hint={error ?? "The scanner has not found qualifying structures yet. Check back after the next pass."}
          />
        ) : (
          <div className="panel dense-scroll overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left muted">
                  <th className="px-3 py-2 font-medium">Symbol</th>
                  <th className="px-3 py-2 font-medium">Pattern</th>
                  <th className="px-3 py-2 font-medium">Direction</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Stage</th>
                  <th className="px-3 py-2 text-right font-medium">Completion</th>
                  <th className="px-3 py-2 text-right font-medium">Confidence</th>
                  <th className="px-3 py-2 text-right font-medium">First seen</th>
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
                    <td className="px-3 py-2">
                      <DirectionBadge direction={r.direction} />
                    </td>
                    <td className="px-3 py-2">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="px-3 py-2 muted">{r.stage?.replace(/_/g, " ") || "—"}</td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">{r.completion.toFixed(0)}%</td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">{(r.confidence * 100).toFixed(0)}%</td>
                    <td className="px-3 py-2 text-right muted">{timeAgo(r.first_seen)}</td>
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
