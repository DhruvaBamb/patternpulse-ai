"use client";

import { useState } from "react";
import { Play } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { EmptyState, Spinner } from "@/components/ui";
import { timeAgo } from "@/lib/utils";

export default function BacktestingPage() {
  const { data, loading, reload } = useApi(() => api.backtests(), [], 15000);
  const [launching, setLaunching] = useState(false);

  const launch = async () => {
    setLaunching(true);
    try {
      await api.runBacktest();
      setTimeout(reload, 3000);
    } finally {
      setLaunching(false);
    }
  };

  const runs = data?.runs ?? [];

  return (
    <div>
      <PageHeader
        title="Backtesting"
        subtitle="Walk-forward evaluation of detected patterns (no look-ahead)"
        actions={
          <button
            onClick={launch}
            disabled={launching}
            className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs hover:bg-ink-100 disabled:opacity-60 dark:hover:bg-ink-800"
          >
            <Play size={12} className={launching ? "animate-pulse" : ""} />
            {launching ? "Queueing…" : "Run backtest"}
          </button>
        }
      />
      <Page>
        {loading && !data ? (
          <Spinner />
        ) : runs.length === 0 ? (
          <EmptyState
            title="No backtests yet"
            hint="Launch a run to evaluate how detected patterns performed historically. Results aggregate per pattern: hit rate, expectancy, profit factor."
          />
        ) : (
          <div className="space-y-4">
            {runs.map((r) => (
              <div key={r.id} className="panel">
                <div className="flex items-center justify-between border-b px-4 py-2.5">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="font-mono">{r.id.slice(0, 10)}</span>
                    <span
                      className={
                        r.status === "completed"
                          ? "text-bull"
                          : r.status === "failed"
                            ? "text-bear"
                            : "text-amber-500"
                      }
                    >
                      {r.status}
                    </span>
                    <span className="muted">{timeAgo(r.created_at)}</span>
                  </div>
                  {r.overall ? (
                    <span className="font-mono text-xs muted">
                      {String(r.overall.n_trades ?? 0)} trades · hit rate{" "}
                      {(((r.overall.hit_rate as number) ?? 0) * 100).toFixed(0)}%
                    </span>
                  ) : null}
                </div>
                {r.pattern_stats && Object.keys(r.pattern_stats).length > 0 ? (
                  <div className="dense-scroll overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b text-left muted">
                          <th className="px-4 py-2 font-medium">Pattern</th>
                          <th className="px-4 py-2 text-right font-medium">Obs</th>
                          <th className="px-4 py-2 text-right font-medium">Hit rate</th>
                          <th className="px-4 py-2 text-right font-medium">Avg return</th>
                          <th className="px-4 py-2 text-right font-medium">Median</th>
                          <th className="px-4 py-2 text-right font-medium">Profit factor</th>
                          <th className="px-4 py-2 text-right font-medium">Max DD</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(r.pattern_stats)
                          .sort((a, b) => (b[1].observations ?? 0) - (a[1].observations ?? 0))
                          .map(([pid, s]) => (
                            <tr key={pid} className="border-b last:border-0">
                              <td className="px-4 py-1.5 font-mono">{pid.replace(/_/g, " ")}</td>
                              <td className="px-4 py-1.5 text-right font-mono tabular-nums">{s.observations}</td>
                              <td className="px-4 py-1.5 text-right font-mono tabular-nums">
                                {((s.hit_rate ?? 0) * 100).toFixed(0)}%
                              </td>
                              <td
                                className={`px-4 py-1.5 text-right font-mono tabular-nums ${
                                  (s.avg_return_pct ?? 0) >= 0 ? "text-bull" : "text-bear"
                                }`}
                              >
                                {(s.avg_return_pct ?? 0).toFixed(1)}%
                              </td>
                              <td className="px-4 py-1.5 text-right font-mono tabular-nums">
                                {(s.median_return_pct ?? 0).toFixed(1)}%
                              </td>
                              <td className="px-4 py-1.5 text-right font-mono tabular-nums">
                                {s.profit_factor ?? "—"}
                              </td>
                              <td className="px-4 py-1.5 text-right font-mono tabular-nums">
                                {(s.max_drawdown_pct ?? 0).toFixed(1)}%
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="px-4 py-6 text-center text-xs muted">
                    {r.status === "running" ? "Running — this can take a few minutes…" : "No pattern statistics."}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        <p className="mt-4 text-2xs muted">
          Statistics are computed from detected formations with entry at breakout close and exits at the
          pattern&apos;s own target/invalidation levels. Past structure does not guarantee future outcomes.
        </p>
      </Page>
    </div>
  );
}
