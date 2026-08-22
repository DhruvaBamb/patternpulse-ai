"use client";

import { use } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { api } from "../../../lib/api";
import { useApi } from "../../../hooks/useApi";
import { PageHeader, Page } from "../../../components/page-header";
import { DetectionPanel } from "../../../components/detection-panel";
import { DirectionBadge, ErrorState, Spinner, StatusBadge } from "../../../components/ui";
import { fmtPrice, timeAgo } from "../../../lib/utils";

export default function DetectionDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data: d, loading, error, reload } = useApi(() => api.detection(id), [id], 30000);

  const timeline = (d?.completion_history ?? []).slice(-20);

  return (
    <div>
      <PageHeader
        title={d ? d.pattern : "Formation"}
        subtitle={d ? `${d.symbol} · ${d.interval} · first seen ${timeAgo(d.first_seen)}` : id}
        actions={
          <Link
            href={`/symbol/${encodeURIComponent(d?.symbol ?? "")}`}
            className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs hover:bg-ink-100 dark:hover:bg-ink-800"
          >
            <ArrowLeft size={12} /> Chart
          </Link>
        }
      />
      <Page>
        {loading && !d ? (
          <Spinner />
        ) : error || !d ? (
          <ErrorState title="Detection not found" hint={error ?? undefined} onRetry={reload} />
        ) : (
          <div className="grid gap-4 lg:grid-cols-3">
            <div className="space-y-4 lg:col-span-1">
              <DetectionPanel detection={d} />
              {d.signal ? (
                <div className="panel px-4 py-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium">Signal</span>
                    <span className="font-mono text-sm font-semibold">{d.signal.action}</span>
                  </div>
                  <p className="mt-1.5 text-xs muted">{d.signal.reason}</p>
                  {d.signal.evidence?.length ? (
                    <ul className="mt-2 space-y-0.5 text-2xs muted">
                      {d.signal.evidence.map((e, i) => (
                        <li key={i}>✓ {e}</li>
                      ))}
                    </ul>
                  ) : null}
                  {d.signal.warnings?.length ? (
                    <ul className="mt-1.5 space-y-0.5 text-2xs text-amber-600 dark:text-amber-400">
                      {d.signal.warnings.map((w, i) => (
                        <li key={i}>⚠ {w}</li>
                      ))}
                    </ul>
                  ) : null}
                  <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t pt-2.5 text-xs">
                    {(
                      [
                        ["Entry", d.signal.entry],
                        ["Target", d.signal.target],
                        ["Stop", d.signal.stop_loss],
                        ["R:R", d.signal.risk_reward ? `${d.signal.risk_reward.toFixed(1)}:1` : null],
                      ] as const
                    ).map(([label, v]) => (
                      <div key={label} className="flex justify-between">
                        <dt className="muted">{label}</dt>
                        <dd className="font-mono tabular-nums">{typeof v === "number" ? fmtPrice(v) : "—"}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ) : null}
            </div>

            <div className="space-y-4 lg:col-span-2">
              <div className="panel">
                <div className="border-b px-4 py-2.5 text-xs font-medium">Formation evolution</div>
                {timeline.length < 2 ? (
                  <div className="px-4 py-8 text-center text-xs muted">
                    Completion history builds as the scanner re-observes this formation.
                  </div>
                ) : (
                  <div className="p-4">
                    <div className="flex h-40 items-end gap-1">
                      {timeline.map((p, i) => (
                        <div
                          key={i}
                          className="flex-1 rounded-t bg-amber-500/70 transition-all"
                          style={{ height: `${Math.max(4, p.value)}%` }}
                          title={`${new Date(p.ts).toLocaleString()} → ${p.value.toFixed(0)}%`}
                        />
                      ))}
                    </div>
                    <div className="mt-1 flex justify-between text-2xs muted">
                      <span>{new Date(timeline[0].ts).toLocaleString()}</span>
                      <span>{timeline[timeline.length - 1].value.toFixed(0)}%</span>
                    </div>
                  </div>
                )}
              </div>

              <div className="panel">
                <div className="border-b px-4 py-2.5 text-xs font-medium">Lifecycle events</div>
                {d.events?.length ? (
                  <ol className="divide-y">
                    {d.events.map((e, i) => (
                      <li key={i} className="flex items-center justify-between px-4 py-2 text-xs">
                        <StatusBadge status={e.event.toUpperCase()} />
                        <span className="muted">{JSON.stringify(e.detail).slice(0, 80)}</span>
                        <span className="muted">{timeAgo(e.at)}</span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <div className="px-4 py-6 text-center text-xs muted">No events recorded yet.</div>
                )}
              </div>

              <div className="panel">
                <div className="border-b px-4 py-2.5 text-xs font-medium">Geometric keypoints</div>
                <div className="grid grid-cols-2 gap-2 p-4 sm:grid-cols-3">
                  {Object.entries(d.keypoints ?? {}).map(([k, v]) => {
                    const kp = v as { index?: number; price?: number };
                    return (
                      <div key={k} className="rounded border px-2.5 py-1.5">
                        <div className="text-2xs uppercase tracking-wide muted">{k.replace(/_/g, " ")}</div>
                        <div className="font-mono text-xs tabular-nums">
                          {kp?.price !== undefined ? fmtPrice(kp.price) : String(v)}
                        </div>
                      </div>
                    );
                  })}
                  {Object.keys(d.keypoints ?? {}).length === 0 ? (
                    <div className="col-span-full py-4 text-center text-xs muted">No keypoints recorded.</div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        )}
      </Page>
    </div>
  );
}
