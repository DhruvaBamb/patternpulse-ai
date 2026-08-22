"use client";

import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { EmptyState, Spinner, Stat } from "@/components/ui";
import { timeAgo } from "@/lib/utils";

export default function AnalyticsPage() {
  const models = useApi(() => api.models(), [], 30000);
  const training = useApi(() => api.trainingStatus(), [], 15000);
  const health = useApi(() => api.health(), [], 30000);

  const cv = health.data?.cv_detector;
  const runs = models.data?.runs ?? [];
  const t = training.data as Record<string, unknown> | undefined;

  return (
    <div>
      <PageHeader
        title="Model Analytics"
        subtitle={`Detector status, training runs and evaluation metrics · registry v${models.data?.registry_version ?? "?"}`}
      />
      <Page>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat
            label="CV detector"
            value={cv?.available ? "available" : "unavailable"}
            sub={cv?.available ? cv.reason || "weights loaded" : cv?.reason || "no trained weights"}
          />
          <Stat label="Device" value={(cv as { device?: string } | undefined)?.device ?? "—"} />
          <Stat label="Training runs" value={runs.length} />
          <Stat
            label="Latest epoch history"
            value={runs[0]?.history_epochs ?? 0}
            sub={runs[0]?.run ?? "no runs"}
          />
        </div>

        <div className="mt-4 panel">
          <div className="border-b px-4 py-2.5 text-xs font-medium">Training status</div>
          <div className="p-4">
            {t && t.status !== "unknown" ? (
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-4">
                {Object.entries(t).slice(0, 12).map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-2xs uppercase tracking-wide muted">{k.replace(/_/g, " ")}</dt>
                    <dd className="mt-0.5 font-mono">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <div className="py-4 text-center text-xs muted">
                No training monitor attached. Metrics appear after running{" "}
                <code className="rounded bg-ink-200/50 px-1 py-0.5 font-mono text-2xs dark:bg-ink-800">
                  python -m ml.training.train --auto-resume
                </code>
                .
              </div>
            )}
          </div>
        </div>

        <div className="mt-4 panel">
          <div className="border-b px-4 py-2.5 text-xs font-medium">Training runs</div>
          {models.loading && !models.data ? (
            <Spinner />
          ) : runs.length === 0 ? (
            <EmptyState
              title="No training runs found"
              hint="Generate a dataset and train the detector — artifacts land under artifacts/runs/."
            />
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left muted">
                  <th className="px-4 py-2 font-medium">Run</th>
                  <th className="px-4 py-2 text-right font-medium">Epochs recorded</th>
                  <th className="px-4 py-2 font-medium">Metadata</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run} className="border-b last:border-0">
                    <td className="px-4 py-1.5 font-mono">{r.run}</td>
                    <td className="px-4 py-1.5 text-right font-mono tabular-nums">{r.history_epochs}</td>
                    <td className="px-4 py-1.5 muted">{r.has_meta ? "config + env fingerprint saved" : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="mt-4 panel px-4 py-3 text-xs muted">
          <div className="mb-1 font-medium text-[var(--text)]">Evaluation workflow</div>
          <ol className="list-inside list-decimal space-y-1">
            <li>
              <code className="font-mono text-2xs">python -m ml.datasets.synthetic.generate --samples 200</code>
            </li>
            <li>
              <code className="font-mono text-2xs">python -m ml.training.train --auto-resume</code> (resumes
              automatically after interruption)
            </li>
            <li>
              <code className="font-mono text-2xs">
                python -m ml.evaluation.evaluate --weights artifacts/runs/&lt;run&gt;/best.pt
              </code>{" "}
              → mAP@0.5, mAP@0.5:0.95, per-class AP and confusion matrix under artifacts/eval/
            </li>
          </ol>
        </div>
      </Page>
    </div>
  );
}
