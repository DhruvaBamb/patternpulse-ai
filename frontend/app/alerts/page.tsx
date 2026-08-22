"use client";

import { useState } from "react";
import { Bell, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { EmptyState, Spinner } from "@/components/ui";
import { timeAgo } from "@/lib/utils";

export default function AlertsPage() {
  const alerts = useApi(() => api.alerts(), [], 20000);
  const notifications = useApi(() => api.notifications(false), [], 20000);
  const registry = useApi(() => api.registry(), []);

  const [pattern, setPattern] = useState("");
  const [conditionType, setConditionType] = useState("completion");
  const [value, setValue] = useState(70);
  const [confidenceMin, setConfidenceMin] = useState(85);
  const [creating, setCreating] = useState(false);

  const create = async () => {
    setCreating(true);
    try {
      await api.createAlert({
        pattern_id: pattern || null,
        condition: {
          type: conditionType,
          value: conditionType === "completion" ? value : undefined,
          confidence_min: confidenceMin / 100,
        },
      });
      alerts.reload();
    } finally {
      setCreating(false);
    }
  };

  return (
    <div>
      <PageHeader title="Alerts" subtitle="In-app notifications on formation events (email/Telegram adapters planned)" />
      <Page>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="panel px-4 py-3">
            <div className="mb-3 flex items-center gap-1.5 text-xs font-medium">
              <Bell size={13} /> New alert
            </div>
            <div className="space-y-2.5 text-xs">
              <label className="block">
                <span className="mb-1 block text-2xs uppercase tracking-wide muted">Pattern (any if empty)</span>
                <select
                  value={pattern}
                  onChange={(e) => setPattern(e.target.value)}
                  className="w-full rounded-md border bg-[var(--panel)] px-2 py-1.5"
                >
                  <option value="">Any pattern</option>
                  {(registry.data?.patterns ?? [])
                    .filter((p) => p.implemented)
                    .map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block text-2xs uppercase tracking-wide muted">Condition</span>
                <select
                  value={conditionType}
                  onChange={(e) => setConditionType(e.target.value)}
                  className="w-full rounded-md border bg-[var(--panel)] px-2 py-1.5"
                >
                  <option value="completion">Completion threshold crossed</option>
                  <option value="first_detected">Pattern first detected</option>
                  <option value="breakout_confirmed">Breakout confirmed</option>
                </select>
              </label>
              {conditionType === "completion" ? (
                <label className="block">
                  <span className="mb-1 block text-2xs uppercase tracking-wide muted">
                    Completion ≥ {value}%
                  </span>
                  <input
                    type="range"
                    min={30}
                    max={100}
                    step={5}
                    value={value}
                    onChange={(e) => setValue(Number(e.target.value))}
                    className="w-full"
                  />
                </label>
              ) : null}
              <label className="block">
                <span className="mb-1 block text-2xs uppercase tracking-wide muted">
                  Confidence ≥ {confidenceMin}%
                </span>
                <input
                  type="range"
                  min={50}
                  max={100}
                  step={5}
                  value={confidenceMin}
                  onChange={(e) => setConfidenceMin(Number(e.target.value))}
                  className="w-full"
                />
              </label>
              <button
                onClick={create}
                disabled={creating}
                className="w-full rounded-md border px-3 py-1.5 font-medium hover:bg-ink-100 disabled:opacity-60 dark:hover:bg-ink-800"
              >
                {creating ? "Creating…" : "Create alert"}
              </button>
            </div>
          </div>

          <div className="panel">
            <div className="border-b px-4 py-2.5 text-xs font-medium">Active alerts</div>
            {alerts.loading && !alerts.data ? (
              <Spinner />
            ) : (alerts.data?.alerts ?? []).length === 0 ? (
              <EmptyState title="No alerts configured" hint="Create one on the left — evaluated after every scanner pass." />
            ) : (
              <ul className="divide-y">
                {(alerts.data?.alerts ?? []).map((a) => (
                  <li key={a.id} className="flex items-center justify-between gap-2 px-4 py-2.5 text-xs">
                    <div>
                      <div className="font-mono">{a.pattern_id?.replace(/_/g, " ") ?? "any pattern"}</div>
                      <div className="text-2xs muted">
                        {String(a.condition?.type ?? "")}
                        {a.condition?.value !== undefined ? ` ≥ ${String(a.condition.value)}` : ""} · conf ≥{" "}
                        {(((a.condition?.confidence_min as number) ?? 0) * 100).toFixed(0)}%
                        {a.last_fired_at ? ` · fired ${timeAgo(a.last_fired_at)}` : ""}
                      </div>
                    </div>
                    <button
                      onClick={() => api.deleteAlert(a.id).then(alerts.reload)}
                      aria-label="Delete alert"
                      className="muted hover:text-bear"
                    >
                      <Trash2 size={13} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="mt-4 panel">
          <div className="border-b px-4 py-2.5 text-xs font-medium">Notification feed</div>
          {(notifications.data?.notifications ?? []).length === 0 ? (
            <EmptyState title="No notifications yet" hint="Fired alerts appear here in-app." />
          ) : (
            <ul className="divide-y">
              {(notifications.data?.notifications ?? []).map((n) => (
                <li key={n.id} className="px-4 py-2.5 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{n.title}</span>
                    <span className="text-2xs muted">{timeAgo(n.created_at)}</span>
                  </div>
                  <p className="mt-0.5 muted">{n.body}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Page>
    </div>
  );
}
