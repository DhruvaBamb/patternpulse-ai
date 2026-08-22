"use client";

import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { Spinner } from "@/components/ui";

export default function SettingsPage() {
  const health = useApi(() => api.health(), [], 30000);
  const models = useApi(() => api.models(), [], 60000);
  const chat = useApi(() => api.chatStatus(), []);
  const registry = useApi(() => api.registry(), []);

  const h = health.data;

  const rows: { label: string; value: string; note?: string }[] = [
    {
      label: "Backend",
      value: h?.status === "ok" ? "running" : "unreachable",
      note: h ? `${h.app} (${h.env})` : "start with: uvicorn backend.app.main:app --reload",
    },
    {
      label: "Scanner",
      value: h?.scanner?.running ? "running" : "idle",
      note: h?.scanner?.last_scan_at
        ? `last pass ${new Date(h.scanner.last_scan_at).toLocaleString()}`
        : "first pass in progress",
    },
    {
      label: "CV detector",
      value: h?.cv_detector?.available ? "weights loaded" : "not trained",
      note:
        h?.cv_detector?.available
          ? h.cv_detector.device ?? ""
          : h?.cv_detector?.reason ?? "geometry-only mode active until a model is trained",
    },
    {
      label: "PulseAI (Groq)",
      value: chat.data?.available ? `enabled · ${chat.data.model}` : "disabled",
      note: chat.data?.reason ?? undefined,
    },
    {
      label: "Pattern registry",
      value: `${registry.data?.total ?? "—"} patterns`,
      note: `version ${registry.data?.registry_version ?? "?"} · implemented ${
        (registry.data?.patterns ?? []).filter((p) => p.implemented).length
      }`,
    },
    {
      label: "Market data",
      value: "yfinance",
      note: "provider adapters for Kite/MT5/Polygon/TwelveData/Binance are stubbed until credentials exist",
    },
  ];

  return (
    <div>
      <PageHeader title="Settings" subtitle="System status and configuration overview" />
      <Page>
        {health.loading && !h ? (
          <Spinner />
        ) : (
          <div className="panel divide-y">
            {rows.map((r) => (
              <div key={r.label} className="flex items-start justify-between gap-4 px-4 py-3 text-xs">
                <div>
                  <div className="font-medium">{r.label}</div>
                  {r.note ? <div className="mt-0.5 muted">{r.note}</div> : null}
                </div>
                <div className="font-mono">{r.value}</div>
              </div>
            ))}
          </div>
        )}
        <p className="mt-4 text-2xs muted">
          Tunables live in configs/app.yaml (scanner universe, timeframes, fusion weights, signal gates) and
          .env (database, Groq credentials). Restart the backend after changes.
        </p>
      </Page>
    </div>
  );
}
