"use client";

import { useMemo, useState } from "react";
import { api, type PatternDef } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { DirectionBadge, EmptyState, Spinner } from "@/components/ui";
import { cn } from "@/lib/utils";

const CATEGORIES = [
  { key: "classical", label: "Classical chart patterns" },
  { key: "candlestick", label: "Candlestick patterns" },
  { key: "harmonic", label: "Harmonic patterns" },
];

export default function ExplorerPage() {
  const [category, setCategory] = useState("classical");
  const [query, setQuery] = useState("");
  const registry = useApi(() => api.registry(), []);
  const backtests = useApi(() => api.backtests(), []);

  const latestStats = useMemo(() => {
    const completed = (backtests.data?.runs ?? []).filter((r) => r.status === "completed");
    return completed[0]?.pattern_stats ?? null;
  }, [backtests.data]);

  const patterns = useMemo(() => {
    let p = (registry.data?.patterns ?? []).filter((x) => x.category === category);
    if (query) {
      const q = query.toLowerCase();
      p = p.filter((x) => x.name.toLowerCase().includes(q) || x.id.includes(q));
    }
    return p;
  }, [registry.data, category, query]);

  return (
    <div>
      <PageHeader
        title="Pattern Explorer"
        subtitle={`${registry.data?.total ?? "—"} patterns in registry v${registry.data?.registry_version ?? "?"} — education + measured performance`}
      />
      <Page>
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          {CATEGORIES.map((c) => (
            <button
              key={c.key}
              onClick={() => setCategory(c.key)}
              className={cn(
                "rounded-md border px-2.5 py-1",
                category === c.key ? "bg-ink-200/60 font-medium dark:bg-ink-800" : "muted hover:bg-ink-100 dark:hover:bg-ink-800/60"
              )}
            >
              {c.label}
            </button>
          ))}
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search patterns…"
            className="ml-auto w-48 rounded-md border bg-[var(--panel)] px-2.5 py-1"
            aria-label="Search patterns"
          />
        </div>

        {registry.loading && !registry.data ? (
          <Spinner />
        ) : patterns.length === 0 ? (
          <EmptyState title="No patterns match" />
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {patterns.map((p) => (
              <PatternCard key={p.id} def={p} stats={latestStats?.[p.id] ?? null} />
            ))}
          </div>
        )}
      </Page>
    </div>
  );
}

function PatternCard({ def: d, stats }: { def: PatternDef; stats: Record<string, number> | null }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="panel px-4 py-3">
      <button className="w-full text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-medium">{d.name}</span>
          <DirectionBadge direction={d.direction} />
        </div>
        <div className="mt-1 flex items-center gap-2 text-2xs muted">
          <span>{d.min_bars === d.max_bars ? `${d.min_bars} bars` : `${d.min_bars}–${d.max_bars} bars`}</span>
          {d.supports_partial_detection ? <span>· partial tracking</span> : null}
          {!d.implemented ? <span className="text-amber-500">· not yet available</span> : null}
          {stats ? (
            <span className="ml-auto">
              hit rate {(stats.hit_rate * 100).toFixed(0)}% · n={stats.observations}
            </span>
          ) : null}
        </div>
      </button>
      {open ? (
        <div className="mt-2 space-y-2 border-t pt-2 text-xs">
          <p className="muted">{d.description}</p>
          {d.stages.length ? (
            <div>
              <div className="mb-1 text-2xs uppercase tracking-wide muted">Formation stages</div>
              <ol className="list-inside list-decimal space-y-0.5 muted">
                {d.stages.map((s) => (
                  <li key={s} className="capitalize">
                    {s.replace(/_/g, " ")}
                  </li>
                ))}
              </ol>
            </div>
          ) : null}
          {d.confirmation_rules.length ? (
            <div>
              <div className="mb-0.5 text-2xs uppercase tracking-wide muted">Confirmation</div>
              <ul className="space-y-0.5 muted">
                {d.confirmation_rules.map((r) => (
                  <li key={r}>✓ {r}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {d.invalidation_rules.length ? (
            <div>
              <div className="mb-0.5 text-2xs uppercase tracking-wide muted">Invalidation</div>
              <ul className="space-y-0.5 muted">
                {d.invalidation_rules.map((r) => (
                  <li key={r}>✗ {r}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {stats ? (
            <div className="rounded border px-2.5 py-1.5 font-mono text-2xs muted">
              backtest: {stats.observations} observations · hit rate {(stats.hit_rate * 100).toFixed(1)}% · median
              move {stats.median_return_pct?.toFixed(1)}% · PF {stats.profit_factor ?? "—"}
            </div>
          ) : (
            <div className="rounded border border-dashed px-2.5 py-1.5 text-2xs muted">
              no backtest statistics yet — run one from the Backtesting page
            </div>
          )}
          {d.confused_with.length ? (
            <div className="text-2xs muted">commonly confused with: {d.confused_with.join(", ")}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
