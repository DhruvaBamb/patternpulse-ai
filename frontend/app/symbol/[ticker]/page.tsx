"use client";

import { use, useState } from "react";
import Link from "next/link";
import { Bot, RefreshCw, Star } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { PageHeader, Page } from "@/components/page-header";
import { CandleChart, overlaysForDetection } from "@/components/candle-chart";
import { DetectionPanel } from "@/components/detection-panel";
import { DirectionBadge, ErrorState, Meter, Spinner, StatusBadge } from "@/components/ui";
import { cn } from "@/lib/utils";

const TIMEFRAMES = ["15m", "1h", "1d"] as const;

export default function SymbolPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = use(params);
  const [interval, setIntervalTf] = useState<string>("1d");
  const [fresh, setFresh] = useState(false);
  const [toggles, setToggles] = useState<Record<string, boolean>>({
    breakout: true,
    support: true,
    resistance: true,
    target: false,
    invalidation: false,
  });

  const candles = useApi(
    () => api.candles(ticker, interval, 250),
    [ticker, interval],
    60000
  );
  const patterns = useApi(
    () => api.symbolPatterns(ticker, interval, fresh),
    [ticker, interval, fresh],
    45000
  );

  const detections = patterns.data?.patterns ?? [];
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = detections.find((d) => d.id === selectedId) ?? detections[0] ?? null;
  const overlays = overlaysForDetection(selected, toggles);

  const watch = useApi(() => api.watchlist(), []);
  const watching = (watch.data?.items ?? []).some((w) => w.symbol === ticker);

  const reload = () => {
    setFresh(!fresh);
    patterns.reload();
    candles.reload();
  };

  return (
    <div>
      <PageHeader
        title={decodeURIComponent(ticker)}
        subtitle={`${detections.length} tracked formation${detections.length === 1 ? "" : "s"} · ${interval}`}
        actions={
          <>
            <button
              onClick={() => api.addWatch(ticker).then(() => watch.reload())}
              disabled={watching}
              className={cn(
                "flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs",
                watching ? "opacity-60" : "hover:bg-ink-100 dark:hover:bg-ink-800"
              )}
            >
              <Star size={12} className={watching ? "fill-amber-400 text-amber-400" : ""} />
              {watching ? "Watching" : "Watch"}
            </button>
            <Link
              href={`/assistant?q=${encodeURIComponent(`What patterns are forming in ${decodeURIComponent(ticker)}?`)}`}
              className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs hover:bg-ink-100 dark:hover:bg-ink-800"
            >
              <Bot size={12} /> Ask PulseAI
            </Link>
            <button
              onClick={reload}
              className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs hover:bg-ink-100 dark:hover:bg-ink-800"
            >
              <RefreshCw size={12} className={cn(patterns.loading && "animate-spin")} /> Rescan
            </button>
          </>
        }
      />
      <Page>
        <div className="mb-3 flex items-center gap-1 text-xs">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => setIntervalTf(tf)}
              className={cn(
                "rounded-md border px-2.5 py-1 font-mono",
                interval === tf ? "bg-ink-200/60 font-medium dark:bg-ink-800" : "muted hover:bg-ink-100 dark:hover:bg-ink-800/60"
              )}
            >
              {tf}
            </button>
          ))}
          {patterns.data && !patterns.data.cv_status.available ? (
            <span className="ml-2 text-2xs text-amber-500" title={patterns.data.cv_status.reason}>
              CV model not trained — geometry-only detection
            </span>
          ) : null}
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="panel lg:col-span-2">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-2">
              <span className="text-xs font-medium">Price · {interval}</span>
              <div className="flex flex-wrap gap-1">
                {(["breakout", "support", "resistance", "target", "invalidation"] as const).map((k) => (
                  <button
                    key={k}
                    onClick={() => setToggles((t) => ({ ...t, [k]: !t[k] }))}
                    className={cn(
                      "rounded border px-1.5 py-0.5 text-2xs capitalize",
                      toggles[k] ? "border-transparent bg-ink-200/60 dark:bg-ink-800" : "muted"
                    )}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>
            <div className="p-2">
              {candles.loading && !candles.data ? (
                <Spinner label="Loading candles…" />
              ) : candles.error ? (
                <ErrorState
                  title="Candles unavailable"
                  hint={candles.error === "Failed to fetch" ? "Backend not reachable." : candles.error}
                  onRetry={candles.reload}
                />
              ) : candles.data ? (
                <CandleChart candles={candles.data.candles} overlays={overlays} height={440} />
              ) : null}
            </div>
          </div>

          <div className="space-y-4">
            {detections.length === 0 ? (
              <div className="panel px-4 py-10 text-center">
                <div className="text-sm">No pattern currently tracked</div>
                <p className="mt-1 text-xs muted">
                  Nothing meets the confidence/completion gates on this timeframe. The scanner re-checks
                  automatically; you can also trigger a manual rescan.
                </p>
              </div>
            ) : (
              <>
                {detections.length > 1 ? (
                  <div className="flex flex-wrap gap-1">
                    {detections.map((d) => (
                      <button
                        key={d.id}
                        onClick={() => setSelectedId(d.id)}
                        className={cn(
                          "rounded border px-2 py-0.5 text-2xs",
                          (selected?.id ?? "") === d.id ? "border-transparent bg-ink-200/60 dark:bg-ink-800" : "muted"
                        )}
                      >
                        {d.pattern}
                      </button>
                    ))}
                  </div>
                ) : null}
                <DetectionPanel detection={selected} />
              </>
            )}

            {patterns.data?.signal ? (
              <div className="panel px-4 py-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">Latest signal</span>
                  <span
                    className={cn(
                      "font-mono text-sm font-semibold",
                      patterns.data.signal.action === "BUY"
                        ? "text-bull"
                        : patterns.data.signal.action === "SELL"
                          ? "text-bear"
                          : "muted"
                    )}
                  >
                    {patterns.data.signal.action}
                  </span>
                </div>
                <p className="mt-1.5 text-xs muted">{patterns.data.signal.reason}</p>
              </div>
            ) : null}
          </div>
        </div>
      </Page>
    </div>
  );
}
