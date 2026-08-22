"use client";

import { cn } from "@/lib/utils";

export function Badge({
  children,
  className,
  tone,
}: {
  children: React.ReactNode;
  className?: string;
  tone?: "bull" | "bear" | "neutral" | "warn" | "info";
}) {
  const tones = {
    bull: "text-bull bg-[#26a69a]/10",
    bear: "text-bear bg-[#ef5350]/10",
    neutral: "muted bg-ink-200/40 dark:bg-ink-800",
    warn: "text-amber-500 bg-amber-500/10",
    info: "text-sky-500 bg-sky-500/10",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-1.5 py-0.5 text-2xs font-medium uppercase tracking-wide",
        tone ? tones[tone] : tones.neutral,
        className
      )}
    >
      {children}
    </span>
  );
}

export function DirectionBadge({ direction }: { direction?: string | null }) {
  if (direction === "bullish") return <Badge tone="bull">bullish</Badge>;
  if (direction === "bearish") return <Badge tone="bear">bearish</Badge>;
  return <Badge>neutral</Badge>;
}

export function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    EARLY: "text-sky-500 bg-sky-500/10",
    FORMING: "text-amber-500 bg-amber-500/10",
    NEAR_COMPLETION: "text-violet-400 bg-violet-500/10",
    CONFIRMED: "text-bull bg-[#26a69a]/10",
    FAILED: "text-bear bg-[#ef5350]/10",
    INVALIDATED: "text-bear bg-[#ef5350]/10",
    EXPIRED: "muted bg-ink-200/40 dark:bg-ink-800",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-1.5 py-0.5 text-2xs font-medium uppercase tracking-wide",
        styles[status] ?? styles.EXPIRED
      )}
    >
      {status.replace(/_/g, " ").toLowerCase()}
    </span>
  );
}

export function Meter({ value, label, tone }: { value: number; label: string; tone?: string }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs muted">{label}</span>
        <span className="font-mono text-xs tabular-nums">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-200/60 dark:bg-ink-800">
        <div
          className={cn("h-full rounded-full transition-all duration-500", tone ?? "bg-ink-400 dark:bg-ink-300")}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="panel px-4 py-3">
      <div className="text-2xs uppercase tracking-wider muted">{label}</div>
      <div className="mt-1 font-mono text-lg tabular-nums">{value}</div>
      {sub ? <div className="mt-0.5 text-xs muted">{sub}</div> : null}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-12 muted text-sm">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
      {label ?? "Loading…"}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1 py-12 text-center">
      <div className="text-sm font-medium">{title}</div>
      {hint ? <div className="max-w-sm text-xs muted">{hint}</div> : null}
    </div>
  );
}

export function ErrorState({ title, hint, onRetry }: { title: string; hint?: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      <div className="text-sm font-medium text-bear">{title}</div>
      {hint ? <div className="max-w-md text-xs muted">{hint}</div> : null}
      {onRetry ? (
        <button
          onClick={onRetry}
          className="mt-1 rounded border px-3 py-1 text-xs hover:bg-ink-100 dark:hover:bg-ink-800"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
