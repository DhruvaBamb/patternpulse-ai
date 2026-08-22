"use client";

import Link from "next/link";
import type { Detection } from "@/lib/api";
import { DirectionBadge, Meter, StatusBadge } from "./ui";
import { fmtPrice, timeAgo } from "@/lib/utils";

export function DetectionPanel({ detection: d }: { detection: Detection | null }) {
  if (!d) return null;
  const levels = d.levels ?? {};
  const levelRows: { label: string; keys: string[] }[] = [
    { label: "Potential breakout", keys: ["breakout", "neckline", "rim"] },
    { label: "Target", keys: ["target"] },
    { label: "Support", keys: ["support", "lower", "basin"] },
    { label: "Resistance", keys: ["resistance", "upper"] },
    { label: "Invalidation", keys: ["invalidation"] },
  ];

  return (
    <div className="panel px-4 py-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold">{d.pattern}</div>
          <div className="mt-1 flex items-center gap-1.5">
            <DirectionBadge direction={d.direction} />
            <StatusBadge status={d.status} />
            <span className="text-2xs muted">{d.interval}</span>
          </div>
        </div>
        <Link href={`/active/${d.id}`} className="text-2xs muted hover:underline">
          details →
        </Link>
      </div>

      <div className="mt-3 space-y-2.5">
        <Meter value={d.completion} label="Completion" tone="bg-amber-500" />
        <Meter value={d.confidence * 100} label="AI confidence (fused)" tone="bg-bull" />
        <Meter value={d.geometry_score * 100} label="Geometry score" tone="bg-sky-500" />
      </div>

      <dl className="mt-3 space-y-1 border-t pt-2.5 text-xs">
        {levelRows.map(({ label, keys }) => {
          const found = keys.find((k) => typeof levels[k] === "number");
          if (found === undefined) return null;
          return (
            <div key={label} className="flex justify-between">
              <dt className="muted">{label}</dt>
              <dd className="font-mono tabular-nums">{fmtPrice(levels[found])}</dd>
            </div>
          );
        })}
        <div className="flex justify-between">
          <dt className="muted">First seen</dt>
          <dd className="font-mono">{timeAgo(d.first_seen)}</dd>
        </div>
      </dl>

      {d.detected_breakout ? (
        <div className="mt-2 rounded bg-[#26a69a]/10 px-2 py-1 text-2xs text-bull">
          breakout registered — watching for confirmation follow-through
        </div>
      ) : (
        <div className="mt-2 rounded bg-amber-500/10 px-2 py-1 text-2xs text-amber-600 dark:text-amber-400">
          awaiting breakout — pattern not yet confirmed
        </div>
      )}
      {d.violations?.length ? (
        <div className="mt-2 text-2xs muted">notes: {d.violations.slice(0, 2).join("; ")}</div>
      ) : null}
    </div>
  );
}
