import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtPrice(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export function directionColor(direction?: string | null): string {
  if (direction === "bullish") return "text-bull";
  if (direction === "bearish") return "text-bear";
  return "muted";
}

export function timeAgo(iso?: string | null): string {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export const STATUS_STYLES: Record<string, string> = {
  EARLY: "bg-sky-500/10 text-sky-500",
  FORMING: "bg-amber-500/10 text-amber-500",
  NEAR_COMPLETION: "bg-violet-500/10 text-violet-400",
  CONFIRMED: "text-bull bg-[#26a69a]/10",
  FAILED: "bg-red-500/10 text-red-400",
  INVALIDATED: "bg-red-500/10 text-red-400",
  EXPIRED: "muted bg-opacity-10",
};

export function statusLabel(s: string): string {
  return s.replace(/_/g, " ").toLowerCase();
}
