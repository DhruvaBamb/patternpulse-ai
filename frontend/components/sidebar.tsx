"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Bell,
  Bot,
  BookOpen,
  CandlestickChart,
  Compass,
  Cpu,
  FlaskConical,
  Gauge,
  LineChart,
  ListChecks,
  Settings,
  Star,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./theme-toggle";

const NAV = [
  { href: "/", label: "Overview", icon: Gauge },
  { href: "/markets", label: "Markets", icon: BarChart3 },
  { href: "/radar", label: "Pattern Radar", icon: Compass },
  { href: "/active", label: "Active Patterns", icon: ListChecks },
  { href: "/explorer", label: "Pattern Explorer", icon: BookOpen },
  { href: "/backtesting", label: "Backtesting", icon: FlaskConical },
  { href: "/analytics", label: "Model Analytics", icon: Cpu },
  { href: "/assistant", label: "AI Assistant", icon: Bot },
  { href: "/watchlist", label: "Watchlist", icon: Star },
  { href: "/alerts", label: "Alerts", icon: Bell },
  { href: "/settings", label: "Settings", icon: Settings },
];

const SYMBOL_PAGES_NOTE = "Stock pages live under /symbol/[ticker]";

export function Sidebar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<Awaited<ReturnType<typeof api.health>> | null>(null);
  const [notes, setNotes] = useState(0);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await api.health();
        const n = await api.notifications();
        if (alive) {
          setHealth(h);
          setNotes(n.notifications.length);
        }
      } catch {
        if (alive) setHealth(null);
      }
    };
    poll();
    const t = setInterval(poll, 30000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  return (
    <aside className="sticky top-0 flex h-screen w-56 shrink-0 flex-col border-r" style={{ background: "var(--panel)" }}>
      <div className="flex items-center gap-2 px-4 py-4">
        <Activity size={16} className="text-bull" />
        <div>
          <div className="text-sm font-semibold leading-tight">PatternPulse</div>
          <div className="text-2xs muted">market pattern intelligence</div>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 dense-scroll" aria-label="Main">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] transition-colors",
                active
                  ? "bg-ink-200/50 font-medium dark:bg-ink-800"
                  : "muted hover:bg-ink-100 hover:text-inherit dark:hover:bg-ink-800/50"
              )}
            >
              <Icon size={15} strokeWidth={1.75} />
              <span className="flex-1">{label}</span>
              {href === "/alerts" && notes > 0 ? (
                <span className="rounded-full bg-amber-500/20 px-1.5 text-2xs text-amber-500">{notes}</span>
              ) : null}
            </Link>
          );
        })}
      </nav>

      <div className="border-t px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-2xs muted" title={SYMBOL_PAGES_NOTE}>
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                health ? (health.cv_detector?.available ? "bg-bull" : "bg-amber-500") : "bg-bear"
              )}
            />
            {health
              ? health.cv_detector?.available
                ? "CV + geometry"
                : "geometry-only"
              : "backend offline"}
          </div>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
