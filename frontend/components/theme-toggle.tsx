"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "./theme-provider";
import { cn } from "@/lib/utils";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const options = [
    { key: "light", icon: Sun, label: "Light" },
    { key: "system", icon: Monitor, label: "System" },
    { key: "dark", icon: Moon, label: "Dark" },
  ] as const;
  return (
    <div className="flex items-center rounded-md border p-0.5" role="group" aria-label="Theme">
      {options.map(({ key, icon: Icon, label }) => (
        <button
          key={key}
          onClick={() => setTheme(key)}
          title={label}
          aria-label={`${label} theme`}
          aria-pressed={theme === key}
          className={cn(
            "rounded p-1.5 transition-colors",
            theme === key ? "bg-ink-200/60 dark:bg-ink-800" : "hover:bg-ink-100 dark:hover:bg-ink-800/60"
          )}
        >
          <Icon size={13} className="muted" />
        </button>
      ))}
    </div>
  );
}
