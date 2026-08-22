"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface FetchState<T> {
  data: T | null;
  error: string | null;
  code: string | null;
  loading: boolean;
  reload: () => void;
}

export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = [], intervalMs?: number): FetchState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [code, setCode] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await fnRef.current();
      setData(d);
      setError(null);
      setCode(null);
    } catch (e) {
      const err = e as { message?: string; code?: string };
      setError(err.message ?? "request failed");
      setCode(err.code ?? "unknown");
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!intervalMs) return;
    const t = setInterval(load, intervalMs);
    return () => clearInterval(t);
  }, [load, intervalMs]);

  return { data, error, code, loading, reload: load };
}
