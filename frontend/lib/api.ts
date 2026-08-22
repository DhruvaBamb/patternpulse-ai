/** Typed backend client. All fetches go through the Next.js rewrite to :8000. */

export interface Candle {
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketSymbol {
  symbol: string;
  name: string;
  market: string;
  last_price: number | null;
  change_pct: number | null;
  updated_at: string | null;
}

export interface Detection {
  id: string;
  symbol: string;
  interval: string;
  pattern_id: string;
  pattern: string;
  category: string;
  direction: string;
  status: string;
  completion: number;
  confidence: number;
  geometry_score: number;
  visual_score: number | null;
  source: string;
  stage: string;
  detected_breakout: boolean;
  levels: Record<string, number>;
  keypoints: Record<string, unknown>;
  lines: Record<string, unknown>;
  violations: string[];
  completion_history?: { ts: string; value: number }[];
  confidence_history?: { ts: string; value: number }[];
  first_seen: string | null;
  last_seen: string | null;
  events?: { event: string; detail: Record<string, unknown>; at: string }[];
  signal?: SignalPayload | null;
}

export interface SignalPayload {
  action: string;
  pattern: string;
  pattern_id: string;
  completion: number;
  confidence: number;
  entry: number | null;
  target: number | null;
  stop_loss: number | null;
  invalidation: number | null;
  risk_reward: number | null;
  reason: string;
  evidence: string[];
  warnings: string[];
}

export interface PatternDef {
  id: string;
  name: string;
  category: string;
  direction: string;
  description: string;
  min_bars: number;
  max_bars: number;
  supports_partial_detection: boolean;
  implemented: boolean;
  confirmation_rules: string[];
  invalidation_rules: string[];
  stages: string[];
  confused_with: string[];
  cv_class_id: number | null;
}

export interface RadarRow extends Detection {}

export interface BacktestRunInfo {
  id: string;
  status: string;
  params: Record<string, unknown>;
  overall?: Record<string, unknown>;
  pattern_stats?: Record<string, Record<string, number>>;
  created_at: string;
}

export interface WatchlistRow {
  symbol: string;
  last_price: number | null;
  change_pct: number | null;
  pattern: string | null;
  completion: number | null;
  confidence: number | null;
  interval: string | null;
  updated_at: string | null;
}

class ApiError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    let code = "http_error";
    let message = `request failed (${res.status})`;
    try {
      const body = await res.json();
      const detail = body.detail ?? body;
      code = detail.code ?? code;
      message = detail.message ?? detail.detail ?? message;
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(code, message);
  }
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new ApiError("http_error", `request failed (${res.status})`);
  return res.json();
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(path, { method: "DELETE" });
  if (!res.ok) throw new ApiError("http_error", `request failed (${res.status})`);
  return res.json();
}

export const api = {
  markets: () => get<{ symbols: MarketSymbol[] }>("/api/markets"),
  candles: (symbol: string, interval: string, lookback = 250) =>
    get<{ symbol: string; interval: string; candles: Candle[] }>(
      `/api/symbols/${encodeURIComponent(symbol)}/candles?interval=${interval}&lookback=${lookback}`
    ),
  symbolPatterns: (symbol: string, interval: string, fresh = false) =>
    get<{ symbol: string; interval: string; patterns: Detection[]; signal: SignalPayload | null; cv_status: { available: boolean; reason: string } }>(
      `/api/symbols/${encodeURIComponent(symbol)}/patterns?interval=${interval}${fresh ? "&fresh=true" : ""}`
    ),
  activePatterns: (params: Record<string, string | number | undefined> = {}) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") q.set(k, String(v));
    return get<{ count: number; patterns: Detection[] }>(`/api/patterns/active?${q}`);
  },
  detection: (id: string) => get<Detection>(`/api/patterns/${encodeURIComponent(id)}`),
  radar: (sort = "confidence") => get<{ rows: RadarRow[] }>(`/api/radar?sort=${sort}`),
  heatmap: () =>
    get<{ timeframes: string[]; symbols: string[]; cells: Record<string, { pattern: string; direction: string; confidence: number; completion: number; last_seen: string | null }> }>(
      "/api/heatmap"
    ),
  registry: (category?: string) =>
    get<{ registry_version: string; total: number; patterns: PatternDef[] }>(
      `/api/registry/patterns${category ? `?category=${category}` : ""}`
    ),
  patternDef: (id: string) => get<PatternDef>(`/api/registry/patterns/${id}`),
  backtests: () => get<{ runs: BacktestRunInfo[] }>("/api/backtests"),
  runBacktest: (symbols?: string[], interval?: string) =>
    post<{ status: string }>("/api/backtests/run", { symbols, interval }),
  models: () =>
    get<{ registry_version: string; runs: { run: string; has_meta: boolean; history_epochs: number }[] }>("/api/models"),
  trainingStatus: () => get<Record<string, unknown>>("/api/training/status"),
  health: () =>
    get<{
      status: string;
      app?: string;
      env?: string;
      scanner: { running?: boolean; last_scan_at?: string | null; last_scan_summary?: Record<string, unknown> };
      cv_detector: { available: boolean; reason?: string; device?: string };
      chatbot: { available: boolean; reason?: string | null };
    }>("/api/health"),
  watchlist: () => get<{ items: WatchlistRow[] }>("/api/watchlist"),
  addWatch: (symbol: string) => post<{ status: string }>("/api/watchlist", { symbol }),
  removeWatch: (symbol: string) => del<{ status: string }>(`/api/watchlist/${encodeURIComponent(symbol)}`),
  alerts: () => get<{ alerts: { id: number; symbol: string | null; pattern_id: string | null; condition: Record<string, unknown>; enabled: boolean; last_fired_at: string | null }[] }>("/api/alerts"),
  createAlert: (body: { symbol?: string | null; pattern_id?: string | null; condition: Record<string, unknown> }) =>
    post<{ id: number; status: string }>("/api/alerts", body),
  deleteAlert: (id: number) => del<{ status: string }>(`/api/alerts/${id}`),
  notifications: (onlyUnread = true) =>
    get<{ notifications: { id: number; title: string; body: string; created_at: string }[] }>(
      `/api/notifications?only_unread=${onlyUnread ? "true" : "false"}`
    ),
  chatStatus: () => get<{ available: boolean; model: string | null; reason: string | null }>("/api/chat/status"),
  confluence: (symbol: string, interval = "1d") =>
    get<{
      symbol: string;
      interval: string;
      direction?: string;
      score: number;
      factors: { factor: string; weight: number }[];
      patterns?: { pattern: string; direction: string; completion: number; confidence: number; status: string }[];
      note?: string;
    }>(`/api/symbols/${encodeURIComponent(symbol)}/confluence?interval=${interval}`),
  crossTimeframe: (symbol: string) =>
    get<{
      symbol: string;
      timeframes: Record<string, { summary: string; direction: string | null; strongest?: { pattern: string; completion: number; confidence: number; status: string } }>;
      context_score: number | null;
      alignment?: string;
      note?: string;
    }>(`/api/symbols/${encodeURIComponent(symbol)}/timeframes`),
  similarHistorical: (detectionId: string) =>
    get<{
      pattern: string;
      similar_found: number;
      similar: { id: string; symbol: string; interval: string; status: string; completion: number; confidence: number; similarity: number }[];
      confirmed: number;
      resolved_in_favour: number;
      note?: string;
    }>(`/api/patterns/${encodeURIComponent(detectionId)}/similar`),
};

export { ApiError };
