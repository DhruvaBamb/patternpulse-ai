# Backend

FastAPI application in `backend/app/`.

## Run

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Lifespan: DB init → services (market data, detection, alerts, chatbot) →
scanner worker start (skipped when `APP_ENV=test`). SQLite by default;
PostgreSQL via `DATABASE_URL` (compose profile included).

## API surface

| Route | Purpose |
|---|---|
| `GET /api/health` | status: scanner, CV detector, chatbot |
| `GET /api/registry/patterns[/{id}]` | pattern registry (77 defs, versioned) |
| `GET /api/markets` | configured universe + quotes |
| `GET /api/symbols/{s}` | quote + active pattern summary |
| `GET /api/symbols/{s}/candles` | validated OHLCV |
| `GET /api/symbols/{s}/patterns?fresh=true` | scan now + tracked formations + latest signal |
| `GET /api/patterns/active` | all tracked formations (filters) |
| `GET /api/patterns/{id}` | formation detail: events, history, signal |
| `GET /api/radar` | radar rows (sorted) + scanner status |
| `GET /api/heatmap` | symbols × timeframes cells |
| `GET /api/backtests`, `POST /api/backtests/run` | backtest list / launch (queued) |
| `POST /api/chat`, `GET /api/chat/status` | PulseAI streaming chat |
| `GET /api/models`, `GET /api/training/status` | model/training info |
| `GET/POST/DELETE /api/watchlist…` | watchlist CRUD |
| `GET/POST/DELETE /api/alerts…` | alert CRUD |
| `GET /api/notifications`, `POST /api/notifications/read` | in-app feed |
| `GET /api/stream` | SSE: detections, alerts, scanner events |

Errors return typed payloads `{code, message, detail}` — the UI maps codes
(`market_data_unavailable`, `insufficient_candles`, `unknown_pattern`, …)
to informative states instead of raw 500s.

## Scanner worker

`backend/app/workers/scanner.py` — in-process thread on an interval from
`configs/app.yaml`; scans universe × timeframes outside HTTP handlers,
persists formations with stable IDs, expires stale ones, then evaluates
alerts. A `TaskQueue` interface abstracts submission for a future
Celery/Redis adapter.

## Formation lifecycle

`EARLY → FORMING → NEAR_COMPLETION → CONFIRMED` with `FAILED/INVALIDATED/
EXPIRED` exits. Completion/confidence histories accrue per scan;
`pattern_events` records first_seen, stage transitions, breakouts,
confirmations, expiries.

## Logging & security

structlog JSON with request-id/symbol/pattern context; secrets only via
environment (`.env.example` documents every key); CORS from settings; no
broker or LLM credentials ever reach the browser (chat goes through the
backend).
