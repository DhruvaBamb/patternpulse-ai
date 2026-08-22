# Frontend

Next.js 15 (App Router) + TypeScript + Tailwind + lightweight-charts.

```bash
cd frontend && npm install && npm run dev    # http://localhost:3000
npm run build                                # production build
```

The dev/proxy rewrites `/api/*` to the backend (`BACKEND_URL`, default
`:8000`), so no CORS juggling during development.

## Routes (14)

| Route | Content |
|---|---|
| `/` | Overview: headline stats, heatmap, strongest formations, markets table |
| `/markets` | universe with live quotes |
| `/radar` | Pattern Radar: sortable/filterable formation table |
| `/symbol/[ticker]` | candlestick chart + toggleable overlays (breakout/support/resistance/target/invalidation), detection panel, signal, timeframe switch, watch + Ask PulseAI |
| `/active` | all tracked formations |
| `/active/[id]` | formation detail: evolution chart, lifecycle events, keypoints, full signal reasoning |
| `/explorer` | registry-driven education cards with backtest stats |
| `/backtesting` | launch runs, per-pattern statistics tables |
| `/analytics` | detector/training status, evaluation workflow |
| `/assistant` | PulseAI streaming chat with suggested questions |
| `/watchlist` | saved symbols + quick add |
| `/alerts` | alert builder + notification feed |
| `/settings` | system status overview |

## Design language

Dense-but-readable financial product: 8px spacing rhythm, tabular numerals,
restrained palette (single bull/bear accent pair), no decorative gradients
or glassmorphism. Dark / Light / System themes via a class on `<html>` +
CSS variables. Every data surface has explicit loading, empty and error
states (backend error codes map to actionable hints). Financial-safety
language appears on Overview ("research-oriented technical analysis — not
investment advice") and never claims guaranteed outcomes.

## Data layer

`lib/api.ts` — typed client for every backend route. `hooks/useApi.ts` —
fetch state machine (loading/error/code) with polling for live surfaces.
Components favour plain tables + compact panels over card grids.
