# Architecture

PatternPulse AI is a **hybrid pattern-intelligence platform**: a computer-vision
detector and an independent mathematical geometry engine both evaluate the same
market windows, and a fusion layer combines their evidence into calibrated,
completion-aware, explainable signals.

```
LIVE / HISTORICAL OHLCV (yfinance adapter; provider ABC for Kite/MT5/…)
        │
        ▼
Data validation (dedupe, OHLC sanity, gaps, timezone)
        │
        ▼
Chart Rendering Engine (deterministic, configs/rendering.yaml)
        │
        ├────────────────────┐
        ▼                    ▼
CV Detector (YOLO)     OHLC Geometry Engine
bbox + class + conf    pivots, trendlines, Fibonacci
        │                    │
        └────────┬───────────┘
                 ▼
          Pattern Fusion Engine (configurable weights)
                 │
     ┌───────────┼─────────────┐
     ▼           ▼             ▼
Completion   Confidence    Context analysis
estimator    calibration   (trend, volume, RSI-ready)
     └───────────┼─────────────┘
                 ▼
        Signal / Ranking Engine
                 │
   ┌─────────────┼──────────────┐
   ▼             ▼              ▼
Dashboard /   Pattern Radar   PulseAI (Groq,
Live charts   + heatmap      tool-calling agent)
```

## Layers

| Layer | Location | Responsibility |
|---|---|---|
| Registry | `ml/registry/` | 77 pattern definitions (YAML), versioning, cv class ids |
| Market data | `ml/datasets/providers/` | provider ABC + yfinance adapter (+ unconfigured future adapters) |
| Validation | `ml/datasets/validation.py` | OHLCV sanitisation + issue report |
| Rendering | `ml/datasets/rendering/` | deterministic candlestick images (train == inference) |
| Geometry | `ml/geometry/` | pivots, classical/candlestick/harmonic validators |
| Fusion | `ml/inference/fusion.py` | weighted evidence combination, penalties for disagreement/absence |
| Completion | `ml/completion/estimator.py` | per-pattern stage machines → formation % |
| Signals | `ml/inference/signals.py` | BUY/SELL/HOLD/WATCH + entry/target/stop/R:R + reasoning |
| Pipeline | `ml/inference/pipeline.py` | end-to-end analysis of one window |
| Training | `ml/training/` | YOLO training with full checkpoint/resume + signal safety |
| Dataset | `ml/datasets/synthetic/` | labelled synthetic generation (self-verified) |
| Evaluation | `ml/evaluation/` | mAP metrics, calibration (isotonic/Platt), experiments |
| Backtesting | `ml/backtesting/` | walk-forward outcome simulation + per-pattern stats |
| Backend | `backend/app/` | FastAPI, SQLAlchemy, scanner worker, SSE, alerts, chatbot |
| Frontend | `frontend/` | Next.js App Router UI (14 routes) |

## Key decisions

1. **Honest degradation.** Without trained CV weights the platform runs
   geometry-only and labels it everywhere (UI badge, API `source` field).
   Fused confidence is penalised (`no_visual_penalty`) because a single
   witness is weaker evidence than two independent ones.
2. **Registry as single source of truth.** Validators, dataset generators,
   CV class ids, the explorer UI and the chatbot all resolve patterns through
   the same registry (content-hashed version).
3. **Completion is structural, not cosmetic.** Validators report which
   formation stages exist; the estimator weights the final confirmation
   stage at 25% so nothing reads ~100% before its decisive event.
4. **No look-ahead anywhere.** Rendering windows, dataset truncation and the
   backtest engine only ever see bars up to the decision index.
5. **Workers without mandatory infrastructure.** The scanner runs on an
   in-process scheduler + `TaskQueue` interface; Celery/Redis can replace it
   later without touching call sites.
6. **Training recoverability.** Checkpoints persist weights, optimizer,
   scheduler, RNG states, history and environment fingerprint; SIGINT/SIGTERM
   write an emergency checkpoint before exit (spec §35–39).

## Realtime

`GET /api/stream` (SSE) fans out detection lifecycle events, alert fires and
scan summaries from an in-process `EventBus`. The scanner publishes after
every universe pass; formations keep stable IDs across passes
(`PATTERN-<SYM>-<TF>-<date>-<nnnn>`).
