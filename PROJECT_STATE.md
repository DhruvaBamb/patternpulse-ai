# PROJECT_STATE.md — PatternPulse AI

Single source of truth for cross-session progress. Update after every work session.

- **Date**: 2026-08-22
- **Repo**: `/Users/dhruvasbamb/Desktop/Patterns` (git repo, branch `main`)
- **Machine**: macOS arm64 (Apple Silicon) — MPS available, no CUDA.

## Status snapshot (verified live 2026-08-22)

| Area | State | Evidence |
|---|---|---|
| Registry (77 patterns) | ✅ | `/api/registry/patterns` returns 77; integrity tests green |
| Market data + validation | ✅ | live yfinance scan, 0 fetch failures after fix |
| Renderer | ✅ | determinism tests green |
| Geometry engine | ✅ | known-answer fixtures (H&S, doubles, cup, engulfing, Gartley) green |
| Fusion / completion / signals | ✅ | AAPL live scan produced staged formations + reasoned BUY/WATCH |
| Synthetic generator | ✅ | 231 verified positives in smoke run; ~50% verify rate (honest, see below) |
| Training + checkpoint/resume | ✅ implemented | not yet executed end-to-end (needs a real training run) |
| Backend + scanner + SSE | ✅ | 255 live formations tracked across 17 symbols × 3 timeframes |
| Frontend | ✅ | `next build` — 14/14 routes compile |
| PulseAI (Groq) | ✅ | live tool-calling verified (model fallback works) |
| Backtesting | ✅ implemented | engine + UI + queue; run it from /backtesting |
| Tests / docs / Docker | ✅ | 51 pytest green; docs/ complete; compose files written |

## Current mode

**Geometry-only** (no trained CV weights). The UI/API label this honestly;
fused confidence carries the no-visual penalty. Train the detector to
activate the full CV+geometry pipeline.

## Known limitations (tracked, honest)

1. Synthetic verify pass keeps ~50% of generated positives (validators are
   stricter than the idealised generators). Kept positives are guaranteed
   detectable; improve generator fidelity for higher yield.
2. Geometry-only mode surfaces several overlapping interpretations per
   window (e.g. Rounded Top + H&S on the same bars). CV training + fusion
   disagreement penalties are the designed mitigations.
3. Training pipeline is implemented but no full run has been executed yet —
   mAP numbers do not exist and are not claimed anywhere.
4. Real-data annotation workflow (CVAT/Roboflow) is scaffolded, not wired.
5. The Groq model catalogue rotates; the assistant falls back through a
   model list (verified working).

## Next development tasks

1. `python -m ml.datasets.synthetic.generate --samples 300` then
   `python -m ml.training.train --auto-resume` on this Mac (MPS).
2. Evaluate mAP; iterate on generator fidelity for weak classes.
3. Run backtests from the UI to populate per-pattern statistics.
4. Wire the real-data annotation import (`ml/datasets/real/`).

## Architecture decisions

1. **Hybrid detection**: YOLO proposes candidates from rendered chart images; an independent
   OHLC geometry engine scores the same window; a fusion engine combines both (weights in
   `configs/app.yaml`). If no trained weights exist the CV branch reports "unavailable" and
   geometry-only detections are published above `fusion.geometry_only_floor`. No fake values.
2. **Registry-driven**: every pattern (all ~78) lives in `ml/registry/patterns/*.yaml`.
   The `implemented` flag marks which have working validators; the UI labels the rest
   "Not yet available" instead of pretending.
3. **SQLite default DB** (`DATABASE_URL`), SQLAlchemy 2.0 — swap to PostgreSQL by env var.
4. **Background scanning** runs in an in-process APScheduler worker (no Redis dependency);
   a `TaskQueue` interface exists so Celery/Redis can be dropped in later.
5. **Training resume**: custom ultralytics `Trainer` saves full state (epoch, optimizer,
   scheduler, RNG, EMA, history, config, dataset version) each epoch + on SIGINT/SIGTERM.
   `python -m ml.training.train --auto-resume` continues exactly where a run stopped.
6. **Deterministic rendering**: identical `configs/rendering.yaml` used for synthetic
   generation, training, evaluation and live inference.
7. **No future leakage**: synthetic windows are cut at pattern-relative indices; real-data
   windows end at the labelled bar; splits are chronological.

## Key paths

- Registry: `ml/registry/` (loader `registry.py`, definitions `patterns/*.yaml`)
- Geometry: `ml/geometry/` (pivots, classical validators, candlestick rules, harmonics)
- Rendering: `ml/datasets/rendering/renderer.py`
- Training: `ml/training/train.py` (+ `checkpointing.py`)
- Detection pipeline: `ml/inference/pipeline.py`
- Scanner worker: `backend/app/workers/scanner.py`
- Chatbot: `backend/app/services/chatbot/`
- Frontend: `frontend/` (Next.js App Router, Tailwind, shadcn/ui, lightweight-charts)

## Commands

```bash
# Backend
uv run uvicorn backend.app.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev

# Tests
uv run pytest

# ML
uv run python -m ml.datasets.synthetic.generate --samples 200
uv run python -m ml.training.train --auto-resume
uv run python -m ml.evaluation.evaluate
uv run python -m ml.backtesting.run

# Seed market data cache
uv run python -m scripts.seed_market_data
```

## Model status

- No trained weights committed (too large for git). Train locally via command above.
- Without weights the platform runs in geometry-only mode — honestly labelled in the UI.

## Next development tasks

1. Train detector once synthetic dataset is generated; evaluate mAP.
2. Add real-data annotation workflow (CVAT/Roboflow export in `ml/datasets/real/`).
3. Optional: Celery adapter, ONNX export benchmark (`ml/export/`).
