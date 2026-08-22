# PROJECT_STATE.md — PatternPulse AI

Single source of truth for cross-session progress. Update after every work session.

- **Date**: 2026-08-22
- **Repo**: `/Users/dhruvasbamb/Desktop/Patterns` (git repo, branch `main`)
- **Machine**: macOS arm64 (Apple Silicon) — MPS available, no CUDA.

## Status snapshot

| Area | State |
|---|---|
| Repo scaffold, configs, gitignore, env | ✅ Done |
| Pattern registry (~78 defs) | ✅ Done — `ml/registry/` |
| Market data (yfinance) + validation | ✅ Done — `ml/datasets/providers/`, `ml/datasets/validation.py` |
| Chart renderer | ✅ Done — `ml/datasets/rendering/` |
| Geometry engine (classical + candlestick + harmonic) | ✅ Done — `ml/geometry/` |
| Fusion / completion / signal engine | ✅ Done — `ml/inference/fusion.py`, `ml/completion/`, `backend/app/services/signals.py` |
| Synthetic dataset generator | ✅ Done — `ml/datasets/synthetic/` |
| YOLO training + checkpoint/resume | ✅ Done — `ml/training/` |
| Backend API + DB + scanner + SSE | ✅ Done — `backend/app/` |
| Frontend (Next.js) | ✅ Done — `frontend/` |
| PulseAI (Groq) chatbot | ✅ Done — `backend/app/services/chatbot/` |
| Backtesting engine | ✅ Done — `ml/backtesting/` |
| Tests / docs / Docker | ✅ Done |

> Update the table above as work proceeds; details in CHANGELOG.md.

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
