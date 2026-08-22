# PROJECT_STATE.md — PatternPulse AI

Single source of truth for cross-session progress. Update after every work session.

- **Date**: 2026-08-22
- **Repo**: `/Users/dhruvasbamb/Desktop/Patterns` (git repo, branch `main`)
- **Machine**: macOS arm64 (Apple Silicon) — MPS available, no CUDA.

## Status snapshot (updated 2026-08-22 after full training cycle)

| Area | State | Evidence |
|---|---|---|
| Registry (77 patterns) | ✅ | `/api/registry/patterns` returns 77; integrity tests green |
| Market data + validation | ✅ | live yfinance scan, 0 fetch failures |
| Renderer | ✅ | determinism tests green |
| Geometry engine | ✅ | known-answer fixtures green |
| Fusion / completion / signals | ✅ | live scans produce staged formations + reasoned signals |
| Synthetic dataset | ✅ | **9,640 images** (6,427 verified positives + 3,213 negatives; 6,738/1,439/1,463 split); verify keep-rate 42% (8,973 rejected — honest) |
| Training (MPS, yolo11n, 25 epochs) | ✅ executed | losses: box 1.83→0.30, cls 5.94→1.28; weights at `artifacts/runs/pattern_detector_v001/checkpoints/best.pt` |
| Evaluation (test split) | ✅ executed | **mAP@0.5 = 0.469, mAP@0.5:0.95 = 0.452, P = 0.502, R = 0.608**; best classes (3-bar candlesticks) AP 0.99; worst (1-bar doji family) AP 0.0; `artifacts/eval/metrics.json` |
| System A/B/C comparison (§93) | ✅ executed | A CV-only P=0.807/R=0.688/F1=0.743; B geometry-only P=0.028/R=0.404; C production hybrid P=0.071/R=0.212 (latency 13/8/20 ms); `artifacts/eval/systems_comparison.json` |
| Partial-reveal experiment (§94) | ✅ executed | recognition accuracy rises 0.19→0.54 with reveal fraction; overall MACE 34.9%; `artifacts/eval/partial_reveal.json` |
| Hard-example mining (§42) | ✅ executed | 100 FPs exported to `artifacts/hard_examples/…/report.json` |
| ONNX export + benchmark (§44) | ✅ executed | best.pt.onnx; MPS 5.9 ms vs ONNX-CPU 13.4 ms mean; `artifacts/runs/.../export_report.json` |
| Backtesting (§45–47) | ✅ executed | 125 trades over 5 symbols daily; hit rate 0.376; per-pattern stats live in DB/API/Explorer/PulseAI |
| CV detector live | ✅ | `/api/health` → available, device mps, weights wired in `configs/app.yaml` |
| Backend + scanner + SSE | ✅ | live formations tracked across the universe |
| Frontend | ✅ | `next build` 14/14 routes; confluence/MTF/similar panels |
| PulseAI (Groq) | ✅ | tool-calling verified incl. backtest-stats answers |
| Tests / docs / Docker | ✅ | **60 pytest green**; docs/ complete; compose files |

## Current mode

**CV + geometry hybrid active** (trained weights loaded, device mps).
Important honest caveat: the detector was trained purely on synthetic
charts. On synthetic test images it fires (AP up to 0.99 for distinctive
candlesticks), but on **real market charts it currently produces zero
detections at conf≥0.05** — a pronounced synthetic→real domain gap (spec
§10 predicted this). The platform therefore still publishes geometry-only
findings on real data, correctly labelled, with the no-visual confidence
penalty. Closing the gap requires annotated real data (CVAT/Roboflow
workflow scaffolded in `ml/datasets/real/`).

## Known limitations (tracked, honest)

1. **Domain gap**: synthetic-only training does not transfer to real charts
   yet (see above). Real-data annotation is the top priority for the next
   iteration.
2. mAP 0.469 is a *first-cycle* number from 25 MPS epochs / 6.7k synthetic
   images; weak classes are visually tiny 1-bar patterns (doji family) —
   candidates for higher render resolution or a dedicated classifier.
3. Geometry-only B system emits multiple overlapping interpretations per
   window (drives its FP count in the A/B/C study); on real charts the
   production fusion gates + radar ranking mitigate the noise.
4. Synthetic verify keep-rate is 42% (validators stricter than generators).
5. The Groq model catalogue rotates; the assistant falls back through a
   model list (verified working).
6. Backtest hit rate 0.376 is from a single walk-forward pass of the
   geometry-only signal with pattern-level exits; treat as baseline, not
   an optimised strategy.

## Next development tasks

1. Annotate real charts (Roboflow/CVAT) and fine-tune → close domain gap.
2. Longer training / more epochs + higher-res crops for 1-bar patterns.
3. Re-run mining → retraining loop; re-run A/B/C after real-data blend.
4. Tune fusion weights against validation data (ml.evaluation.optimize).

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
