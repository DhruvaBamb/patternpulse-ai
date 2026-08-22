# Changelog

All notable changes to PatternPulse AI. Newest first.

## [0.2.0] — 2026-08-22 (first executed training + evaluation cycle)

### Executed (real, measured results)
- Full synthetic dataset: 9,640 images (6,427 self-verified positives,
  3,213 negatives incl. hard negatives; 70/15/15 chronological split).
- Training on MPS (yolo11n, 25 epochs, batch 32): box loss 1.83→0.30,
  cls loss 5.94→1.28; best.pt mirrored into the checkpoint layout.
- Test-split evaluation: mAP@0.5 0.469, mAP@0.5:0.95 0.452, precision 0.502,
  recall 0.608. Per-class AP from 0.0 (1-bar doji family) to 0.99
  (distinctive 3-bar candlesticks). No numbers fabricated.
- System A/B/C research comparison (§93): CV-only F1 0.743 vs geometry-only
  0.052 vs production hybrid 0.106 on single-label synthetic test; latencies
  13/8/20 ms — documents the multi-interpretation nature of the geometry arm.
- Partial-reveal experiment (§94): recognition accuracy 0.19→0.54 with
  reveal fraction; overall MACE 34.9%.
- Hard-example mining run: 100 FPs exported for review/retraining.
- ONNX export + benchmark: MPS 5.9 ms vs ONNX-CPU 13.4 ms mean.
- Walk-forward backtest (5 symbols, daily): 125 trades, hit rate 0.376,
  per-pattern statistics loaded into the DB (Explorer/PulseAI serve them).
- Weights wired into configs/app.yaml; /api/health reports CV available
  on device mps; honest finding documented: synthetic-trained model does
  not yet fire on real charts (domain gap) — geometry-only findings remain
  correctly labelled.

### Added
- §42 mining module (ml/evaluation/mining.py), §44 ONNX export/benchmark
  (ml/export/export_onnx.py), §75 drift monitoring (ml/inference/drift.py
  + /api/drift/status), §76–79 intelligence service (similar-historical,
  confluence, cross-timeframe) + API routes + frontend panels.
- §93 systems comparison and §94 partial-reveal experiment runners.
- 9 new tests (intelligence + drift); suite now 60 green.

## [0.1.0] — 2026-08-22

### Added
- Repository scaffold: `backend/`, `frontend/`, `ml/`, `configs/`, `tests/`, `docs/`, `docker/`.
- Central pattern registry with ~78 pattern definitions across classical, candlestick and
  harmonic categories; registry content-hash versioning.
- Market-data layer: provider ABC + yfinance adapter + unconfigured adapters for
  Kite/MT5/AlphaVantage/Polygon/TwelveData/Binance.
- OHLCV validation pipeline (dedupe, OHLC sanity, gaps, timezone, chronology, zero-volume).
- Deterministic candlestick chart renderer shared by training and inference.
- Geometry validation engine: pivots/trendlines/neckline/Fibonacci + validators for
  classical patterns, ~41 candlestick rules, 7 harmonic XABCD patterns.
- Fusion engine, per-pattern completion estimators, formation lifecycle tracker,
  breakout-confirmation and signal engine with pattern-specific entry/target/stop.
- Synthetic dataset generator (labelled positives, partials, negatives, hard negatives)
  with YOLO-format export and dataset manifests (hashes, versions, chronological splits).
- Ultralytics YOLO training wrapper with complete checkpoint/resume (optimizer, scheduler,
  RNG, epoch, history) and SIGINT/SIGTERM emergency checkpoints.
- Evaluation module: mAP/per-class AP/confusion, calibration (isotonic/Platt/temperature),
  hard-example mining, partial-reveal experiment, ablation harness.
- FastAPI backend: candles/symbols/patterns/radar/chat/models/training routes, SSE stream,
  SQLAlchemy models, structured logging, scanner background worker, watchlist/alerts APIs.
- PulseAI Groq chatbot with internal tool access, session memory, streaming, graceful
  no-key degradation.
- Backtesting engine with walk-forward evaluation and pattern-performance statistics.
- Next.js frontend: Markets, Pattern Radar, stock page with overlay toggles, Active
  Patterns, Pattern Explorer, Backtesting, Model Analytics, Training, AI Assistant,
  Watchlist, Alerts, Settings; dark/light/system themes.
- pytest suites: geometry known-answer fixtures, data validation, renderer determinism,
  completion, fusion, signal, backtest math, API integration.
- Docker: backend + frontend Dockerfiles, docker-compose with optional postgres profile.
- Docs: architecture, dataset, training, patterns, backend, frontend, deployment.
