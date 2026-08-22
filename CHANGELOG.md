# Changelog

All notable changes to PatternPulse AI. Newest first.

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
