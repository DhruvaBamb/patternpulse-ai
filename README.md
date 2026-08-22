# PatternPulse AI

**Real-Time Chart & Candlestick Pattern Detection, Formation Tracking, AI Market Intelligence & Explainable Signal Platform**

PatternPulse AI detects ~78 classical, candlestick and harmonic chart patterns on live and historical OHLCV data using a **hybrid architecture**: a YOLO computer-vision detector proposes patterns from rendered candlestick images while an independent OHLC geometry engine validates the underlying structure. A fusion engine combines both, estimates how *complete* a forming pattern is, and produces explainable BUY/SELL/HOLD/WATCH signals with entry/target/stop levels.

> Research-oriented technical analysis. Not investment advice, no guaranteed outcomes.

## Quick start

```bash
git clone <repo> && cd patternpulse-ai
cp .env.example .env          # add GROQ_API_KEY to enable the PulseAI assistant

# backend
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

Docker: `docker compose up` (backend + frontend; optional postgres profile).

## ML workflow

```bash
python -m ml.datasets.synthetic.generate --samples 200   # labelled synthetic dataset
python -m ml.training.train --auto-resume                # checkpoint/resume training
python -m ml.evaluation.evaluate --weights artifacts/runs/<run>/best.pt
python -m ml.backtesting.run                             # walk-forward pattern stats
python -m scripts.seed_market_data                       # cache real candles
```

## Architecture

```
OHLCV → Chart Renderer ─┬─ CV Detector (YOLO) ──┐
                         │                        ├─ Fusion → Completion → Signal
                         └─ Geometry Validator ───┘         │
                                    Radar / Dashboard / PulseAI (Groq)
```

- **Registry-driven**: every pattern lives in `ml/registry/patterns/*.yaml`.
- **Honest labelling**: without trained CV weights the platform runs in geometry-only mode and says so.
- **No look-ahead**: training windows, backtests and signals only use past candles.

## Docs

`docs/architecture.md` · `docs/dataset.md` · `docs/training.md` · `docs/patterns.md` · `docs/backend.md` · `docs/frontend.md` · `docs/deployment.md` · `PROJECT_STATE.md`

## Tests

```bash
pytest                      # geometry known-answer, validation, pipeline, API
```
