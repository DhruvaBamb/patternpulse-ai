# Deployment

## Docker (recommended)

```bash
cp .env.example .env      # fill GROQ_API_KEY to enable PulseAI
docker compose up --build # backend :8000, frontend :3000
```

Optional PostgreSQL: `docker compose --profile postgres up`, then set

```
DATABASE_URL=postgresql+psycopg://patternpulse:patternpulse@postgres:5432/patternpulse
```

## Manual (dev machine)

```bash
# backend
uv venv .venv && uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
uvicorn backend.app.main:app --port 8000

# frontend
cd frontend && npm install && npm run dev
```

## Environment

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLite (default) or PostgreSQL |
| `GROQ_API_KEY` / `GROQ_MODEL` | PulseAI assistant (absent ⇒ visible "not configured" state) |
| `CORS_ORIGINS` | allowed origins for the API |
| `REDIS_URL` | optional future worker queue (in-process default) |
| `*_API_KEY` | future market-data providers (Kite, Polygon, …) |

Secrets live only in `.env` (git-ignored); `.env.example` is the template.

## ML artifacts

Train on the host (GPU recommended, MPS works):

```bash
python -m ml.datasets.synthetic.generate --samples 200
python -m ml.training.train --auto-resume
```

Then set `model.weights: artifacts/runs/<run>/checkpoints/best.pt` in
`configs/app.yaml` and restart — health/UI switch to "CV + geometry" mode.

## Operations notes

- The scanner runs inside the backend process; scale by sharding the
  universe across instances if needed (config-driven).
- SQLite file lives under `data/` (volume `pp-data` in compose); artifacts
  volume `pp-artifacts` keeps checkpoints/datasets.
- Health endpoint `/api/health` is the compose healthcheck.
