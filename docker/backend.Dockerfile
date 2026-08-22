FROM python:3.12-slim

WORKDIR /app

# system deps for pillow/opencv headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY ml ./ml
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .

# non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
