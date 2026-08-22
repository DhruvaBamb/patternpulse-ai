"""FastAPI application assembly."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .core.config import get_settings
from .core.logging import configure_logging, get_logger, new_request_id
from .db.session import init_db
from .services.alerts import AlertEngine
from .services.chatbot.assistant import PulseAssistant
from .services.detection import DetectionService
from .services.market_data import MarketDataService
from .workers.scanner import TaskQueue

configure_logging()
logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    app.state.market_data = MarketDataService()
    app.state.detection = DetectionService(app.state.market_data)
    app.state.assistant = PulseAssistant()
    app.state.alerts = AlertEngine()
    app.state.task_queue = TaskQueue()

    from .services.analysis import AnalysisService

    app.state.analysis = AnalysisService()
    app.state.drift_status = _drift_status_snapshot()

    if settings.app_env != "test":
        from .workers.scanner import ScannerWorker

        app.state.scanner = ScannerWorker(app.state.detection, app.state.alerts)
        app.state.scanner.start()
        logger.info("app_started", env=settings.app_env,
                    cv_available=app.state.detection.pipeline.detector.available,
                    chatbot_available=app.state.assistant.available)
    yield
    scanner = getattr(app.state, "scanner", None)
    if scanner:
        scanner.stop()
    logger.info("app_stopped")


def _drift_status_snapshot() -> dict:
    """Best-effort drift-reference snapshot for the status endpoint."""
    try:
        from ml.inference.drift import load_or_build_reference

        root = Path("artifacts/datasets")
        cands = sorted((p for p in root.iterdir() if (p / "data.yaml").exists()), key=lambda p: p.stat().st_mtime)
        if not cands:
            return {"configured": False, "note": "no dataset to build a reference from"}
        ref = load_or_build_reference(cands[-1])
        return {
            "configured": True,
            "reference": {"n_images": ref.n_images, "resolution": list(ref.resolution),
                          "image_mean": round(ref.image_mean, 1), "edge_density": round(ref.edge_density_mean, 4)},
            "note": "compare live windows via ml.inference.drift.monitor_window",
        }
    except Exception as exc:
        return {"configured": False, "note": f"reference unavailable: {exc}"}


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Real-Time Chart & Candlestick Pattern Detection, Formation Tracking & Explainable Signals",
        lifespan=lifespan,
    )
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        new_request_id()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "-")
        return response

    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error",
                     "message": "An unexpected error occurred. Check server logs.",
                     "detail": str(exc)},
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("backend.app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)


if __name__ == "__main__":
    run()
