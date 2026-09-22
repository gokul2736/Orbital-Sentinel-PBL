"""FastAPI application factory for Orbital Sentinel."""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, cleanup on shutdown."""
    app.state.start_time = time.time()
    app.state.model_artifacts = None

    try:
        from orbital_sentinel.inference.pipeline import load_inference_artifacts
        artifacts = load_inference_artifacts()
        if artifacts.get("model") is not None:
            app.state.model_artifacts = artifacts
            logger.info("Model loaded: %s (%d features)",
                        artifacts["model_name"], len(artifacts.get("feature_names", [])))
        else:
            logger.warning("No trained model found in models_saved/")
    except Exception as e:
        logger.warning("Failed to load model artifacts: %s", e)

    yield

    app.state.model_artifacts = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Orbital Sentinel API",
        description="ML-powered satellite conjunction risk assessment service. "
                    "Predicts collision probabilities from Conjunction Data Messages (CDMs) "
                    "using ensemble ML models with physics-based verification.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
        )

    from orbital_sentinel.api.routes import health, predictions, cdm, models
    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(cdm.router)
    app.include_router(models.router)

    return app
