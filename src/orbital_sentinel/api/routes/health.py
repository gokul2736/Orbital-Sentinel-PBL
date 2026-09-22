"""Health and status endpoints."""

import time

from fastapi import APIRouter, Request

from orbital_sentinel.api.schemas import HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Basic health check endpoint."""
    artifacts = getattr(request.app.state, "model_artifacts", None)
    model_loaded = artifacts is not None and artifacts.get("model") is not None
    uptime = time.time() - request.app.state.start_time

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=artifacts.get("model_name") if artifacts else None,
        feature_count=len(artifacts["feature_names"]) if artifacts and artifacts.get("feature_names") else None,
        uptime_seconds=round(uptime, 1),
        version="1.0.0",
    )


@router.get("/api/v1/status")
def system_status(request: Request) -> dict:
    """Detailed system status."""
    artifacts = getattr(request.app.state, "model_artifacts", None)
    model_loaded = artifacts is not None and artifacts.get("model") is not None
    uptime = time.time() - request.app.state.start_time

    from pathlib import Path
    models_dir = Path("models_saved")
    available = sorted(p.stem.replace("_model", "") for p in models_dir.glob("*_model.joblib")) if models_dir.exists() else []

    data_dir = Path("data/raw/esa_kelvins")
    dataset_exists = (data_dir / "train_data.csv").exists()

    return {
        "status": "operational",
        "uptime_seconds": round(uptime, 1),
        "model": {
            "loaded": model_loaded,
            "name": artifacts.get("model_name") if artifacts else None,
            "feature_count": len(artifacts["feature_names"]) if artifacts and artifacts.get("feature_names") else 0,
        },
        "models_available": available,
        "dataset_available": dataset_exists,
        "version": "1.0.0",
    }
