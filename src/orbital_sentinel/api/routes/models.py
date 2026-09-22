"""Model management endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from orbital_sentinel.api.dependencies import get_model_artifacts
from orbital_sentinel.api.schemas import ModelInfoResponse

router = APIRouter(prefix="/api/v1/models", tags=["Models"])

MODELS_DIR = Path("models_saved")


@router.get("", response_model=list[str])
def list_models() -> list[str]:
    """List available trained models."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem.replace("_model", "") for p in MODELS_DIR.glob("*_model.joblib"))


@router.get("/info", response_model=ModelInfoResponse)
def model_info(artifacts: dict = Depends(get_model_artifacts)) -> ModelInfoResponse:
    """Get information about the currently loaded model."""
    available = sorted(p.stem.replace("_model", "") for p in MODELS_DIR.glob("*_model.joblib")) if MODELS_DIR.exists() else []

    return ModelInfoResponse(
        model_name=artifacts["model_name"],
        feature_count=len(artifacts["feature_names"]),
        feature_names=artifacts["feature_names"],
        models_available=available,
    )


@router.get("/features")
def feature_names(artifacts: dict = Depends(get_model_artifacts)) -> dict:
    """Get feature names used by the current model."""
    return {
        "feature_names": artifacts["feature_names"],
        "count": len(artifacts["feature_names"]),
    }


@router.get("/{model_name}")
def model_detail(model_name: str) -> dict:
    """Get details for a specific model."""
    model_path = MODELS_DIR / f"{model_name}_model.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    import os
    stat = model_path.stat()

    results_path = Path("proofs/pipeline_results.json")
    metrics = None
    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)
        metrics = results.get("model_metrics", {}).get(model_name)

    return {
        "model_name": model_name,
        "file_path": str(model_path),
        "file_size_mb": round(stat.st_size / 1024 / 1024, 2),
        "modified": stat.st_mtime,
        "metrics": metrics,
    }
