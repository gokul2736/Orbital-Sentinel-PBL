"""FastAPI dependency injection for model artifacts and settings."""

from typing import Optional

from fastapi import Request, HTTPException


def get_model_artifacts(request: Request) -> dict:
    """Retrieve loaded model artifacts from app state."""
    artifacts = getattr(request.app.state, "model_artifacts", None)
    if artifacts is None or artifacts.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )
    return artifacts


def get_optional_model_artifacts(request: Request) -> Optional[dict]:
    """Retrieve model artifacts if available, None otherwise."""
    artifacts = getattr(request.app.state, "model_artifacts", None)
    if artifacts is None or artifacts.get("model") is None:
        return None
    return artifacts
