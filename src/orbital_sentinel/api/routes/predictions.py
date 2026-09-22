"""Prediction endpoints."""

import time

from fastapi import APIRouter, Depends, HTTPException
import pandas as pd

from orbital_sentinel.api.dependencies import get_model_artifacts
from orbital_sentinel.api.schemas import (
    CDMInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

router = APIRouter(prefix="/api/v1/predict", tags=["Predictions"])


@router.post("", response_model=PredictionResponse)
def predict_single(cdm: CDMInput, artifacts: dict = Depends(get_model_artifacts)) -> PredictionResponse:
    """Run full prediction on a single conjunction event.

    Includes physics verification, risk fusion, SHAP explanation, and uncertainty estimation.
    """
    from orbital_sentinel.inference.pipeline import predict_single_event

    row_dict = cdm.model_dump(exclude_none=True)

    try:
        result = predict_single_event(
            row_dict,
            artifacts["model"],
            artifacts["scaler"],
            artifacts["feature_names"],
            include_physics=True,
            include_explanation=True,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        prediction=result["prediction"],
        risk_category=result["risk_category"],
        confidence=result.get("confidence"),
        interval=result.get("interval"),
        physics_result=result.get("physics_result"),
        fusion=result.get("fusion"),
        explanation=result.get("explanation"),
        top_features=result.get("top_features"),
    )


@router.post("/quick", response_model=PredictionResponse)
def predict_quick(cdm: CDMInput, artifacts: dict = Depends(get_model_artifacts)) -> PredictionResponse:
    """Fast prediction without physics verification or SHAP explanation."""
    from orbital_sentinel.inference.pipeline import predict_single_event

    row_dict = cdm.model_dump(exclude_none=True)

    try:
        result = predict_single_event(
            row_dict,
            artifacts["model"],
            artifacts["scaler"],
            artifacts["feature_names"],
            include_physics=False,
            include_explanation=False,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        prediction=result["prediction"],
        risk_category=result["risk_category"],
        confidence=result.get("confidence"),
        interval=result.get("interval"),
    )


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, artifacts: dict = Depends(get_model_artifacts)) -> BatchPredictionResponse:
    """Run predictions on a batch of conjunction events."""
    from orbital_sentinel.inference.pipeline import predict_batch as run_batch

    rows = [cdm.model_dump(exclude_none=True) for cdm in request.events]
    df = pd.DataFrame(rows)

    try:
        results = run_batch(df, artifacts["model"], artifacts["scaler"], artifacts["feature_names"])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {e}")

    high_count = sum(1 for r in results if r["risk_category"] == "HIGH")
    medium_count = sum(1 for r in results if r["risk_category"] == "MEDIUM")

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        high_risk_count=high_count,
        medium_risk_count=medium_count,
    )
