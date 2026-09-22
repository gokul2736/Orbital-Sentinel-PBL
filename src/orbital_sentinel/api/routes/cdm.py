"""CDM data ingestion and analysis endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request

from orbital_sentinel.api.dependencies import get_optional_model_artifacts
from orbital_sentinel.api.schemas import CDMFetchRequest, CDMFetchResponse, CDMInput, PredictionResponse

router = APIRouter(prefix="/api/v1/cdm", tags=["CDM Data"])


@router.post("/fetch", response_model=CDMFetchResponse)
def fetch_live_cdms(request_body: CDMFetchRequest, request: Request) -> CDMFetchResponse:
    """Fetch live CDMs from Space-Track.org and optionally run predictions."""
    try:
        from orbital_sentinel.ingestion.cdm_api import SpaceTrackClient
        from orbital_sentinel.ingestion.cdm_mapper import map_cdm_to_esa_format
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Ingestion module not available: {e}")

    try:
        client = SpaceTrackClient()
        client.login()
        raw_cdms = client.fetch_cdm(limit=request_body.limit)
        client.close()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Space-Track fetch failed: {e}")

    mapped_cdms = []
    for cdm in raw_cdms:
        mapped = map_cdm_to_esa_format(cdm)
        mapped_cdms.append({
            "raw": {k: cdm.get(k) for k in ["CDM_ID", "TCA", "MISS_DISTANCE", "RELATIVE_SPEED",
                                              "SAT_1_NAME", "SAT_2_NAME", "CREATION_DATE"]},
            "mapped": mapped,
        })

    predictions = None
    if request_body.run_predictions:
        artifacts = get_optional_model_artifacts(request)
        if artifacts:
            from orbital_sentinel.inference.pipeline import predict_single_event
            predictions = []
            for cdm_data in mapped_cdms:
                try:
                    result = predict_single_event(
                        cdm_data["mapped"],
                        artifacts["model"],
                        artifacts["scaler"],
                        artifacts["feature_names"],
                        include_physics=True,
                        include_explanation=False,
                    )
                    predictions.append({
                        "prediction": result["prediction"],
                        "risk_category": result["risk_category"],
                        "confidence": result.get("confidence"),
                    })
                except Exception:
                    predictions.append({"prediction": None, "risk_category": "ERROR", "confidence": None})

    return CDMFetchResponse(
        cdms=mapped_cdms,
        count=len(mapped_cdms),
        predictions=predictions,
    )


@router.get("/sample")
def sample_data() -> dict:
    """Return a sample from the ESA Kelvins dataset for testing."""
    try:
        from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
        df = load_esa_kelvins(split="train")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset not available: {e}")

    sample = df.sample(n=min(5, len(df)), random_state=42)
    return {
        "samples": sample.to_dict(orient="records"),
        "total_rows": len(df),
        "columns": list(df.columns),
    }


@router.post("/analyze", response_model=PredictionResponse)
def analyze_cdm(cdm: CDMInput, request: Request) -> PredictionResponse:
    """Full analysis of a CDM: prediction + physics + fusion + explanation."""
    artifacts = get_optional_model_artifacts(request)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded for analysis")

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
        raise HTTPException(status_code=422, detail=f"Analysis failed: {e}")

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
