"""Pydantic schemas for the Orbital Sentinel API."""

from typing import Optional

from pydantic import BaseModel, Field


class CDMInput(BaseModel):
    """Input schema for a single Conjunction Data Message."""

    event_id: Optional[str] = None
    mission_id: Optional[str] = None
    time_to_tca: Optional[float] = None
    risk: Optional[float] = None
    max_risk_estimate: Optional[float] = None
    max_risk_scaling: Optional[float] = None
    miss_distance: Optional[float] = None
    relative_speed: Optional[float] = None
    relative_position_r: Optional[float] = None
    relative_position_t: Optional[float] = None
    relative_position_n: Optional[float] = None
    relative_velocity_r: Optional[float] = None
    relative_velocity_t: Optional[float] = None
    relative_velocity_n: Optional[float] = None
    t_time_lastob_start: Optional[float] = None
    t_time_lastob_end: Optional[float] = None
    t_recommended_od_span: Optional[float] = None
    t_actual_od_span: Optional[float] = None
    t_obs_available: Optional[float] = None
    t_obs_used: Optional[float] = None
    t_residuals_accepted: Optional[float] = None
    t_weighted_rms: Optional[float] = None
    t_rcs_estimate: Optional[float] = None
    t_cd_area_over_mass: Optional[float] = None
    t_cr_area_over_mass: Optional[float] = None
    t_sedr: Optional[float] = None
    t_j2k_sma: Optional[float] = None
    t_j2k_ecc: Optional[float] = None
    t_j2k_inc: Optional[float] = None
    t_ct_r: Optional[float] = None
    t_cn_r: Optional[float] = None
    t_cn_t: Optional[float] = None
    t_crdot_r: Optional[float] = None
    t_crdot_t: Optional[float] = None
    t_crdot_n: Optional[float] = None
    t_ctdot_r: Optional[float] = None
    t_ctdot_t: Optional[float] = None
    t_ctdot_n: Optional[float] = None
    t_ctdot_rdot: Optional[float] = None
    t_cndot_r: Optional[float] = None
    t_cndot_t: Optional[float] = None
    t_cndot_n: Optional[float] = None
    t_cndot_rdot: Optional[float] = None
    t_cndot_tdot: Optional[float] = None
    c_object_type: Optional[str] = None
    c_time_lastob_start: Optional[float] = None
    c_time_lastob_end: Optional[float] = None
    c_recommended_od_span: Optional[float] = None
    c_actual_od_span: Optional[float] = None
    c_obs_available: Optional[float] = None
    c_obs_used: Optional[float] = None
    c_residuals_accepted: Optional[float] = None
    c_weighted_rms: Optional[float] = None
    c_rcs_estimate: Optional[float] = None
    c_cd_area_over_mass: Optional[float] = None
    c_cr_area_over_mass: Optional[float] = None
    c_sedr: Optional[float] = None
    c_j2k_sma: Optional[float] = None
    c_j2k_ecc: Optional[float] = None
    c_j2k_inc: Optional[float] = None
    c_ct_r: Optional[float] = None
    c_cn_r: Optional[float] = None
    c_cn_t: Optional[float] = None
    c_crdot_r: Optional[float] = None
    c_crdot_t: Optional[float] = None
    c_crdot_n: Optional[float] = None
    c_ctdot_r: Optional[float] = None
    c_ctdot_t: Optional[float] = None
    c_ctdot_n: Optional[float] = None
    c_ctdot_rdot: Optional[float] = None
    c_cndot_r: Optional[float] = None
    c_cndot_t: Optional[float] = None
    c_cndot_n: Optional[float] = None
    c_cndot_rdot: Optional[float] = None
    c_cndot_tdot: Optional[float] = None
    t_span: Optional[float] = None
    c_span: Optional[float] = None
    t_h_apo: Optional[float] = None
    t_h_per: Optional[float] = None
    c_h_apo: Optional[float] = None
    c_h_per: Optional[float] = None
    geocentric_latitude: Optional[float] = None
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    mahalanobis_distance: Optional[float] = None
    t_position_covariance_det: Optional[float] = None
    c_position_covariance_det: Optional[float] = None
    t_sigma_r: Optional[float] = None
    c_sigma_r: Optional[float] = None
    t_sigma_t: Optional[float] = None
    c_sigma_t: Optional[float] = None
    t_sigma_n: Optional[float] = None
    c_sigma_n: Optional[float] = None
    t_sigma_rdot: Optional[float] = None
    c_sigma_rdot: Optional[float] = None
    t_sigma_tdot: Optional[float] = None
    c_sigma_tdot: Optional[float] = None
    t_sigma_ndot: Optional[float] = None
    c_sigma_ndot: Optional[float] = None
    F10: Optional[float] = None
    F3M: Optional[float] = None
    SSN: Optional[float] = None
    AP: Optional[float] = None

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    """Response from a single prediction."""

    prediction: float
    risk_category: str
    confidence: Optional[float] = None
    interval: Optional[tuple[float, float]] = None
    physics_result: Optional[dict] = None
    fusion: Optional[dict] = None
    explanation: Optional[str] = None
    top_features: Optional[list[dict]] = None


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    events: list[CDMInput]


class BatchPredictionResponse(BaseModel):
    """Response from batch predictions."""

    predictions: list[dict]
    total: int
    high_risk_count: int
    medium_risk_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    feature_count: Optional[int] = None
    uptime_seconds: float
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    feature_count: int
    feature_names: list[str]
    models_available: list[str]


class CDMFetchRequest(BaseModel):
    """Request to fetch live CDMs."""

    limit: int = Field(default=5, ge=1, le=50)
    run_predictions: bool = True


class CDMFetchResponse(BaseModel):
    """Response from CDM fetch."""

    cdms: list[dict]
    count: int
    predictions: Optional[list[dict]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str
