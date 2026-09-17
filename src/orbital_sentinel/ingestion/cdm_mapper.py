"""Map Space-Track CDM fields to ESA Kelvins format for inference."""

import numpy as np
import pandas as pd


SPACE_TRACK_TO_ESA = {
    "CDM_ID": "event_id",
    "TCA": "tca",
    "MISS_DISTANCE": "miss_distance",
    "RELATIVE_SPEED": "relative_speed",
    "RELATIVE_POSITION_R": "relative_position_r",
    "RELATIVE_POSITION_T": "relative_position_t",
    "RELATIVE_POSITION_N": "relative_position_n",
    "RELATIVE_VELOCITY_R": "relative_velocity_r",
    "RELATIVE_VELOCITY_T": "relative_velocity_t",
    "RELATIVE_VELOCITY_N": "relative_velocity_n",
    "OBJECT1_TIME_LASTOB_START": "t_time_lastob_start",
    "OBJECT1_TIME_LASTOB_END": "t_time_lastob_end",
    "OBJECT1_RECOMMENDED_OD_SPAN": "t_recommended_od_span",
    "OBJECT1_ACTUAL_OD_SPAN": "t_actual_od_span",
    "OBJECT1_OBS_AVAILABLE": "t_obs_available",
    "OBJECT1_OBS_USED": "t_obs_used",
    "OBJECT1_RESIDUALS_ACCEPTED": "t_residuals_accepted",
    "OBJECT1_WEIGHTED_RMS": "t_weighted_rms",
    "OBJECT1_RCS_SIZE": "t_rcs_estimate",
    "OBJECT1_CD_AREA_OVER_MASS": "t_cd_area_over_mass",
    "OBJECT1_CR_AREA_OVER_MASS": "t_cr_area_over_mass",
    "OBJECT1_SEDR": "t_sedr",
    "OBJECT1_SEMI_MAJOR_AXIS": "t_j2k_sma",
    "OBJECT1_ECCENTRICITY": "t_j2k_ecc",
    "OBJECT1_INCLINATION": "t_j2k_inc",
    "OBJECT1_CT_R": "t_ct_r",
    "OBJECT1_CN_R": "t_cn_r",
    "OBJECT1_CN_T": "t_cn_t",
    "OBJECT1_CRDOT_R": "t_crdot_r",
    "OBJECT1_CRDOT_T": "t_crdot_t",
    "OBJECT1_CRDOT_N": "t_crdot_n",
    "OBJECT1_CTDOT_R": "t_ctdot_r",
    "OBJECT1_CTDOT_T": "t_ctdot_t",
    "OBJECT1_CTDOT_N": "t_ctdot_n",
    "OBJECT1_CTDOT_RDOT": "t_ctdot_rdot",
    "OBJECT1_CNDOT_R": "t_cndot_r",
    "OBJECT1_CNDOT_T": "t_cndot_t",
    "OBJECT1_CNDOT_N": "t_cndot_n",
    "OBJECT1_CNDOT_RDOT": "t_cndot_rdot",
    "OBJECT1_CNDOT_TDOT": "t_cndot_tdot",
    "OBJECT2_OBJECT_TYPE": "c_object_type",
    "OBJECT2_TIME_LASTOB_START": "c_time_lastob_start",
    "OBJECT2_TIME_LASTOB_END": "c_time_lastob_end",
    "OBJECT2_RECOMMENDED_OD_SPAN": "c_recommended_od_span",
    "OBJECT2_ACTUAL_OD_SPAN": "c_actual_od_span",
    "OBJECT2_OBS_AVAILABLE": "c_obs_available",
    "OBJECT2_OBS_USED": "c_obs_used",
    "OBJECT2_RESIDUALS_ACCEPTED": "c_residuals_accepted",
    "OBJECT2_WEIGHTED_RMS": "c_weighted_rms",
    "OBJECT2_RCS_SIZE": "c_rcs_estimate",
    "OBJECT2_CD_AREA_OVER_MASS": "c_cd_area_over_mass",
    "OBJECT2_CR_AREA_OVER_MASS": "c_cr_area_over_mass",
    "OBJECT2_SEDR": "c_sedr",
    "OBJECT2_SEMI_MAJOR_AXIS": "c_j2k_sma",
    "OBJECT2_ECCENTRICITY": "c_j2k_ecc",
    "OBJECT2_INCLINATION": "c_j2k_inc",
    "OBJECT2_CT_R": "c_ct_r",
    "OBJECT2_CN_R": "c_cn_r",
    "OBJECT2_CN_T": "c_cn_t",
    "OBJECT2_CRDOT_R": "c_crdot_r",
    "OBJECT2_CRDOT_T": "c_crdot_t",
    "OBJECT2_CRDOT_N": "c_crdot_n",
    "OBJECT2_CTDOT_R": "c_ctdot_r",
    "OBJECT2_CTDOT_T": "c_ctdot_t",
    "OBJECT2_CTDOT_N": "c_ctdot_n",
    "OBJECT2_CTDOT_RDOT": "c_ctdot_rdot",
    "OBJECT2_CNDOT_R": "c_cndot_r",
    "OBJECT2_CNDOT_T": "c_cndot_t",
    "OBJECT2_CNDOT_N": "c_cndot_n",
    "OBJECT2_CNDOT_RDOT": "c_cndot_rdot",
    "OBJECT2_CNDOT_TDOT": "c_cndot_tdot",
}

RCS_SIZE_MAP = {"SMALL": 0.1, "MEDIUM": 1.0, "LARGE": 10.0}


def _safe_float(val, default=0.0):
    if val is None or val == "" or val == "N/A":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def map_cdm_to_esa_format(cdm: dict) -> dict:
    """Convert a single Space-Track CDM record to ESA Kelvins column format."""
    mapped = {}
    for st_key, esa_key in SPACE_TRACK_TO_ESA.items():
        val = cdm.get(st_key)
        if esa_key == "c_object_type":
            mapped[esa_key] = str(val) if val else "UNKNOWN"
        elif esa_key in ("event_id", "tca"):
            mapped[esa_key] = val
        else:
            mapped[esa_key] = _safe_float(val)

    for prefix in ("t_", "c_"):
        rcs_key = f"{prefix}rcs_estimate"
        if isinstance(mapped.get(rcs_key), str):
            mapped[rcs_key] = RCS_SIZE_MAP.get(mapped[rcs_key], 1.0)

    if "TCA" in cdm and "CREATION_DATE" in cdm:
        try:
            tca = pd.Timestamp(cdm["TCA"])
            created = pd.Timestamp(cdm["CREATION_DATE"])
            mapped["time_to_tca"] = (tca - created).total_seconds() / 86400.0
        except Exception:
            mapped["time_to_tca"] = 2.0
    else:
        mapped["time_to_tca"] = 2.0

    for prefix, sma_key, ecc_key in [("t_", "t_j2k_sma", "t_j2k_ecc"),
                                       ("c_", "c_j2k_sma", "c_j2k_ecc")]:
        sma = mapped.get(sma_key, 7000)
        ecc = mapped.get(ecc_key, 0.01)
        mapped[f"{prefix}h_apo"] = sma * (1 + ecc) - 6378.137
        mapped[f"{prefix}h_per"] = sma * (1 - ecc) - 6378.137
        mapped[f"{prefix}span"] = mapped.get(f"{prefix}actual_od_span", 0)

    mapped.setdefault("max_risk_estimate", -20.0)
    mapped.setdefault("max_risk_scaling", 1.0)
    mapped.setdefault("risk", np.nan)
    mapped.setdefault("geocentric_latitude", 0.0)
    mapped.setdefault("azimuth", 0.0)
    mapped.setdefault("elevation", 0.0)
    mapped.setdefault("mahalanobis_distance", 0.0)
    mapped.setdefault("t_position_covariance_det", 0.0)
    mapped.setdefault("c_position_covariance_det", 0.0)
    mapped.setdefault("F10", 150.0)
    mapped.setdefault("F3M", 150.0)
    mapped.setdefault("SSN", 50.0)
    mapped.setdefault("AP", 10.0)

    for prefix in ("t_", "c_"):
        for suffix in ("sigma_r", "sigma_t", "sigma_n",
                        "sigma_rdot", "sigma_tdot", "sigma_ndot"):
            mapped.setdefault(f"{prefix}{suffix}", 0.0)

    return mapped


def map_cdm_batch(cdm_records: list) -> pd.DataFrame:
    """Convert a list of Space-Track CDMs to an ESA-format DataFrame."""
    rows = [map_cdm_to_esa_format(r) for r in cdm_records]
    return pd.DataFrame(rows)
