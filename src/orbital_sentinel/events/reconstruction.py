"""Reconstruct conjunction event timelines from CDM sequences."""

import numpy as np


def reconstruct_timeline(event_cdms: list) -> dict:
    """Build a timeline from a list of CDMs for one event.

    Each CDM dict expected to have at minimum: time_to_tca, risk, miss_distance.
    """
    if not event_cdms:
        return {
            "first_cdm_time": None,
            "last_cdm_time": None,
            "total_cdms": 0,
            "risk_evolution": [],
            "miss_distance_evolution": [],
            "convergence_assessment": "no_data",
        }

    sorted_cdms = sorted(
        event_cdms, key=lambda x: x.get("time_to_tca", 0), reverse=True
    )

    risk_evolution = []
    miss_distance_evolution = []

    for cdm in sorted_cdms:
        tca = cdm.get("time_to_tca")
        risk = cdm.get("risk")
        md = cdm.get("miss_distance")

        if tca is not None and risk is not None:
            risk_evolution.append((float(tca), float(risk)))
        if tca is not None and md is not None:
            miss_distance_evolution.append((float(tca), float(md)))

    first_tca = sorted_cdms[0].get("time_to_tca")
    last_tca = sorted_cdms[-1].get("time_to_tca")

    convergence = _assess_convergence(risk_evolution)

    return {
        "first_cdm_time": float(first_tca) if first_tca is not None else None,
        "last_cdm_time": float(last_tca) if last_tca is not None else None,
        "total_cdms": len(sorted_cdms),
        "risk_evolution": risk_evolution,
        "miss_distance_evolution": miss_distance_evolution,
        "convergence_assessment": convergence,
    }


def _assess_convergence(risk_evolution: list) -> str:
    """Assess whether risk values are converging as TCA approaches."""
    if len(risk_evolution) < 3:
        return "insufficient_data"

    risks = [r for _, r in risk_evolution]
    diffs = np.abs(np.diff(risks))

    first_half = diffs[: len(diffs) // 2]
    second_half = diffs[len(diffs) // 2 :]

    if len(first_half) == 0 or len(second_half) == 0:
        return "insufficient_data"

    if np.mean(second_half) < np.mean(first_half) * 0.7:
        return "converging"
    elif np.mean(second_half) > np.mean(first_half) * 1.3:
        return "diverging"
    else:
        return "stable"


def summarize_event(event_cdms: list) -> dict:
    """Generate a summary for a conjunction event."""
    if not event_cdms:
        return {
            "event_id": None,
            "peak_risk": None,
            "final_risk": None,
            "total_updates": 0,
            "risk_range": (None, None),
            "miss_distance_range": (None, None),
            "duration_days": 0.0,
        }

    sorted_cdms = sorted(
        event_cdms, key=lambda x: x.get("time_to_tca", 0), reverse=True
    )

    event_id = sorted_cdms[0].get("event_id")
    risks = [c.get("risk") for c in sorted_cdms if c.get("risk") is not None]
    miss_distances = [
        c.get("miss_distance") for c in sorted_cdms if c.get("miss_distance") is not None
    ]
    tcas = [
        c.get("time_to_tca") for c in sorted_cdms if c.get("time_to_tca") is not None
    ]

    peak_risk = float(max(risks)) if risks else None
    final_risk = float(risks[-1]) if risks else None
    risk_range = (float(min(risks)), float(max(risks))) if risks else (None, None)
    md_range = (
        (float(min(miss_distances)), float(max(miss_distances)))
        if miss_distances
        else (None, None)
    )
    duration = float(max(tcas) - min(tcas)) if len(tcas) >= 2 else 0.0

    return {
        "event_id": event_id,
        "peak_risk": peak_risk,
        "final_risk": final_risk,
        "total_updates": len(sorted_cdms),
        "risk_range": risk_range,
        "miss_distance_range": md_range,
        "duration_days": duration,
    }
