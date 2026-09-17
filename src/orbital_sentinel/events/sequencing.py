"""Analyze CDM sequences within a conjunction event."""

import numpy as np
import pandas as pd


def extract_event_sequences(df: pd.DataFrame) -> dict:
    """Group DataFrame by event_id, return dict mapping event_id to CDM list sorted by time_to_tca."""
    if "event_id" not in df.columns:
        raise ValueError("DataFrame must contain 'event_id' column")

    sequences = {}
    for event_id, group in df.groupby("event_id"):
        sorted_group = group.sort_values("time_to_tca", ascending=False)
        sequences[event_id] = sorted_group.to_dict("records")

    return sequences


def compute_risk_trend(sequence: list) -> dict:
    """Compute risk trend from a CDM sequence.

    Expects each CDM dict to have 'time_to_tca' and 'risk' (log10 collision probability).
    """
    if len(sequence) < 2:
        return {
            "trend": "insufficient_data",
            "rate_of_change": 0.0,
            "is_converging": False,
            "n_points": len(sequence),
        }

    sorted_seq = sorted(sequence, key=lambda x: x.get("time_to_tca", 0), reverse=True)

    risks = np.array([s.get("risk", np.nan) for s in sorted_seq])
    times = np.array([s.get("time_to_tca", np.nan) for s in sorted_seq])

    valid = ~(np.isnan(risks) | np.isnan(times))
    risks = risks[valid]
    times = times[valid]

    if len(risks) < 2:
        return {
            "trend": "insufficient_data",
            "rate_of_change": 0.0,
            "is_converging": False,
            "n_points": int(valid.sum()),
        }

    diffs = np.diff(risks)
    mean_diff = float(diffs.mean())

    if abs(mean_diff) < 0.1:
        trend = "stable"
    elif mean_diff > 0:
        trend = "increasing"
    else:
        trend = "decreasing"

    time_span = times[0] - times[-1]
    rate = mean_diff / max(abs(time_span), 1e-10)

    recent_diffs = diffs[-min(3, len(diffs)):]
    is_converging = bool(np.std(recent_diffs) < np.std(diffs)) if len(diffs) > 2 else False

    return {
        "trend": trend,
        "rate_of_change": float(rate),
        "is_converging": is_converging,
        "n_points": len(risks),
        "latest_risk": float(risks[-1]),
        "earliest_risk": float(risks[0]),
        "risk_change": float(risks[-1] - risks[0]),
    }


def identify_escalating_events(df: pd.DataFrame, threshold: float = -5.0) -> list:
    """Find events where risk trend is increasing and latest risk exceeds threshold."""
    sequences = extract_event_sequences(df)
    escalating = []

    for event_id, seq in sequences.items():
        trend_info = compute_risk_trend(seq)

        if trend_info["trend"] != "increasing":
            continue

        latest_risk = trend_info.get("latest_risk", -np.inf)
        if latest_risk > threshold:
            escalating.append({
                "event_id": event_id,
                "latest_risk": latest_risk,
                "trend": trend_info,
                "n_cdms": len(seq),
            })

    return sorted(escalating, key=lambda x: x["latest_risk"], reverse=True)
