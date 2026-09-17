"""Natural language explanations and decision-support summaries."""

from typing import List, Optional


def explain_prediction(
    risk_assessment: dict,
    top_features: List[dict],
    physics_result: Optional[dict] = None,
) -> str:
    ra = risk_assessment
    category = ra.get("risk_category", "UNKNOWN")
    fused = ra.get("fused_risk", 0.0)
    conf = ra.get("confidence", 0.0)

    parts = [
        f"This conjunction is assessed as {category} risk "
        f"(log10(Pc) = {fused:.1f}, confidence: {conf:.2f})."
    ]

    if top_features:
        drivers = []
        for feat in top_features[:3]:
            name = feat["feature"]
            sv = feat["shap_value"]
            direction = "+" if sv > 0 else ""
            drivers.append(f"{name} (SHAP contribution: {direction}{sv:.2f})")
        parts.append("The primary risk drivers are: " + ", ".join(drivers) + ".")

    if physics_result is not None:
        phys_log10 = physics_result.get("log10_pc")
        interval = ra.get("prediction_interval")
        if phys_log10 is not None:
            if interval and interval[0] <= phys_log10 <= interval[1]:
                parts.append(
                    f"Physics verification confirms the assessment "
                    f"(analytic Pc = 10^{phys_log10:.1f}, within ML prediction interval)."
                )
            else:
                parts.append(
                    f"Physics verification shows some disagreement "
                    f"(analytic Pc = 10^{phys_log10:.1f}, outside ML prediction interval). "
                    "Manual review is recommended."
                )
    else:
        parts.append("No physics verification is available for this conjunction.")

    action = recommended_action(
        category,
        conf,
        time_to_tca=ra.get("time_to_tca"),
    )
    parts.append(f"Recommendation: {action}")

    return " ".join(parts)


def generate_decision_summary(assessment: dict) -> dict:
    ra = assessment.get("risk_assessment", assessment)
    category = ra.get("risk_category", "UNKNOWN")
    fused = ra.get("fused_risk", 0.0)
    conf = ra.get("confidence", 0.0)
    agreement = ra.get("ml_physics_agreement", "UNKNOWN")
    warnings = ra.get("warnings", [])

    key_factors = []
    feat_exp = assessment.get("feature_explanation")
    if feat_exp and isinstance(feat_exp, list):
        for f in feat_exp[:5]:
            key_factors.append({
                "feature": f.get("feature", ""),
                "shap_value": f.get("shap_value", 0.0),
                "direction": f.get("direction", ""),
            })

    headline = _build_headline(category, fused, conf)

    explanation_parts = [f"Fused log10(Pc) = {fused:.2f}."]
    if ra.get("ml_prediction") is not None:
        explanation_parts.append(f"ML prediction: {ra['ml_prediction']:.2f}.")
    if ra.get("physics_pc") is not None:
        explanation_parts.append(f"Physics Pc: 10^{ra['physics_pc']:.2f}.")
    explanation_parts.append(f"Confidence: {conf:.0%}.")
    explanation = " ".join(explanation_parts)

    action = recommended_action(
        category,
        conf,
        time_to_tca=assessment.get("conjunction_data", {}).get("time_to_tca"),
    )

    return {
        "risk_level": category,
        "headline": headline,
        "explanation": explanation,
        "key_factors": key_factors,
        "physics_agreement": agreement,
        "confidence": conf,
        "recommended_action": action,
        "warnings": warnings,
    }


def _build_headline(category: str, fused: float, confidence: float) -> str:
    conf_label = "high" if confidence >= 0.7 else "moderate" if confidence >= 0.4 else "low"
    headlines = {
        "CRITICAL": f"CRITICAL collision risk detected with {conf_label} confidence",
        "HIGH": f"High collision risk — immediate attention required ({conf_label} confidence)",
        "MODERATE": f"Moderate collision risk — further analysis recommended ({conf_label} confidence)",
        "LOW": f"Low collision risk — routine monitoring sufficient ({conf_label} confidence)",
        "NEGLIGIBLE": f"Negligible collision risk ({conf_label} confidence)",
    }
    return headlines.get(category, f"{category} risk (log10(Pc) = {fused:.1f})")


def recommended_action(
    risk_category: str,
    confidence: float,
    time_to_tca: Optional[float] = None,
) -> str:
    urgency = ""
    if time_to_tca is not None:
        if time_to_tca < 6:
            urgency = " (TCA < 6 hours — URGENT)"
        elif time_to_tca < 24:
            urgency = " (TCA < 24 hours)"

    if risk_category == "CRITICAL":
        if confidence >= 0.6:
            return f"IMMEDIATE maneuver planning required{urgency}"
        return f"Escalate urgently for independent verification — high risk but uncertain{urgency}"

    if risk_category == "HIGH":
        if confidence >= 0.5:
            return f"Escalate to conjunction assessment team within 4 hours{urgency}"
        return f"Flag for priority review — elevated risk with low confidence{urgency}"

    if risk_category == "MODERATE":
        return f"Schedule detailed analysis, monitor subsequent CDMs{urgency}"

    if risk_category == "LOW":
        return "Continue routine monitoring"

    return "No action required, archive for statistics"
