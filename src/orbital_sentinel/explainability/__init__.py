"""Explainability layer — SHAP analysis and natural language explanations."""

from .shap_analysis import compute_shap_values
from .explanations import explain_prediction, generate_decision_summary

__all__ = ["compute_shap_values", "explain_prediction", "generate_decision_summary"]
