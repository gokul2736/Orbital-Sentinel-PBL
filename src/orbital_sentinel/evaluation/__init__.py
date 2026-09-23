"""Evaluation toolkit for Orbital Sentinel models."""

from orbital_sentinel.evaluation.metrics import (
    evaluate_at_threshold,
    evaluate_by_risk_band,
    evaluate_regression,
)
from orbital_sentinel.evaluation.reports import compare_models, generate_model_report
from orbital_sentinel.evaluation.thresholding import (
    apply_risk_categories,
    find_optimal_threshold,
)
from orbital_sentinel.evaluation.cross_validation import (
    cross_validate_model,
    compare_models_cv,
    event_kfold_split,
)
from orbital_sentinel.evaluation.threshold_optimization import (
    multi_threshold_evaluation,
    optimize_threshold,
)

__all__ = [
    "evaluate_regression",
    "evaluate_by_risk_band",
    "evaluate_at_threshold",
    "generate_model_report",
    "compare_models",
    "find_optimal_threshold",
    "apply_risk_categories",
    "cross_validate_model",
    "compare_models_cv",
    "event_kfold_split",
    "multi_threshold_evaluation",
    "optimize_threshold",
]
