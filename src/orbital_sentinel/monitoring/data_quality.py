"""Monitor incoming data quality for production use."""

from typing import Optional

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "event_id", "time_to_tca", "miss_distance",
    "c_object_type", "t_span", "c_span",
    "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
]

DEFAULT_THRESHOLDS = {
    "null_rate": 0.05,
    "out_of_range_rate": 0.01,
    "schema_missing_columns": 0,
    "mean_shift_sigma": 3.0,
    "std_ratio_min": 0.3,
    "std_ratio_max": 3.0,
    "range_violation_rate": 0.02,
}


class DataQualityMonitor:
    """Monitors data quality of incoming CDM batches."""

    def __init__(self, expected_columns: Optional[list] = None):
        self.expected_columns = expected_columns or EXPECTED_COLUMNS

    def check_batch(self, df: pd.DataFrame) -> dict:
        """Check a batch of incoming data for quality issues."""
        report = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "null_rates": {},
            "out_of_range": {},
            "schema": {"missing_columns": [], "extra_columns": [], "conformant": True},
        }

        missing = [c for c in self.expected_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in self.expected_columns]
        report["schema"]["missing_columns"] = missing
        report["schema"]["extra_columns"] = extra
        report["schema"]["conformant"] = len(missing) == 0

        for col in df.columns:
            null_rate = float(df[col].isna().mean())
            report["null_rates"][col] = null_rate

        range_checks = {
            "time_to_tca": (0.0, None),
            "miss_distance": (0.0, None),
            "relative_speed": (0.0, None),
        }
        for col, (lo, hi) in range_checks.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            violations = np.zeros(len(series), dtype=bool)
            if lo is not None:
                violations |= series.values < lo
            if hi is not None:
                violations |= series.values > hi
            report["out_of_range"][col] = {
                "violation_rate": float(violations.mean()),
                "n_violations": int(violations.sum()),
                "min_value": float(series.min()),
                "max_value": float(series.max()),
            }

        report["duplicate_rows"] = int(df.duplicated().sum())
        report["overall_null_rate"] = float(df.isna().mean().mean()) if len(df) > 0 else 0.0

        return report

    def compare_to_baseline(self, df: pd.DataFrame, baseline_stats: dict) -> dict:
        """Compare new batch statistics to baseline training data stats.

        baseline_stats expected format per feature:
            {"mean": float, "std": float, "min": float, "max": float}
        """
        comparison = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in baseline_stats:
                continue
            bl = baseline_stats[col]
            series = df[col].dropna()
            if len(series) == 0:
                continue

            current_mean = float(series.mean())
            current_std = float(series.std()) if len(series) > 1 else 0.0
            bl_mean = bl.get("mean", 0.0)
            bl_std = bl.get("std", 1.0)
            bl_min = bl.get("min", -np.inf)
            bl_max = bl.get("max", np.inf)

            mean_shift = abs(current_mean - bl_mean) / max(bl_std, 1e-10)
            std_ratio = current_std / max(bl_std, 1e-10)

            out_of_baseline = float(((series < bl_min) | (series > bl_max)).mean())

            comparison[col] = {
                "current_mean": current_mean,
                "current_std": current_std,
                "baseline_mean": bl_mean,
                "baseline_std": bl_std,
                "mean_shift_sigma": float(mean_shift),
                "std_ratio": float(std_ratio),
                "out_of_baseline_range_rate": out_of_baseline,
            }

        return comparison

    def generate_alerts(self, quality_report: dict, thresholds: Optional[dict] = None) -> list:
        """Generate alert strings for any quality metric exceeding thresholds."""
        t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        alerts = []

        missing_cols = quality_report.get("schema", {}).get("missing_columns", [])
        if len(missing_cols) > t["schema_missing_columns"]:
            alerts.append(f"SCHEMA: Missing columns: {missing_cols}")

        for col, rate in quality_report.get("null_rates", {}).items():
            if rate > t["null_rate"]:
                alerts.append(
                    f"NULL_RATE: Column '{col}' has {rate:.1%} null values "
                    f"(threshold: {t['null_rate']:.1%})"
                )

        for col, info in quality_report.get("out_of_range", {}).items():
            if info["violation_rate"] > t["out_of_range_rate"]:
                alerts.append(
                    f"OUT_OF_RANGE: Column '{col}' has {info['violation_rate']:.1%} "
                    f"out-of-range values (n={info['n_violations']})"
                )

        for col, info in quality_report.get("comparison", {}).items():
            if info.get("mean_shift_sigma", 0) > t["mean_shift_sigma"]:
                alerts.append(
                    f"DRIFT: Column '{col}' mean shifted by "
                    f"{info['mean_shift_sigma']:.1f} sigma from baseline"
                )
            std_ratio = info.get("std_ratio", 1.0)
            if std_ratio < t["std_ratio_min"] or std_ratio > t["std_ratio_max"]:
                alerts.append(
                    f"DRIFT: Column '{col}' std ratio = {std_ratio:.2f} "
                    f"(expected {t['std_ratio_min']:.1f}-{t['std_ratio_max']:.1f})"
                )
            if info.get("out_of_baseline_range_rate", 0) > t["range_violation_rate"]:
                alerts.append(
                    f"RANGE: Column '{col}' has "
                    f"{info['out_of_baseline_range_rate']:.1%} values outside baseline range"
                )

        return alerts
