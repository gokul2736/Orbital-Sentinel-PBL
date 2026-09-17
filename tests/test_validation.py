import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
from orbital_sentinel.validation.schema import validate_schema, REQUIRED_COLUMNS
from orbital_sentinel.validation.quality import assess_quality
from orbital_sentinel.validation.leakage import check_leakage


def test_validate_schema_passes_on_real_data():
    df = load_esa_kelvins(split="train")
    issues = validate_schema(df)
    assert issues == [], f"Schema validation issues: {issues}"


def test_validate_schema_catches_missing_columns():
    df = pd.DataFrame({"event_id": [1], "risk": [-5.0]})
    issues = validate_schema(df)
    assert len(issues) > 0
    assert any("Missing required columns" in issue for issue in issues)


def test_validate_schema_catches_non_numeric_risk():
    cols = {col: [0] for col in REQUIRED_COLUMNS}
    cols["risk"] = ["not_a_number"]
    df = pd.DataFrame(cols)
    issues = validate_schema(df)
    assert any("not numeric" in issue for issue in issues)


def test_assess_quality_returns_expected_keys():
    df = load_esa_kelvins(split="train")
    result = assess_quality(df)
    expected_keys = {
        "total_rows", "total_columns", "null_row_count",
        "columns_with_nulls", "risk_floor_pct",
        "negative_miss_distance_count", "negative_speed_count",
        "outlier_columns",
    }
    assert set(result.keys()) == expected_keys
    assert result["total_rows"] > 0
    assert result["total_columns"] > 0
    assert isinstance(result["columns_with_nulls"], list)
    assert isinstance(result["outlier_columns"], list)


def test_check_leakage_detects_known_leakage_columns():
    df = pd.DataFrame({
        "risk": [-5.0, -10.0, -15.0],
        "miss_distance": [100.0, 200.0, 300.0],
        "max_risk_estimate": [-4.0, -9.0, -14.0],
    })
    feature_columns = ["miss_distance", "max_risk_estimate"]
    warnings = check_leakage(df, feature_columns)
    assert len(warnings) > 0
    assert any("leakage" in w.lower() or "max_risk_estimate" in w for w in warnings)


def test_check_leakage_detects_risk_named_columns():
    df = pd.DataFrame({
        "risk": [-5.0, -10.0],
        "custom_risk_score": [-4.0, -9.0],
        "miss_distance": [100.0, 200.0],
    })
    feature_columns = ["custom_risk_score", "miss_distance"]
    warnings = check_leakage(df, feature_columns)
    assert any("risk" in w.lower() for w in warnings)
