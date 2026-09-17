import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins, load_both_splits


def test_load_esa_kelvins_returns_dataframe_with_expected_columns():
    df = load_esa_kelvins(split="train")
    assert len(df) > 0
    for col in [
        "event_id", "time_to_tca", "risk", "miss_distance",
        "relative_speed", "relative_position_r", "relative_position_t",
        "relative_position_n", "relative_velocity_r", "relative_velocity_t",
        "relative_velocity_n", "t_j2k_sma", "t_j2k_ecc",
    ]:
        assert col in df.columns, f"Missing expected column: {col}"


def test_train_split_has_rows():
    df = load_esa_kelvins(split="train")
    assert len(df) > 0


def test_invalid_split_raises_value_error():
    with pytest.raises(ValueError, match="Unknown split"):
        load_esa_kelvins(split="invalid_split")
