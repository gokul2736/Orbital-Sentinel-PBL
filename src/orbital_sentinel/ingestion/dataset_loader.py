from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from orbital_sentinel.config.settings import get_settings


def load_esa_kelvins(
    data_dir: Optional[str] = None, split: str = "train"
) -> pd.DataFrame:
    settings = get_settings()

    if data_dir is not None:
        base = Path(data_dir)
    else:
        base = settings.project_root / settings.data.raw_dir

    if split == "train":
        path = base / settings.data.train_file
    elif split == "test":
        path = base / settings.data.test_file
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'test'.")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    return pd.read_csv(path)


def load_both_splits(
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_esa_kelvins(data_dir=data_dir, split="train")
    test_df = load_esa_kelvins(data_dir=data_dir, split="test")
    return train_df, test_df
