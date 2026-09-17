from typing import Tuple

import numpy as np
import pandas as pd


def event_based_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_events = df["event_id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_events)

    n_val = max(1, int(len(unique_events) * test_size))
    val_events = set(unique_events[:n_val])

    val_mask = df["event_id"].isin(val_events)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    return train_df, val_df
