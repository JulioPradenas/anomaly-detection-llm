"""Basic preprocessing utilities on top of the raw BGL DataFrame."""

import pandas as pd


SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "WARN": 1, "ERROR": 2,
                  "SEVERE": 3, "FATAL": 4}


def add_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Map log level to a numeric severity score (0=INFO … 4=FATAL)."""
    df = df.copy()
    df["severity_score"] = (
        df["level"].astype(str).str.upper().map(SEVERITY_ORDER).fillna(0).astype("int8")
    )
    return df


def filter_time_range(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Filter DataFrame to a timestamp window."""
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["timestamp"] >= pd.Timestamp(start)
    if end:
        mask &= df["timestamp"] <= pd.Timestamp(end)
    return df[mask].reset_index(drop=True)


def train_test_split_temporal(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically — last `test_fraction` of time as holdout.

    Temporal split avoids data leakage: the model never sees future events
    during training.
    """
    cutoff_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:cutoff_idx].reset_index(drop=True)
    test = df.iloc[cutoff_idx:].reset_index(drop=True)
    return train, test
