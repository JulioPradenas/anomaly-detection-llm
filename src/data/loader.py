"""Parser for BGL (Blue Gene/L) supercomputer log dataset from Loghub."""

from pathlib import Path

import pandas as pd

BGL_COLUMNS = [
    "label",
    "timestamp_int",
    "date",
    "node",
    "datetime_str",
    "node_repeat",
    "type",
    "component",
    "level",
    "content",
]


def load_bgl_logs(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load and parse BGL log file into a structured DataFrame.

    BGL format (space-separated, content may contain spaces):
      <label> <ts_int> <date> <node> <datetime> <node> <type> <component> <level> <content>

    Args:
        path: Path to BGL.log file.
        nrows: If set, load only the first n rows (useful for development).

    Returns:
        DataFrame with typed columns and a boolean `is_anomaly` column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BGL log not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= nrows:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 9)
            if len(parts) < 9:
                continue
            # Pad content if the line has no content field
            if len(parts) == 9:
                parts.append("")
            rows.append(parts)

    df = pd.DataFrame(rows, columns=BGL_COLUMNS)
    df = _cast_types(df)
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    # Anomaly: label != "-"
    df["is_anomaly"] = df["label"] != "-"

    # Parse datetime — BGL format: 2005-06-03-15.42.50.675872
    df["timestamp"] = pd.to_datetime(
        df["datetime_str"],
        format="%Y-%m-%d-%H.%M.%S.%f",
        errors="coerce",
    )
    # Fallback: try date column if datetime_str failed
    mask_failed = df["timestamp"].isna()
    if mask_failed.any():
        df.loc[mask_failed, "timestamp"] = pd.to_datetime(
            df.loc[mask_failed, "date"], format="%Y.%m.%d", errors="coerce"
        )

    df["timestamp_int"] = pd.to_numeric(df["timestamp_int"], errors="coerce")

    # Categorical columns to save memory
    for col in ["node", "type", "component", "level"]:
        df[col] = df[col].astype("category")

    # Drop helper columns not needed downstream
    df = df.drop(columns=["datetime_str", "node_repeat"])

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def load_bgl_sample(path: str | Path, n: int = 10_000) -> pd.DataFrame:
    """Load a small sample of BGL logs — useful for quick tests."""
    return load_bgl_logs(path, nrows=n)
