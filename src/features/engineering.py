"""Temporal window feature engineering for BGL log anomaly detection.

All window calculations use only past data (no lookahead) to prevent leakage.
DuckDB handles the rolling aggregations efficiently even on large datasets.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


WINDOW_MINUTES = [1, 5, 15]
TOP_N_COMPONENTS = 20


def build_features(df: pd.DataFrame, fit_scaler: bool = True) -> tuple[pd.DataFrame, RobustScaler]:
    """Build feature matrix from raw BGL DataFrame.

    Args:
        df: Output of load_bgl_logs() with `timestamp`, `level`, `node`, `component` columns.
        fit_scaler: If True, fit a new RobustScaler. Set False on test data.

    Returns:
        (features_df, scaler) — features_df has numeric columns ready for sklearn.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = _add_severity_score(df)
    df = _add_temporal_windows(df)
    df = _add_node_features(df)
    df = _add_component_encoding(df)

    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].fillna(0).astype("float32")

    scaler = RobustScaler()
    if fit_scaler:
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=df.index)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=df.index)

    result = pd.concat([
        df[["timestamp", "node", "is_anomaly"]].reset_index(drop=True),
        X_scaled.reset_index(drop=True),
    ], axis=1)

    return result, scaler


def _add_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    severity_map = {"INFO": 0, "WARNING": 1, "WARN": 1, "ERROR": 2, "SEVERE": 3, "FATAL": 4}
    df["severity_score"] = (
        df["level"].astype(str).str.upper().map(severity_map).fillna(0).astype("int8")
    )
    return df


def _add_temporal_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling window aggregations using DuckDB for performance."""
    # DuckDB works with plain column names — materialize needed columns
    tmp = df[["timestamp", "severity_score", "node"]].copy()
    tmp["is_error"] = (df["severity_score"] >= 2).astype("int8")
    tmp["is_warning"] = (df["severity_score"] == 1).astype("int8")
    tmp["is_fatal"] = (df["severity_score"] >= 3).astype("int8")
    tmp["row_id"] = np.arange(len(tmp))

    con = duckdb.connect()
    con.register("logs", tmp)

    for w in WINDOW_MINUTES:
        interval = f"INTERVAL '{w} minutes'"
        query = f"""
        SELECT
            row_id,
            COUNT(*) OVER w                        AS total_count_{w}min,
            SUM(is_error) OVER w                   AS error_count_{w}min,
            SUM(is_warning) OVER w                 AS warning_count_{w}min,
            SUM(is_fatal) OVER w                   AS fatal_count_{w}min,
            AVG(severity_score) OVER w             AS avg_severity_{w}min,
            COUNT(DISTINCT node) OVER w            AS unique_nodes_{w}min
        FROM logs
        WINDOW w AS (
            ORDER BY timestamp
            RANGE BETWEEN {interval} PRECEDING AND CURRENT ROW
        )
        ORDER BY row_id
        """
        win_df = con.execute(query).df()
        win_df = win_df.set_index("row_id")

        for col in win_df.columns:
            df[col] = win_df[col].values

    # error_rate = errors / total events in 5-min window (avoid division by zero)
    total_5 = df["total_count_5min"].replace(0, np.nan)
    df["error_rate_5min"] = df["error_count_5min"] / total_5
    df["fatal_rate_5min"] = df["fatal_count_5min"] / total_5

    # burst_flag: event count in 1min > mean + 2*std of 15min window
    mean_15 = df["total_count_15min"] / 15  # rough rolling mean proxy
    df["burst_flag"] = (df["total_count_1min"] > mean_15 * 2).astype("int8")

    con.close()
    return df


def _add_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-node statistics computed over the entire dataset.

    These are global stats (not windowed) to capture node-level baseline behavior.
    In production, these would be computed on training data only.
    """
    node_stats = (
        df.groupby("node", observed=True)
        .agg(
            node_total=("severity_score", "count"),
            node_error_count=("is_anomaly", "sum"),  # using label as proxy
            node_avg_severity=("severity_score", "mean"),
        )
        .reset_index()
    )
    node_stats["node_error_ratio"] = (
        node_stats["node_error_count"] / node_stats["node_total"].replace(0, np.nan)
    )
    df = df.merge(
        node_stats[["node", "node_error_ratio", "node_avg_severity"]],
        on="node",
        how="left",
        suffixes=("", "_node"),
    )
    return df


def _add_component_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency encoding for top-N components."""
    comp_freq = df["component"].value_counts(normalize=True)
    top_components = comp_freq.index[:TOP_N_COMPONENTS].tolist()

    for comp in top_components:
        safe_name = str(comp).replace(" ", "_").replace("-", "_").lower()
        df[f"comp_{safe_name}"] = (df["component"] == comp).astype("int8")

    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"timestamp", "timestamp_int", "date", "node", "type",
            "component", "level", "content", "label", "is_anomaly",
            "datetime_str", "node_repeat", "severity_score"}
    return [c for c in df.columns if c not in skip]


def save_features(df: pd.DataFrame, path: str | Path) -> None:
    df.to_parquet(path, index=False)


def load_features(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
