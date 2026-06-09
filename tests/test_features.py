"""Tests for feature engineering pipeline."""

import pandas as pd

from src.features.engineering import build_features


def _make_log_df(n: int = 200) -> pd.DataFrame:
    """Minimal DataFrame that mimics output of load_bgl_logs + add_severity_score."""
    import numpy as np

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2005-06-03 15:00", periods=n, freq="10s")
    levels = rng.choice(["INFO", "WARNING", "ERROR", "FATAL"], n, p=[0.7, 0.15, 0.1, 0.05])
    nodes = rng.choice(["R01-M1", "R02-M1", "R03-M1"], n)
    components = rng.choice(["KERNEL", "APP", "NET"], n)

    severity_map = {"INFO": 0, "WARNING": 1, "ERROR": 2, "FATAL": 4}
    severity = [severity_map[lvl] for lvl in levels]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "node": pd.Categorical(nodes),
            "level": pd.Categorical(levels),
            "component": pd.Categorical(components),
            "content": ["test message"] * n,
            "label": ["-"] * n,
            "is_anomaly": [False] * n,
            "severity_score": severity,
        }
    )


def test_build_features_returns_expected_columns():
    df = _make_log_df(200)
    feat_df, scaler = build_features(df, fit_scaler=True)
    assert "timestamp" in feat_df.columns
    assert "node" in feat_df.columns
    assert "is_anomaly" in feat_df.columns
    assert "error_count_5min" in feat_df.columns
    assert "fatal_count_1min" in feat_df.columns
    assert "burst_flag" in feat_df.columns


def test_build_features_no_nulls_in_feature_cols():
    df = _make_log_df(200)
    feat_df, _ = build_features(df, fit_scaler=True)
    feature_cols = [c for c in feat_df.columns if c not in {"timestamp", "node", "is_anomaly"}]
    assert feat_df[feature_cols].isnull().sum().sum() == 0


def test_build_features_shape_preserved():
    df = _make_log_df(200)
    feat_df, _ = build_features(df, fit_scaler=True)
    assert len(feat_df) == len(df)


def test_build_features_scaler_returned():
    from sklearn.preprocessing import RobustScaler

    df = _make_log_df(100)
    _, scaler = build_features(df, fit_scaler=True)
    assert isinstance(scaler, RobustScaler)


def test_build_features_error_count_is_numeric():
    # Columns are RobustScaler-transformed so values can be negative — just verify dtype
    df = _make_log_df(200)
    feat_df, _ = build_features(df)
    assert feat_df["error_count_5min"].dtype == "float32"
    assert feat_df["fatal_count_5min"].dtype == "float32"
