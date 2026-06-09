"""Tests for data loading and preprocessing."""

import textwrap
from pathlib import Path

import pytest

from src.data.loader import load_bgl_logs
from src.data.preprocessor import add_severity_score, train_test_split_temporal

BGL_SAMPLE = textwrap.dedent("""\
    - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
    - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.51.000000 R02-M1-N0-C:J12-U11 RAS KERNEL WARNING memory corrected
    APPREAD 1117869872 2005.06.04 R23-M1-N8-I:J18-U11 2005-06-04-00.24.32.398284 R23-M1-N8-I:J18-U11 RAS APP FATAL ciod: failed to read
""")


@pytest.fixture
def bgl_file(tmp_path: Path) -> Path:
    log_file = tmp_path / "BGL.log"
    log_file.write_text(BGL_SAMPLE)
    return log_file


def test_load_bgl_logs_shape(bgl_file: Path):
    df = load_bgl_logs(bgl_file)
    assert len(df) == 3
    assert "is_anomaly" in df.columns
    assert "timestamp" in df.columns


def test_load_bgl_logs_anomaly_detection(bgl_file: Path):
    df = load_bgl_logs(bgl_file)
    assert df["is_anomaly"].sum() == 1  # only the APPREAD row
    assert df.loc[df["is_anomaly"], "label"].iloc[0] == "APPREAD"


def test_load_bgl_logs_timestamp_parsed(bgl_file: Path):
    df = load_bgl_logs(bgl_file)
    import pandas as pd

    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].notna().all()


def test_load_bgl_logs_nrows(bgl_file: Path):
    df = load_bgl_logs(bgl_file, nrows=2)
    assert len(df) == 2


def test_load_bgl_logs_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_bgl_logs("/nonexistent/path/BGL.log")


def test_load_bgl_logs_categorical_columns(bgl_file: Path):
    df = load_bgl_logs(bgl_file)
    for col in ["node", "component", "level"]:
        assert df[col].dtype.name == "category", f"{col} should be category"


def test_add_severity_score(bgl_file: Path):
    df = load_bgl_logs(bgl_file)
    df = add_severity_score(df)
    assert "severity_score" in df.columns
    # FATAL row should have score 4
    fatal_rows = df[df["level"].astype(str) == "FATAL"]
    assert (fatal_rows["severity_score"] == 4).all()
    # INFO rows should have score 0
    info_rows = df[df["level"].astype(str) == "INFO"]
    assert (info_rows["severity_score"] == 0).all()


def test_train_test_split_temporal_no_leakage(bgl_file: Path):
    df = load_bgl_logs(bgl_file)
    train, test = train_test_split_temporal(df, test_fraction=0.33)
    # All train timestamps must be <= all test timestamps
    assert train["timestamp"].max() <= test["timestamp"].min()
