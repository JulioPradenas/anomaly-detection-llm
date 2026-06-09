"""Tests for DriftDetector — Evidently-based feature drift monitoring."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector, DriftReport

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "error_rate": rng.normal(0.05, 0.01, 300),
            "burst_flag": rng.normal(0.1, 0.02, 300),
            "severity_mean": rng.normal(1.2, 0.3, 300),
        }
    )


@pytest.fixture
def no_drift_df(reference_df):
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "error_rate": rng.normal(0.05, 0.01, 100),
            "burst_flag": rng.normal(0.1, 0.02, 100),
            "severity_mean": rng.normal(1.2, 0.3, 100),
        }
    )


@pytest.fixture
def high_drift_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "error_rate": rng.normal(0.8, 0.05, 100),
            "burst_flag": rng.normal(0.9, 0.05, 100),
            "severity_mean": rng.normal(3.5, 0.2, 100),
        }
    )


# ── fit_reference ─────────────────────────────────────────────────────────────


def test_fit_reference_returns_self(reference_df):
    detector = DriftDetector()
    result = detector.fit_reference(reference_df)
    assert result is detector
    assert detector._reference is not None


def test_detect_before_fit_raises():
    detector = DriftDetector()
    with pytest.raises(RuntimeError, match="fit_reference"):
        detector.detect(pd.DataFrame({"x": [1, 2]}))


# ── detect — no drift ─────────────────────────────────────────────────────────


def test_detect_no_drift_low_score(reference_df, no_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    report = detector.detect(no_drift_df)
    assert isinstance(report, DriftReport)
    assert report.drift_score < 0.5
    assert (
        "monitorear" in report.recommendation.lower()
        or "continuar" in report.recommendation.lower()
    )


# ── detect — with drift ───────────────────────────────────────────────────────


def test_detect_high_drift_score(reference_df, high_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    report = detector.detect(high_drift_df)
    assert report.drift_score > 0.0
    assert report.drift_detected is True
    assert len(report.features_drifted) > 0


def test_detect_high_drift_recommendation(reference_df, high_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    report = detector.detect(high_drift_df)
    assert any(
        kw in report.recommendation.lower() for kw in ("re-entrenamiento", "retrain", "drift")
    )


# ── get_drift_score ───────────────────────────────────────────────────────────


def test_get_drift_score_before_detect_returns_zero():
    detector = DriftDetector()
    assert detector.get_drift_score() == 0.0


def test_get_drift_score_after_detect(reference_df, high_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    detector.detect(high_drift_df)
    assert detector.get_drift_score() > 0.0


# ── DriftReport fields ────────────────────────────────────────────────────────


def test_drift_report_fields(reference_df, high_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    report = detector.detect(high_drift_df)
    assert isinstance(report.drift_score, float)
    assert isinstance(report.drift_detected, bool)
    assert isinstance(report.features_drifted, list)
    assert isinstance(report.feature_scores, dict)
    assert isinstance(report.n_features_total, int)
    assert isinstance(report.recommendation, str)


def test_feature_scores_keys_match_columns(reference_df, no_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    report = detector.detect(no_drift_df)
    assert set(report.feature_scores.keys()).issubset(set(reference_df.columns))


# ── generate_html_report_from_data ────────────────────────────────────────────


def test_generate_html_report_creates_file(tmp_path, reference_df, high_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    out = tmp_path / "drift_report.html"
    report = detector.generate_html_report_from_data(high_drift_df, out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert isinstance(report, DriftReport)


def test_generate_html_creates_parent_dirs(tmp_path, reference_df, no_drift_df):
    detector = DriftDetector()
    detector.fit_reference(reference_df)
    out = tmp_path / "reports" / "drift" / "report.html"
    detector.generate_html_report_from_data(no_drift_df, out)
    assert out.exists()
