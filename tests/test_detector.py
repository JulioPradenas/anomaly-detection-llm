"""Tests for AnomalyDetector."""

import numpy as np
import pytest

from src.models.detector import AnomalyDetector


@pytest.fixture
def simple_data():
    """Small numeric dataset — not actual log data, just for model interface testing."""
    rng = np.random.default_rng(42)
    X_normal = rng.normal(0, 1, (200, 5))
    X_anomaly = rng.normal(10, 1, (10, 5))
    return np.vstack([X_normal, X_anomaly])


@pytest.mark.parametrize("model_type", ["isolation_forest", "lof", "ocsvm"])
def test_fit_predict_shape(simple_data, model_type):
    detector = AnomalyDetector(model_type=model_type, contamination=0.04)
    detector.fit(simple_data)
    preds = detector.predict(simple_data)
    assert preds.shape == (len(simple_data),)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("model_type", ["isolation_forest", "lof", "ocsvm"])
def test_score_samples_shape(simple_data, model_type):
    detector = AnomalyDetector(model_type=model_type, contamination=0.04)
    detector.fit(simple_data)
    scores = detector.score_samples(simple_data)
    assert scores.shape == (len(simple_data),)


def test_isolation_forest_detects_obvious_anomalies(simple_data):
    """Anomalies at mean=10 should score higher than normals at mean=0."""
    detector = AnomalyDetector(model_type="isolation_forest", contamination=0.04)
    detector.fit(simple_data)
    scores = detector.score_samples(simple_data)
    # Last 10 samples are anomalies (at mean=10)
    assert scores[-10:].mean() > scores[:200].mean()


def test_invalid_model_type():
    with pytest.raises(ValueError):
        AnomalyDetector(model_type="unknown_model")


def test_save_load_roundtrip(tmp_path, simple_data):
    path = tmp_path / "model.joblib"
    detector = AnomalyDetector(model_type="isolation_forest", contamination=0.04)
    detector.fit(simple_data)
    detector.save(path)

    loaded = AnomalyDetector.load(path)
    preds_original = detector.predict(simple_data)
    preds_loaded = loaded.predict(simple_data)
    np.testing.assert_array_equal(preds_original, preds_loaded)
