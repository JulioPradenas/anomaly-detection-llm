"""Integration tests for FastAPI endpoints using TestClient."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, pipeline


@pytest.fixture(autouse=True)
def reset_pipeline(tmp_path):
    """Ensure pipeline state is clean before each test — redirect model path so lifespan won't find it."""
    original_path = pipeline._model_path
    pipeline._model_path = tmp_path / "nonexistent_model.joblib"
    pipeline._loaded = False
    pipeline.model = None
    yield
    pipeline._model_path = original_path
    pipeline._loaded = False
    pipeline.model = None


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_model_not_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_health_llm_field_present(client):
    response = client.get("/health")
    assert "llm_available" in response.json()


def test_detect_returns_503_when_model_not_loaded(client):
    payload = {
        "logs": [
            {
                "timestamp": "2005-06-03T15:42:50",
                "node": "R02-M1-N0",
                "level": "INFO",
                "component": "KERNEL",
                "content": "test message",
            }
        ]
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 503


def test_summarize_empty_context_returns_400(client):
    payload = {
        "anomaly_id": "test-123",
        "context_window": [],
        "anomaly_score": 0.8,
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 400


def test_summarize_with_mocked_llm(client):
    pipeline._loaded = True

    mock_summarizer = MagicMock()
    mock_summarizer.summarize.return_value = {
        "resumen": "Test anomaly detected.",
        "severidad": "HIGH",
        "causa_probable": "Memory failure",
        "accion_recomendada": "Restart node",
        "response_time_ms": 123.4,
    }
    pipeline.summarizer = mock_summarizer

    payload = {
        "anomaly_id": "test-456",
        "context_window": [
            {
                "timestamp": "2005-06-04T00:24:32",
                "node": "R23-M1-N8",
                "level": "FATAL",
                "component": "APP",
                "content": "ciod: failed to read",
            }
        ],
        "anomaly_score": 0.9,
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["severidad"] == "HIGH"
    assert data["anomaly_id"] == "test-456"
    assert data["response_time_ms"] == 123.4


# ── /model/health ─────────────────────────────────────────────────────────────


def test_model_health_no_features_returns_200(client):
    """When features file doesn't exist, endpoint still returns 200 with fallback."""
    response = client.get("/model/health")
    assert response.status_code == 200
    data = response.json()
    assert "drift_score" in data
    assert "drift_detected" in data
    assert "recommendation" in data
    assert isinstance(data["features_drifted"], list)


def test_model_health_with_features(tmp_path, client):
    """When features file exists, drift score is computed and returned."""
    rng = np.random.default_rng(42)
    n = 200
    feat_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2005-01-01", periods=n, freq="1min"),
            "node": ["R01-M0-N0"] * n,
            "is_anomaly": [False] * 160 + [True] * 40,
            "error_rate": rng.uniform(0, 0.3, n),
            "burst_flag": rng.uniform(0, 0.2, n),
        }
    )
    feat_path = tmp_path / "features_train.parquet"
    feat_df.to_parquet(feat_path, index=False)

    with patch("api.main.Path") as mock_path_cls:
        mock_feat = MagicMock()
        mock_feat.exists.return_value = True
        mock_feat.__str__ = lambda self: str(feat_path)

        def path_side_effect(p):
            if "features_train" in str(p):
                return feat_path
            return Path(p)

        mock_path_cls.side_effect = path_side_effect

        with patch("api.main.load_features", return_value=feat_df):
            response = client.get("/model/health")

    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "lof_v1"
    assert 0.0 <= data["drift_score"] <= 1.0


# ── /agent/chat ───────────────────────────────────────────────────────────────


def test_agent_chat_returns_response(client):
    """Agent endpoint returns response even when LLM is unavailable."""
    from api.main import agent

    agent._available = False
    payload = {"message": "¿Cuántas anomalías hay?", "session_id": "test-session"}
    response = client.post("/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "tools_used" in data
    assert data["session_id"] == "test-session"
    agent._available = None


def test_delete_agent_session(client):
    """DELETE /agent/session/{id} returns cleared=False for non-existent session."""
    response = client.delete("/agent/session/nonexistent-123")
    assert response.status_code == 200
    data = response.json()
    assert data["cleared"] is False
    assert data["session_id"] == "nonexistent-123"


# ── /retrain ──────────────────────────────────────────────────────────────────


def test_retrain_no_features_returns_503(client):
    """When features file doesn't exist, /retrain returns 503."""
    with patch("api.main.Path") as mock_path_cls:
        mock_p = MagicMock()
        mock_p.exists.return_value = False
        mock_path_cls.return_value = mock_p
        response = client.post("/retrain", json={"use_llm_labels": False, "min_samples": 10})
    assert response.status_code == 503


def test_retrain_with_features(tmp_path, client):
    """When features exist, /retrain trains and returns comparison metrics."""
    rng = np.random.default_rng(0)
    n = 300
    feat_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2005-01-01", periods=n, freq="1min"),
            "node": ["R01-M0-N0"] * n,
            "is_anomaly": ([False] * 240) + ([True] * 60),
            "error_rate": rng.uniform(0, 1, n),
            "burst_flag": rng.uniform(0, 1, n),
            "severity_mean": rng.uniform(0, 4, n),
        }
    )

    with (
        patch("api.main.load_features", return_value=feat_df),
        patch("api.main.Path") as mock_path_cls,
    ):

        def path_side_effect(p):
            if "features_train" in str(p):
                m = MagicMock()
                m.exists.return_value = True
                return m
            if "llm_confirmed" in str(p):
                m = MagicMock()
                m.exists.return_value = False
                return m
            m = MagicMock()
            m.exists.return_value = False
            m.__truediv__ = lambda s, o: Path(tmp_path) / o
            m.parent = MagicMock()
            m.parent.mkdir = MagicMock()
            return m

        mock_path_cls.side_effect = path_side_effect

        with (
            patch("api.main.ActiveLearner") as mock_al_cls,
            patch("api.main.evaluate") as mock_eval,
        ):
            mock_learner = MagicMock()
            mock_learner.predict.return_value = np.ones(60, dtype=int)
            mock_learner.score_samples.return_value = np.ones(60) * 0.9
            mock_al_cls.return_value = mock_learner

            mock_result = MagicMock()
            mock_result.f1 = 0.91
            mock_eval.return_value = mock_result

            response = client.post("/retrain", json={"use_llm_labels": False, "min_samples": 10})

    assert response.status_code == 200
    data = response.json()
    assert "f1_after" in data
    assert "model_version" in data
