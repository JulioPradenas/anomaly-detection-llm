"""Integration tests for FastAPI endpoints using TestClient."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app, pipeline


@pytest.fixture(autouse=True)
def reset_pipeline():
    """Ensure pipeline state is clean before each test."""
    pipeline._loaded = False
    pipeline.model = None
    yield


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
