"""Tests for MLflow model registry utilities."""

from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.model_monitor import (
    get_latest_model_info,
    log_drift_report,
    promote_model,
)


@pytest.fixture
def mock_mlflow():
    """Patch mlflow at module level so no real tracking server is needed."""
    with patch("src.monitoring.model_monitor.mlflow") as m:
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id-abc123"
        m.start_run.return_value.__enter__ = lambda s: mock_run
        m.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield m


def test_log_drift_report_returns_run_id(mock_mlflow):
    run_id = log_drift_report(
        drift_score=0.35,
        features_drifted=["error_rate", "burst_flag"],
        tracking_uri="./mlflow_test",
    )
    assert run_id == "test-run-id-abc123"
    mock_mlflow.log_metrics.assert_called_once()
    call_kwargs = mock_mlflow.log_metrics.call_args[0][0]
    assert call_kwargs["drift_score"] == 0.35
    assert call_kwargs["n_features_drifted"] == 2


def test_log_drift_report_with_html(mock_mlflow, tmp_path):
    html_path = tmp_path / "drift_report.html"
    html_path.write_text("<html>report</html>")

    run_id = log_drift_report(
        drift_score=0.1,
        features_drifted=[],
        html_report_path=html_path,
        tracking_uri="./mlflow_test",
    )
    assert run_id == "test-run-id-abc123"
    mock_mlflow.log_artifact.assert_called_once()


def test_log_drift_report_no_html_no_artifact(mock_mlflow):
    log_drift_report(drift_score=0.0, features_drifted=[], tracking_uri="./mlflow_test")
    mock_mlflow.log_artifact.assert_not_called()


def test_get_latest_model_info_no_versions():
    with patch("src.monitoring.model_monitor.MlflowClient") as mock_client_cls:
        client = MagicMock()
        client.search_model_versions.return_value = []
        mock_client_cls.return_value = client

        result = get_latest_model_info("NonExistentModel", tracking_uri="./mlflow_test")

    assert result["model_name"] == "NonExistentModel"
    assert result["versions"] == 0


def test_get_latest_model_info_with_versions():
    with patch("src.monitoring.model_monitor.MlflowClient") as mock_client_cls:
        client = MagicMock()
        v1 = MagicMock()
        v1.version = "1"
        v1.run_id = "run-abc"
        v1.status = "READY"
        v2 = MagicMock()
        v2.version = "2"
        v2.run_id = "run-def"
        v2.status = "READY"
        client.search_model_versions.return_value = [v1, v2]
        mock_client_cls.return_value = client

        result = get_latest_model_info("AnomalyDetector-LOF", tracking_uri="./mlflow_test")

    assert result["latest_version"] == 2
    assert result["run_id"] == "run-def"
    assert result["status"] == "READY"


def test_get_latest_model_info_exception_returns_fallback():
    with patch("src.monitoring.model_monitor.MlflowClient") as mock_client_cls:
        client = MagicMock()
        client.search_model_versions.side_effect = Exception("connection refused")
        mock_client_cls.return_value = client

        result = get_latest_model_info("AnyModel", tracking_uri="./mlflow_test")

    assert result["versions"] == 0


def test_promote_model_calls_set_alias():
    with patch("src.monitoring.model_monitor.MlflowClient") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client

        promote_model("AnomalyDetector-LOF", version=2, stage="Production", tracking_uri="./mlflow_test")

        client.set_registered_model_alias.assert_called_once_with(
            "AnomalyDetector-LOF", "Production", "2"
        )
