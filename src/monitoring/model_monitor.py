"""MLflow model registry utilities — register, promote, and compare model versions."""

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MLFLOW_URI = "./mlflow"
EXPERIMENT_NAME = "anomaly-detection-v2"


def setup_mlflow(tracking_uri: str = MLFLOW_URI) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return MlflowClient(tracking_uri=tracking_uri)


def log_lof_model(
    model_path: str | Path,
    metrics: dict[str, float],
    params: dict[str, Any],
    tracking_uri: str = MLFLOW_URI,
) -> str:
    """Register the LOF model in MLflow and return the run_id."""
    import joblib

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lof_v1") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model = joblib.load(model_path)
        mlflow.sklearn.log_model(model, "lof_model", registered_model_name="AnomalyDetector-LOF")
        if Path(model_path).exists():
            mlflow.log_artifact(str(model_path), artifact_path="model_files")
        return run.info.run_id


def log_active_learner(
    model_path: str | Path,
    metrics: dict[str, float],
    params: dict[str, Any],
    tracking_uri: str = MLFLOW_URI,
) -> str:
    """Register the LightGBM ActiveLearner in MLflow and return the run_id."""
    import joblib

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="active_learner_v1") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        learner = joblib.load(model_path)
        mlflow.sklearn.log_model(
            learner._model, "lgbm_model", registered_model_name="AnomalyDetector-LightGBM"
        )
        if Path(model_path).exists():
            mlflow.log_artifact(str(model_path), artifact_path="model_files")
        return run.info.run_id


def log_drift_report(
    drift_score: float,
    features_drifted: list[str],
    html_report_path: str | Path | None = None,
    tracking_uri: str = MLFLOW_URI,
) -> str:
    """Log a drift detection result to MLflow and return the run_id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="drift_check") as run:
        mlflow.log_metrics(
            {
                "drift_score": round(drift_score, 4),
                "n_features_drifted": len(features_drifted),
            }
        )
        mlflow.log_param("features_drifted", ",".join(features_drifted) or "none")
        if html_report_path and Path(html_report_path).exists():
            mlflow.log_artifact(str(html_report_path), artifact_path="drift_reports")
        return run.info.run_id


def promote_model(
    model_name: str,
    version: int,
    stage: str = "Production",
    tracking_uri: str = MLFLOW_URI,
) -> None:
    """Promote a registered model version to a stage (Staging / Production)."""
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(model_name, stage, str(version))


def get_latest_model_info(
    model_name: str,
    tracking_uri: str = MLFLOW_URI,
) -> dict[str, Any]:
    """Return latest version info for a registered model."""
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return {"model_name": model_name, "versions": 0}
        latest = max(versions, key=lambda v: int(v.version))
        return {
            "model_name": model_name,
            "latest_version": int(latest.version),
            "run_id": latest.run_id,
            "status": latest.status,
        }
    except Exception:
        return {"model_name": model_name, "versions": 0}
