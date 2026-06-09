"""Pydantic v2 schemas for the anomaly detection API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    timestamp: datetime
    node: str
    level: str = "INFO"
    component: str = "UNKNOWN"
    content: str = ""
    label: str = "-"


class AnomalyResult(BaseModel):
    anomaly_id: str
    timestamp: datetime
    node: str
    anomaly_score: float
    is_anomaly: bool
    context_window: list[LogEntry] = Field(default_factory=list)


class DetectRequest(BaseModel):
    logs: list[LogEntry]
    window_minutes: int = Field(default=5, ge=1, le=60)


class DetectResponse(BaseModel):
    anomalies: list[AnomalyResult]
    total_logs: int
    total_anomalies: int
    anomaly_rate: float
    summary_available: bool


class SummarizeRequest(BaseModel):
    anomaly_id: str
    context_window: list[LogEntry]
    anomaly_score: float = 0.5


class SummarizeResponse(BaseModel):
    anomaly_id: str
    resumen: str
    severidad: str
    causa_probable: str
    accion_recomendada: str
    response_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    llm_available: bool
    model_type: str


class RetrainRequest(BaseModel):
    use_llm_labels: bool = True
    min_samples: int = Field(default=100, ge=10)


class RetrainResponse(BaseModel):
    model_version: str
    f1_before: float
    f1_after: float
    n_samples_used: int
    retrain_time_ms: float
    labels_source: str
