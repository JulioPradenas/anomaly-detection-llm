"""Inference pipeline — bridges raw API requests to trained model + LLM."""

import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from api.schemas import AnomalyResult, DetectRequest, LogEntry
from src.data.preprocessor import add_severity_score
from src.llm.summarizer import LogSummarizer, build_anomaly_context
from src.models.detector import AnomalyDetector


DEFAULT_MODEL_PATH = Path("models/saved/lof_v1.joblib")


class AnomalyPipeline:
    """End-to-end inference pipeline: logs → features → detection → summaries."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        summarizer: LogSummarizer | None = None,
    ):
        self.model: AnomalyDetector | None = None
        self.summarizer = summarizer or LogSummarizer()
        self._model_path = model_path
        self._loaded = False

    def load_model(self) -> None:
        if self._model_path.exists():
            self.model = AnomalyDetector.load(self._model_path)
            self._loaded = True
        else:
            # No trained model yet — pipeline still works in demo mode
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, request: DetectRequest) -> list[AnomalyResult]:
        if not self._loaded or self.model is None:
            return []

        df = _logs_to_df(request.logs)
        if df.empty:
            return []

        feature_cols = [c for c in df.columns
                        if c not in {"timestamp", "node", "level", "component",
                                     "content", "label", "severity_score", "is_anomaly"}]
        if not feature_cols:
            return []

        X = df[feature_cols].fillna(0).values
        preds = self.model.predict(X)
        scores = self.model.score_samples(X)

        results = []
        for idx in np.where(preds == 1)[0]:
            row = df.iloc[idx]
            window_start = row["timestamp"] - timedelta(minutes=request.window_minutes / 2)
            window_end = row["timestamp"] + timedelta(minutes=request.window_minutes / 2)
            context_df = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)]

            context_entries = [
                LogEntry(
                    timestamp=r["timestamp"],
                    node=str(r["node"]),
                    level=str(r.get("level", "INFO")),
                    component=str(r.get("component", "UNKNOWN")),
                    content=str(r.get("content", "")),
                )
                for _, r in context_df.iterrows()
            ]

            results.append(AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                timestamp=row["timestamp"],
                node=str(row["node"]),
                anomaly_score=float(scores[idx]),
                is_anomaly=True,
                context_window=context_entries,
            ))

        return results


def _logs_to_df(logs: list[LogEntry]) -> pd.DataFrame:
    """Convert API LogEntry list to a minimal feature DataFrame."""
    if not logs:
        return pd.DataFrame()

    rows = [
        {
            "timestamp": e.timestamp,
            "node": e.node,
            "level": e.level,
            "component": e.component,
            "content": e.content,
            "label": e.label,
        }
        for e in logs
    ]
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_severity_score(df)
    df["is_anomaly"] = df["label"] != "-"

    # Minimal features for inference without full DuckDB pipeline
    df["error_flag"] = (df["severity_score"] >= 2).astype("int8")
    df["fatal_flag"] = (df["severity_score"] >= 3).astype("int8")

    return df
