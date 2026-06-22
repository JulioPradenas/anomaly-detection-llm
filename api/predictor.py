"""Inference pipeline — bridges raw API requests to trained model + LLM."""

import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from api.schemas import AnomalyResult, DetectRequest, LogEntry
from src.data.preprocessor import add_severity_score
from src.features.engineering import build_features
from src.llm.summarizer import LogSummarizer, build_anomaly_context
from src.models.detector import AnomalyDetector


DEFAULT_MODEL_PATH = Path("models/saved/lof_v1.joblib")
DEFAULT_FEATURES_PATH = Path("data/processed/features_train.parquet")
_NON_FEATURE_COLS = {"timestamp", "node", "is_anomaly"}


class AnomalyPipeline:
    """End-to-end inference pipeline: logs → features → detection → summaries."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        features_path: Path = DEFAULT_FEATURES_PATH,
        summarizer: LogSummarizer | None = None,
    ):
        self.model: AnomalyDetector | None = None
        self.summarizer = summarizer or LogSummarizer()
        self._model_path = model_path
        self._features_path = features_path
        self._feature_cols: list[str] | None = None
        self._loaded = False

    def load_model(self) -> None:
        if self._model_path.exists():
            self.model = AnomalyDetector.load(self._model_path)
            self._feature_cols = self._load_feature_schema()
            self._loaded = True
        else:
            # No trained model yet — pipeline still works in demo mode
            self._loaded = False

    def _load_feature_schema(self) -> list[str] | None:
        """Canonical feature column order the model was trained on."""
        if not self._features_path.exists():
            return None
        names = pq.ParquetFile(self._features_path).schema.names
        return [c for c in names if c not in _NON_FEATURE_COLS]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, request: DetectRequest) -> list[AnomalyResult]:
        if not self._loaded or self.model is None:
            return []

        df = _logs_to_df(request.logs)
        if df.empty:
            return []

        feat_df, _ = build_features(df, fit_scaler=True)
        # Align to the exact schema/order the model was trained on; the request
        # may not contain every component, so missing columns default to 0.
        if self._feature_cols is not None:
            X = feat_df.reindex(columns=self._feature_cols, fill_value=0).fillna(0).values
        else:
            X = feat_df.drop(columns=list(_NON_FEATURE_COLS), errors="ignore").fillna(0).values

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
