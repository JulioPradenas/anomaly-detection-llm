"""Anomaly detector wrapper with unified interface for sklearn-compatible models."""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

ModelType = Literal["isolation_forest", "lof", "ocsvm"]


class AnomalyDetector:
    """Unified interface for unsupervised anomaly detection models.

    Wraps IsolationForest, LocalOutlierFactor, and One-Class SVM with
    consistent fit/predict/score methods.
    """

    def __init__(
        self,
        model_type: ModelType = "isolation_forest",
        contamination: float = 0.02,
        random_state: int = 42,
        **model_kwargs,
    ):
        self.model_type = model_type
        self.contamination = contamination
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self._model = self._build_model()
        self.threshold_: float | None = None

    def _build_model(self):
        if self.model_type == "isolation_forest":
            return IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
                n_jobs=-1,
                **self.model_kwargs,
            )
        elif self.model_type == "lof":
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,  # required to call predict on new data
                n_jobs=-1,
                **self.model_kwargs,
            )
        elif self.model_type == "ocsvm":
            return OneClassSVM(
                nu=self.contamination,
                kernel="rbf",
                **self.model_kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: pd.DataFrame | np.ndarray) -> "AnomalyDetector":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._model.fit(X_arr)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return binary predictions: 1 = anomaly, 0 = normal."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        # sklearn convention: -1 = anomaly, 1 = normal
        raw = self._model.predict(X_arr)
        return (raw == -1).astype("int8")

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return anomaly scores — higher = more anomalous.

        Isolation Forest and LOF return negative scores (sklearn convention),
        so we negate them for intuitive interpretation.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if self.model_type == "ocsvm":
            # decision_function: negative = anomaly
            scores = -self._model.decision_function(X_arr)
        else:
            scores = -self._model.score_samples(X_arr)
        return scores

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        return joblib.load(path)
