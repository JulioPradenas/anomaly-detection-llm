"""Active learner: LightGBM trained on LLM-generated weak labels."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class ActiveLearner:
    """Binary classifier trained on LLM-derived labels (HIGH/CRITICAL=1, LOW/MEDIUM=0).

    Wraps LightGBM with a consistent fit/predict interface that mirrors
    AnomalyDetector so both can be evaluated with the same evaluator.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self._feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
    ) -> "ActiveLearner":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        self._model.fit(X_arr, y_arr)
        if feature_names is not None:
            self._feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return binary predictions: 1 = anomaly (HIGH/CRITICAL), 0 = normal."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict(X_arr).astype("int8")

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return probability of anomaly class — used as anomaly score."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict_proba(X_arr)[:, 1]

    def score_samples(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Alias for predict_proba — compatible with AnomalyDetector interface."""
        return self.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        if not hasattr(self._model, "feature_importances_"):
            raise RuntimeError("Model not fitted yet.")
        names = self._feature_names or [
            f"f{i}" for i in range(len(self._model.feature_importances_))
        ]
        return pd.Series(self._model.feature_importances_, index=names).sort_values(ascending=False)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ActiveLearner":
        return joblib.load(path)


def encode_llm_labels(severities: pd.Series) -> pd.Series:
    """Map LLM severity strings to binary labels.

    HIGH / CRITICAL → 1  (anomaly requiring action)
    LOW / MEDIUM    → 0  (informational or minor)
    UNKNOWN         → NaN (excluded from training)
    """
    mapping = {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 0, "LOW": 0}
    return severities.map(mapping)  # UNKNOWN → NaN automatically
