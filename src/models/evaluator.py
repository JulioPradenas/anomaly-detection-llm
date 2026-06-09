"""Evaluation utilities for anomaly detection models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    model_name: str
    precision: float
    recall: float
    f1: float
    aupr: float
    n_true_anomalies: int
    n_predicted_anomalies: int

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "aupr": round(self.aupr, 4),
            "n_true": self.n_true_anomalies,
            "n_pred": self.n_predicted_anomalies,
        }


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    model_name: str = "model",
) -> EvaluationResult:
    """Compute precision, recall, F1 and AUPR for a set of predictions.

    Args:
        y_true: Ground truth binary labels (1 = anomaly).
        y_pred: Binary predictions (1 = anomaly).
        scores: Continuous anomaly scores (higher = more anomalous).
        model_name: Label for the result.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)

    return EvaluationResult(
        model_name=model_name,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        aupr=average_precision_score(y_true, scores),
        n_true_anomalies=int(y_true.sum()),
        n_predicted_anomalies=int(y_pred.sum()),
    )


def compare_models(results: list[EvaluationResult]) -> pd.DataFrame:
    """Return a comparison table sorted by F1 descending."""
    rows = [r.as_dict() for r in results]
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return df


def get_pr_curve(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for plotting a PR curve."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true), np.asarray(scores))
    return precision, recall, thresholds
