"""Tests for evaluation utilities."""

import numpy as np
import pytest

from src.models.evaluator import EvaluationResult, compare_models, evaluate, get_pr_curve


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
    return y_true, y_pred, scores


@pytest.fixture
def imperfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0])  # 1 FP, 1 FN
    scores = np.array([0.1, 0.6, 0.8, 0.3, 0.9, 0.2])
    return y_true, y_pred, scores


def test_evaluate_perfect(perfect_predictions):
    y_true, y_pred, scores = perfect_predictions
    result = evaluate(y_true, y_pred, scores, model_name="test")
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.aupr == 1.0


def test_evaluate_imperfect(imperfect_predictions):
    y_true, y_pred, scores = imperfect_predictions
    result = evaluate(y_true, y_pred, scores, model_name="test_model")
    assert 0 < result.precision <= 1
    assert 0 < result.recall <= 1
    assert 0 < result.f1 <= 1
    assert result.n_true_anomalies == 3
    assert result.n_predicted_anomalies == 3  # [0,1,1,0,1,0] has 3 positive predictions


def test_evaluate_model_name():
    result = evaluate(np.array([0, 1]), np.array([0, 1]), np.array([0.1, 0.9]),
                      model_name="my_model")
    assert result.model_name == "my_model"


def test_evaluation_result_as_dict():
    result = EvaluationResult(
        model_name="IF", precision=0.8, recall=0.7, f1=0.75, aupr=0.82,
        n_true_anomalies=10, n_predicted_anomalies=9
    )
    d = result.as_dict()
    assert d["model"] == "IF"
    assert d["f1"] == 0.75
    assert "aupr" in d


def test_compare_models_sorted_by_f1():
    results = [
        EvaluationResult("LOF", 0.6, 0.5, 0.55, 0.6, 10, 9),
        EvaluationResult("IF", 0.8, 0.75, 0.77, 0.85, 10, 9),
        EvaluationResult("OCSVM", 0.65, 0.6, 0.62, 0.7, 10, 9),
    ]
    df = compare_models(results)
    assert df.iloc[0]["model"] == "IF"
    assert df.iloc[-1]["model"] == "LOF"


def test_get_pr_curve_shape():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.7, 0.9])
    precision, recall, thresholds = get_pr_curve(y_true, scores)
    assert len(precision) == len(recall)
    assert len(thresholds) == len(precision) - 1
