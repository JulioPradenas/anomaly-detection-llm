"""Tests for ActiveLearner — LightGBM trained on LLM-generated labels."""

import numpy as np
import pandas as pd
import pytest

from src.models.active_learner import ActiveLearner, encode_llm_labels


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    feature_names = [f"feat_{i}" for i in range(10)]
    return X, y, feature_names


def test_fit_predict(small_dataset):
    X, y, names = small_dataset
    learner = ActiveLearner(n_estimators=10)
    learner.fit(X, y, feature_names=names)
    preds = learner.predict(X)
    assert preds.shape == (200,)
    assert set(preds).issubset({0, 1})


def test_predict_proba_range(small_dataset):
    X, y, names = small_dataset
    learner = ActiveLearner(n_estimators=10).fit(X, y)
    proba = learner.predict_proba(X)
    assert proba.shape == (200,)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_score_samples_alias(small_dataset):
    X, y, _ = small_dataset
    learner = ActiveLearner(n_estimators=10).fit(X, y)
    assert np.allclose(learner.score_samples(X), learner.predict_proba(X))


def test_feature_importance(small_dataset):
    X, y, names = small_dataset
    learner = ActiveLearner(n_estimators=10).fit(X, y, feature_names=names)
    imp = learner.get_feature_importance()
    assert isinstance(imp, pd.Series)
    assert len(imp) == 10
    assert imp.index.tolist() == sorted(imp.index.tolist(), key=lambda n: imp[n], reverse=True)


def test_feature_importance_before_fit_raises():
    learner = ActiveLearner()
    with pytest.raises(RuntimeError):
        learner.get_feature_importance()


def test_dataframe_input(small_dataset):
    X, y, names = small_dataset
    df = pd.DataFrame(X, columns=names)
    labels = pd.Series(y)
    learner = ActiveLearner(n_estimators=10).fit(df, labels)
    assert learner._feature_names == names
    preds = learner.predict(df)
    assert preds.shape == (200,)


def test_save_load(tmp_path, small_dataset):
    X, y, names = small_dataset
    learner = ActiveLearner(n_estimators=10).fit(X, y, feature_names=names)
    path = tmp_path / "active_learner.joblib"
    learner.save(path)
    loaded = ActiveLearner.load(path)
    assert np.array_equal(learner.predict(X), loaded.predict(X))


def test_get_params():
    learner = ActiveLearner(n_estimators=50, learning_rate=0.1)
    params = learner.get_params()
    assert params["n_estimators"] == 50
    assert params["learning_rate"] == 0.1


# ── encode_llm_labels ────────────────────────────────────────────────────────


def test_encode_high_critical_to_1():
    s = pd.Series(["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"])
    labels = encode_llm_labels(s)
    assert labels[0] == 1
    assert labels[1] == 1
    assert labels[2] == 0
    assert labels[3] == 0
    assert pd.isna(labels[4])


def test_encode_drops_unknown_on_dropna():
    s = pd.Series(["CRITICAL", "UNKNOWN", "LOW"])
    labels = encode_llm_labels(s).dropna()
    assert len(labels) == 2
    assert set(labels.values) == {0.0, 1.0}
