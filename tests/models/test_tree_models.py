"""
Behavioral tests for the from-scratch decision trees (src/models/tree_models.py).

Each test targets one observable behavior of the public API. Focus here is
the classifier's probabilistic surface (``predict_proba``, class-fraction
leaves, arbitrary label handling) — the split machinery itself is exercised
end-to-end by the ensemble tests.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from src.models.tree_models import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
)


@pytest.fixture
def clf_data():
    X, y = make_blobs(
        n_samples=240, centers=[(-2, -2), (2, 2)], cluster_std=1.0, random_state=0
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


# --- predict_proba ---------------------------------------------------------


def test_predict_proba_shape_matches_classes(clf_data):
    X_train, X_test, y_train, _ = clf_data
    tree = DecisionTreeClassifierScratch(max_depth=3).fit(X_train, y_train)
    proba = tree.predict_proba(X_test)
    n_classes = len(np.unique(y_train))
    assert proba.shape == (len(X_test), n_classes)


def test_predict_proba_rows_sum_to_one_and_non_negative(clf_data):
    X_train, X_test, y_train, _ = clf_data
    tree = DecisionTreeClassifierScratch(max_depth=3).fit(X_train, y_train)
    proba = tree.predict_proba(X_test)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_argmax_of_proba_matches_predict(clf_data):
    # a leaf's fraction argmax IS the majority vote predict returns — the two
    # surfaces must agree row by row
    X_train, X_test, y_train, _ = clf_data
    tree = DecisionTreeClassifierScratch(max_depth=3).fit(X_train, y_train)
    proba = tree.predict_proba(X_test)
    pred = tree.predict(X_test)
    assert np.array_equal(tree.classes_[proba.argmax(axis=1)], pred)


def test_leaf_fractions_are_exact_on_1d_data():
    # 1D, one depth-1 split with a unique best threshold (2): the left leaf
    # is pure, the right leaf holds y = {1, 0, 0} — fractions [2/3, 1/3]
    X = np.array([[1.0], [2.0], [3.5], [4.0], [5.0]])
    y = np.array([0, 0, 1, 0, 0])
    tree = DecisionTreeClassifierScratch(max_depth=1).fit(X, y)
    proba = tree.predict_proba(X)
    assert np.allclose(proba[:2], [[1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(proba[2:], [[2 / 3, 1 / 3]] * 3)


def test_non_contiguous_labels_preserved(clf_data):
    # labels 10/20 (not 0/1): predict and proba columns must both use the
    # original labels, in sorted classes_ order
    X_train, X_test, y_train, y_test = clf_data
    y_train_big = np.where(y_train == 0, 10, 20)
    tree = DecisionTreeClassifierScratch(max_depth=3).fit(X_train, y_train_big)
    pred = tree.predict(X_test)
    assert set(np.unique(pred)).issubset({10, 20})
    proba = tree.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # accuracy must survive the relabeling: predictions equal the same tree's
    # 0/1 predictions mapped to {10, 20}
    ref = DecisionTreeClassifierScratch(max_depth=3).fit(X_train, y_train)
    assert np.array_equal(
        pred, np.where(ref.predict(X_test) == 0, 10, 20)
    )


# --- behavior preserved after the proba addition ----------------------------


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classifier_accuracy_on_easy_problem(clf_data, criterion):
    X_train, X_test, y_train, y_test = clf_data
    tree = DecisionTreeClassifierScratch(max_depth=5, criterion=criterion)
    tree.fit(X_train, y_train)
    assert tree.score(X_test, y_test) > 0.8


def test_regressor_still_predicts_and_scores():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(200, 2))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 1 + rng.normal(0, 0.5, 200)
    tree = DecisionTreeRegressorScratch(max_depth=6).fit(X, y)
    assert tree.score(X, y) > 0.8
