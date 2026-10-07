"""
Behavioral tests for the from-scratch SVM (src/models/svm_models.py).

Each test targets one observable behavior of the public API: fit/predict
accuracy, kernel/gamma variants, label handling, support-vector
attributes, decision_function conventions, multiclass one-vs-one
aggregation, hyperparameter validation, and determinism.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.models.svm_models import SVMClassifierScratch


@pytest.fixture
def linear_data():
    """Two well-separated blobs: a linear kernel must be near-perfect."""
    X, y = make_blobs(
        n_samples=240,
        centers=[(-2, -2), (2, 2)],
        cluster_std=0.8,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def moons_data():
    """Interleaved half-moons: requires a non-linear kernel."""
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def multiclass_data():
    """Three well-separated blobs for the one-vs-one path."""
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=0.8, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# --- Accuracy on canonical problems ---------------------------------------


def test_toy_8point_perfect_separation():
    """
    Hand-built 8-point problem that must separate exactly; guards the SMO
    pair-update math (L/H bounds, eta sign, b-branch logic).
    """
    X = np.array(
        [
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [3.0, 3.0],
            [-2.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -2.0],
            [-3.0, -3.0],
        ]
    )
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    svm = SVMClassifierScratch(kernel="linear", C=1.0, max_iter=100)
    svm.fit(X, y)
    assert svm.score(X, y) == 1.0


def test_linear_separable_accuracy(linear_data):
    X_train, X_test, y_train, y_test = linear_data
    svm = SVMClassifierScratch(kernel="linear", C=1.0, max_iter=200)
    assert svm.fit(X_train, y_train) is svm
    assert svm.score(X_test, y_test) > 0.9


def test_rbf_moons_accuracy(moons_data):
    X_train, X_test, y_train, y_test = moons_data
    svm = SVMClassifierScratch(kernel="rbf", gamma="scale", C=1.0)
    svm.fit(X_train, y_train)

    # context: sklearn with the same hyperparameters must also clear the bar
    sk_acc = SVC(kernel="rbf", gamma="scale", C=1.0).fit(X_train, y_train).score(
        X_test, y_test
    )
    assert sk_acc > 0.85
    assert svm.score(X_test, y_test) > 0.85


def test_kernel_variants_fit_and_predict(linear_data):
    X_train, X_test, y_train, y_test = linear_data
    for kernel in ("linear", "poly", "rbf"):
        svm = SVMClassifierScratch(kernel=kernel, C=1.0, max_iter=200)
        svm.fit(X_train, y_train)
        assert svm.predict(X_test).shape == (len(X_test),)
        assert svm.score(X_test, y_test) > 0.8


@pytest.mark.parametrize("gamma", ["scale", "auto", 0.1])
def test_gamma_variants_fit(moons_data, gamma):
    X_train, X_test, y_train, _ = moons_data
    svm = SVMClassifierScratch(kernel="rbf", gamma=gamma, max_iter=200)
    svm.fit(X_train, y_train)
    assert svm.predict(X_test).shape == (len(X_test),)


# --- Label handling --------------------------------------------------------


def test_non_contiguous_labels():
    rng = np.random.RandomState(0)
    X = rng.randn(160, 2)
    y = np.where(X[:, 0] > 0, 10, 20)  # labels 10/20, not 0/1
    svm = SVMClassifierScratch(kernel="linear", max_iter=200)
    svm.fit(X, y)
    assert np.array_equal(svm.classes_, [10, 20])
    assert set(np.unique(svm.predict(X))) == {10, 20}
    assert svm.score(X, y) > 0.95


# --- Support-vector attributes ---------------------------------------------


def test_binary_attribute_shapes(linear_data):
    X_train, _, y_train, _ = linear_data
    svm = SVMClassifierScratch(kernel="linear", max_iter=200)
    svm.fit(X_train, y_train)

    n_train, n_features = X_train.shape
    n_sv = len(svm.support_)

    assert len(svm.classes_) == 2
    assert isinstance(svm.intercept_, float)
    assert np.isfinite(svm.intercept_)
    assert svm.dual_coef_.shape == (n_sv,)
    assert svm.support_vectors_.shape == (n_sv, n_features)
    assert svm.n_support_.shape == (2,)
    assert np.issubdtype(svm.n_support_.dtype, np.integer)
    assert svm.n_support_.sum() == n_sv
    assert len(svm.dual_coef_) == len(svm.support_vectors_)

    # support_ holds original row indices, sorted and in range
    assert np.issubdtype(svm.support_.dtype, np.integer)
    assert np.array_equal(svm.support_, np.sort(svm.support_))
    assert svm.support_[0] >= 0 and svm.support_[-1] < n_train

    # signed dual coefficients: class-0 SVs negative, class-1 SVs positive
    y_idx = np.searchsorted(svm.classes_, y_train)
    sv_classes = y_idx[svm.support_]
    assert np.array_equal(np.sign(svm.dual_coef_) > 0, sv_classes == 1)


def test_support_vectors_strict_subset_recover_labels(linear_data):
    X_train, _, y_train, _ = linear_data
    svm = SVMClassifierScratch(kernel="linear", max_iter=200)
    svm.fit(X_train, y_train)

    # points at or inside the margin must number fewer than all rows
    assert 0 < len(svm.support_) < len(X_train)
    assert np.array_equal(
        svm.predict(X_train[svm.support_]), y_train[svm.support_]
    )


# --- decision_function ------------------------------------------------------


def test_decision_function_binary_conventions(linear_data):
    X_train, X_test, y_train, _ = linear_data
    svm = SVMClassifierScratch(kernel="linear", max_iter=200)
    svm.fit(X_train, y_train)

    decision = svm.decision_function(X_test)
    assert decision.shape == (len(X_test),)
    assert np.issubdtype(decision.dtype, np.floating)

    preds = svm.predict(X_test)
    # positive side of the hyperplane is classes_[1]
    assert ((decision > 0) == (preds == svm.classes_[1])).all()


# --- Multiclass (one-vs-one) -------------------------------------------------


def test_multiclass_one_vs_one(multiclass_data):
    X_train, X_test, y_train, y_test = multiclass_data
    svm = SVMClassifierScratch(kernel="rbf", C=1.0)
    svm.fit(X_train, y_train)

    assert np.array_equal(svm.classes_, np.unique(y_train))
    preds = svm.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset(set(svm.classes_))
    assert svm.score(X_test, y_test) > 0.85

    n_pairs = 3  # C(3, 2) one-vs-one pairs
    n_sv = len(svm.support_)
    assert len(svm.intercept_) == n_pairs
    assert len(svm.dual_coef_) == n_sv
    assert svm.support_vectors_.shape == (n_sv, X_train.shape[1])
    assert svm.n_support_.sum() == n_sv

    decision = svm.decision_function(X_test)
    assert decision.shape == (len(X_test), len(svm.classes_))
    assert np.isfinite(decision).all()
    # prediction is the per-class aggregated score argmax on clean data
    assert np.array_equal(
        svm.classes_[decision.argmax(axis=1)], preds
    )


# --- Determinism -------------------------------------------------------------


def test_same_seed_deterministic(moons_data):
    X_train, X_test, y_train, _ = moons_data
    svm1 = SVMClassifierScratch(kernel="rbf", random_state=7)
    svm2 = SVMClassifierScratch(kernel="rbf", random_state=7)
    svm1.fit(X_train, y_train)
    svm2.fit(X_train, y_train)
    assert np.array_equal(svm1.predict(X_test), svm2.predict(X_test))
    assert np.array_equal(svm1.dual_coef_, svm2.dual_coef_)


# --- Hyperparameter validation -----------------------------------------------


def test_invalid_kernel_raises(moons_data):
    X_train, _, y_train, _ = moons_data
    svm = SVMClassifierScratch(kernel="bogus")
    with pytest.raises(ValueError, match="kernel"):
        svm.fit(X_train, y_train)


def test_invalid_gamma_raises(linear_data):
    X_train, _, y_train, _ = linear_data
    svm = SVMClassifierScratch(kernel="rbf", gamma="bogus")
    with pytest.raises(ValueError, match="gamma"):
        svm.fit(X_train, y_train)


@pytest.mark.parametrize("bad_c", [0, -1.0])
def test_non_positive_C_raises(linear_data, bad_c):
    X_train, _, y_train, _ = linear_data
    svm = SVMClassifierScratch(C=bad_c)
    with pytest.raises(ValueError, match="C"):
        svm.fit(X_train, y_train)


@pytest.mark.parametrize("bad_degree", [0, -2, 1.5])
def test_invalid_degree_raises(linear_data, bad_degree):
    X_train, _, y_train, _ = linear_data
    svm = SVMClassifierScratch(kernel="poly", degree=bad_degree)
    with pytest.raises(ValueError, match="degree"):
        svm.fit(X_train, y_train)


@pytest.mark.parametrize("bad_tol", [0.0, -1e-3])
def test_non_positive_tol_raises(linear_data, bad_tol):
    X_train, _, y_train, _ = linear_data
    svm = SVMClassifierScratch(tol=bad_tol)
    with pytest.raises(ValueError, match="tol"):
        svm.fit(X_train, y_train)


@pytest.mark.parametrize("bad_max_iter", [0, -5])
def test_invalid_max_iter_raises(linear_data, bad_max_iter):
    X_train, _, y_train, _ = linear_data
    svm = SVMClassifierScratch(max_iter=bad_max_iter)
    with pytest.raises(ValueError, match="max_iter"):
        svm.fit(X_train, y_train)


def test_predict_before_fit_raises(linear_data):
    X_train, _, _, _ = linear_data
    svm = SVMClassifierScratch()
    with pytest.raises(ValueError, match="not fitted"):
        svm.predict(X_train[:5])
