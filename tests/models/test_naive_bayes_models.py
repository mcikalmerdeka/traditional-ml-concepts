"""
Behavioral tests for the from-scratch Gaussian Naive Bayes
(src/models/naive_bayes_models.py).

Each test targets one observable behavior of the public API: exact
closed-form agreement with scikit-learn, fitted-attribute shapes and
values, class handling, the smoothing lesson, degenerate single-class
fits, hyperparameter validation, and determinism.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.naive_bayes import GaussianNB

from src.models.naive_bayes_models import GaussianNaiveBayesScratch


@pytest.fixture
def moons_data():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    return X, y


@pytest.fixture
def blobs_data():
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=2.2, random_state=0)
    return X, y


# --- Exact closed-form agreement with scikit-learn ------------------------


def test_predict_proba_matches_sklearn_on_moons(moons_data):
    X, y = moons_data
    sk = GaussianNB(var_smoothing=1e-9).fit(X, y)
    nb = GaussianNaiveBayesScratch(var_smoothing=1e-9).fit(X, y)
    assert np.max(np.abs(nb.predict_proba(X) - sk.predict_proba(X))) < 1e-9


def test_predict_proba_matches_sklearn_on_blobs(blobs_data):
    X, y = blobs_data
    sk = GaussianNB(var_smoothing=1e-9).fit(X, y)
    nb = GaussianNaiveBayesScratch(var_smoothing=1e-9).fit(X, y)
    assert np.max(np.abs(nb.predict_proba(X) - sk.predict_proba(X))) < 1e-9


def test_predict_matches_sklearn(moons_data, blobs_data):
    for X, y in (moons_data, blobs_data):
        sk = GaussianNB(var_smoothing=1e-9).fit(X, y)
        nb = GaussianNaiveBayesScratch(var_smoothing=1e-9).fit(X, y)
        assert np.array_equal(nb.predict(X), sk.predict(X))


# --- Fitted attributes ----------------------------------------------------


def test_fitted_attributes_match_sklearn(moons_data, blobs_data):
    for X, y in (moons_data, blobs_data):
        sk = GaussianNB(var_smoothing=1e-9).fit(X, y)
        nb = GaussianNaiveBayesScratch(var_smoothing=1e-9).fit(X, y)

        assert nb.classes_.shape == sk.classes_.shape
        assert np.array_equal(nb.classes_, sk.classes_)  # sorted class order
        assert nb.class_prior_.shape == sk.class_prior_.shape
        assert np.allclose(
            nb.class_prior_, sk.class_prior_, atol=1e-12, rtol=0.0
        )
        assert nb.theta_.shape == sk.theta_.shape
        assert np.allclose(nb.theta_, sk.theta_, atol=1e-12, rtol=0.0)
        assert nb.var_.shape == sk.var_.shape
        assert np.allclose(nb.var_, sk.var_, atol=1e-12, rtol=0.0)
        assert nb.epsilon_.shape == sk.epsilon_.shape
        assert np.allclose(nb.epsilon_, sk.epsilon_, atol=1e-12, rtol=0.0)


def test_var_includes_smoothing_epsilon(moons_data):
    X, y = moons_data
    nb = GaussianNaiveBayesScratch(var_smoothing=1e-9).fit(X, y)
    # var_ must be per-class population variance plus the shared epsilon
    raw_var = np.stack(
        [X[y == c].var(axis=0) for c in range(len(nb.classes_))]
    )
    assert nb.epsilon_ > 0
    assert np.allclose(
        nb.var_, raw_var + nb.epsilon_, atol=1e-12, rtol=0.0
    )


# --- Class handling -------------------------------------------------------


def test_non_contiguous_labels_preserved(moons_data):
    X, y = moons_data
    y_relabelled = np.where(y == 0, 10, 20)
    sk = GaussianNB().fit(X, y_relabelled)
    nb = GaussianNaiveBayesScratch().fit(X, y_relabelled)
    assert set(np.unique(nb.predict(X))) == {10, 20}
    assert np.array_equal(nb.predict(X), sk.predict(X))


def test_single_class_fit(moons_data):
    X, y = moons_data
    y_single = np.full(len(y), 7)
    nb = GaussianNaiveBayesScratch().fit(X, y_single)
    assert np.array_equal(nb.classes_, [7])
    assert np.allclose(nb.class_prior_, [1.0])
    assert nb.predict_proba(X).shape == (len(X), 1)
    assert np.allclose(nb.predict_proba(X), 1.0)
    assert np.all(nb.predict(X) == 7)


def test_fit_returns_self(moons_data):
    X, y = moons_data
    nb = GaussianNaiveBayesScratch()
    assert nb.fit(X, y) is nb


# --- Smoothing behavior (the app card's teaching point) -------------------


def test_larger_smoothing_flattens_proba_rows(moons_data):
    X, y = moons_data
    tight = GaussianNaiveBayesScratch(var_smoothing=1e-9).fit(X, y)
    loose = GaussianNaiveBayesScratch(var_smoothing=1e-3).fit(X, y)
    tight_max = tight.predict_proba(X).max(axis=1)
    loose_max = loose.predict_proba(X).max(axis=1)
    assert loose_max.mean() < tight_max.mean()


# --- Score sanity ---------------------------------------------------------


def test_score_on_separated_blobs():
    # The pinned agreement fixture (cluster_std=2.2, random_state=0)
    # draws heavily overlapping blob centres -- sklearn's own GaussianNB
    # scores exactly 0.708 there, so a >0.9 sanity threshold needs a
    # genuinely separated draw.
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=1.2, random_state=7)
    nb = GaussianNaiveBayesScratch().fit(X, y)
    assert nb.score(X, y) > 0.9


# --- Validation -----------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_var_smoothing_raises(moons_data, bad):
    X, y = moons_data
    nb = GaussianNaiveBayesScratch(var_smoothing=bad)
    with pytest.raises(ValueError, match="var_smoothing"):
        nb.fit(X, y)


# --- Determinism ----------------------------------------------------------


def test_repeated_fits_are_identical(moons_data):
    X, y = moons_data
    nb1 = GaussianNaiveBayesScratch().fit(X, y)
    nb2 = GaussianNaiveBayesScratch().fit(X, y)
    assert np.array_equal(nb1.theta_, nb2.theta_)
    assert np.array_equal(nb1.var_, nb2.var_)
    assert np.array_equal(nb1.class_prior_, nb2.class_prior_)
    assert np.array_equal(nb1.predict(X), nb2.predict(X))
    assert np.array_equal(nb1.predict_proba(X), nb2.predict_proba(X))
