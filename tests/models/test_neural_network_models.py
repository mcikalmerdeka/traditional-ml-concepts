"""
Behavioral tests for the from-scratch MLP classifier
(src/models/neural_network_models.py).

Each test targets one observable behavior of the public API:
learnability on a standard dataset, numerical correctness of
backpropagation (finite differences), probability outputs, label
handling, loss bookkeeping, early stopping, regularization, hidden-size
flexibility, determinism, and hyperparameter validation.

sklearn's MLPClassifier appears only as a contextual reference (the
dataset is learnable). Exact scratch-vs-sklearn agreement is NOT
asserted: the two use different optimizers and init details, so only
behavioral thresholds are meaningful.

The gradient-check test calls the private ``_loss_and_gradient`` helper
on purpose: backprop correctness is an internal property with no
public-API observable other than "training works", and a direct numeric
check is the strongest (and cheapest) possible assertion for it.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.models.neural_network_models import MLPClassifierScratch


@pytest.fixture
def moons_data():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# --- Learnability ---------------------------------------------------------


def test_moons_learnable(moons_data):
    X_train, X_test, y_train, y_test = moons_data
    mlp = MLPClassifierScratch(
        hidden_layer_sizes=(16,),
        activation="relu",
        max_iter=500,
        random_state=0,
    )
    mlp.fit(X_train, y_train)
    assert mlp.score(X_test, y_test) > 0.85

    # Contextual reference: the standard implementation clears 0.8 on the
    # same split, i.e. the dataset is learnable and the comparison target
    # is in the right ballpark (sklearn typically reports ~0.9 here).
    ref = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0)
    ref.fit(X_train, y_train)
    assert ref.score(X_test, y_test) > 0.80


def test_gradient_check_finite_differences():
    # Tiny deterministic 2-2-2 tanh net; verify analytic gradients from
    # backprop against central finite differences of the true objective
    # (cross-entropy + L2 on weights, biases excluded).
    X, y = make_blobs(n_samples=30, centers=2, random_state=0)
    mlp = MLPClassifierScratch(
        hidden_layer_sizes=(2, 2),
        activation="tanh",
        alpha=0.25,
        max_iter=20,
        random_state=7,
    )
    mlp.fit(X, y)
    y_idx = np.searchsorted(mlp.classes_, y)

    def objective():
        proba = mlp.predict_proba(X)
        ce = -np.mean(np.log(proba[np.arange(len(y_idx)), y_idx]))
        l2 = sum(float((W**2).sum()) for W in mlp.coefs_)
        return ce + mlp.alpha * l2

    analytic_loss, grad_ws, grad_bs = mlp._loss_and_gradient(X, y_idx)
    assert np.isclose(analytic_loss, objective(), atol=1e-9)

    h = 1e-6
    for layer, i, j in [(0, 0, 0), (0, 1, 1), (1, 1, 0)]:
        w = mlp.coefs_[layer][i, j]
        mlp.coefs_[layer][i, j] = w + h
        f_plus = objective()
        mlp.coefs_[layer][i, j] = w - h
        f_minus = objective()
        mlp.coefs_[layer][i, j] = w
        numeric = (f_plus - f_minus) / (2 * h)
        assert abs(numeric - grad_ws[layer][i, j]) < 1e-6, (
            f"weight grad mismatch at layer {layer} entry ({i}, {j}): "
            f"analytic={grad_ws[layer][i, j]}, numeric={numeric}"
        )

    # Biases are part of backprop but NOT of the L2 term.
    layer, i = 1, 1
    b = mlp.intercepts_[layer][i]
    mlp.intercepts_[layer][i] = b + h
    f_plus = objective()
    mlp.intercepts_[layer][i] = b - h
    f_minus = objective()
    mlp.intercepts_[layer][i] = b
    numeric = (f_plus - f_minus) / (2 * h)
    assert abs(numeric - grad_bs[layer][i]) < 1e-6, (
        f"bias grad mismatch at layer {layer} unit {i}: "
        f"analytic={grad_bs[layer][i]}, numeric={numeric}"
    )


# --- Predictions and probabilities ----------------------------------------


def test_multiclass_proba_shape_and_consistency():
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=1.0, random_state=0)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    mlp = MLPClassifierScratch(
        hidden_layer_sizes=(16,), max_iter=300, random_state=0
    )
    mlp.fit(X_train, y_train)
    proba = mlp.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.max(np.abs(proba.sum(axis=1) - 1.0)) < 1e-9
    assert (proba >= 0).all()
    # predict() must be the class-map of proba's argmax columns.
    assert np.array_equal(mlp.predict(X_test), mlp.classes_[proba.argmax(axis=1)])


def test_non_contiguous_labels(moons_data):
    X_train, X_test, y_train, _ = moons_data
    y_train_10_20 = np.where(y_train == 0, 10, 20)
    mlp = MLPClassifierScratch(
        hidden_layer_sizes=(16,), max_iter=300, random_state=0
    )
    mlp.fit(X_train, y_train_10_20)
    assert set(np.unique(mlp.predict(X_test))) == {10, 20}
    assert np.array_equal(mlp.classes_, np.array([10, 20]))
    # Columns of proba map to sorted classes_, so argmax must round-trip.
    proba = mlp.predict_proba(X_test)
    assert np.array_equal(mlp.predict(X_test), mlp.classes_[proba.argmax(axis=1)])


# --- Training bookkeeping ---------------------------------------------------


def test_loss_curve_decreases_and_matches_n_iter(moons_data):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(
        hidden_layer_sizes=(16,), max_iter=200, random_state=0
    )
    mlp.fit(X_train, y_train)
    assert len(mlp.loss_curve_) <= 200
    assert mlp.loss_curve_[0] > mlp.loss_curve_[-1]
    assert mlp.n_iter_ == len(mlp.loss_curve_)
    assert isinstance(mlp.loss_, float)


def test_early_stopping_stops_before_max_iter():
    X, y = make_blobs(n_samples=240, centers=2, cluster_std=0.5, random_state=0)
    mlp = MLPClassifierScratch(
        hidden_layer_sizes=(16,),
        max_iter=2000,
        tol=1e-3,
        random_state=0,
    )
    mlp.fit(X, y)
    assert mlp.n_iter_ < 2000
    assert len(mlp.loss_curve_) == mlp.n_iter_


def test_strong_alpha_shrinks_weights(moons_data):
    X_train, _, y_train, _ = moons_data
    weight_norms = {}
    for alpha in (1e-6, 1.0):
        mlp = MLPClassifierScratch(
            hidden_layer_sizes=(32,), alpha=alpha, max_iter=300, random_state=0
        )
        mlp.fit(X_train, y_train)
        weight_norms[alpha] = sum(float((W**2).sum()) for W in mlp.coefs_)
    assert weight_norms[1.0] < weight_norms[1e-6]


# --- API flexibility --------------------------------------------------------


def test_int_hidden_layer_sizes_equivalent_to_tuple(moons_data):
    X_train, X_test, y_train, _ = moons_data
    mlp_int = MLPClassifierScratch(hidden_layer_sizes=16, max_iter=100, random_state=7)
    mlp_tuple = MLPClassifierScratch(
        hidden_layer_sizes=(16,), max_iter=100, random_state=7
    )
    mlp_int.fit(X_train, y_train)
    mlp_tuple.fit(X_train, y_train)
    assert [W.shape for W in mlp_int.coefs_] == [W.shape for W in mlp_tuple.coefs_]
    # Same architecture + same seed -> identical training trajectory.
    assert mlp_int.loss_curve_ == mlp_tuple.loss_curve_
    assert np.array_equal(mlp_int.predict(X_test), mlp_tuple.predict(X_test))


def test_same_seed_gives_identical_results(moons_data):
    X_train, X_test, y_train, _ = moons_data
    mlp1 = MLPClassifierScratch(hidden_layer_sizes=(16,), max_iter=200, random_state=7)
    mlp2 = MLPClassifierScratch(hidden_layer_sizes=(16,), max_iter=200, random_state=7)
    mlp1.fit(X_train, y_train)
    mlp2.fit(X_train, y_train)
    assert np.array_equal(mlp1.predict(X_test), mlp2.predict(X_test))
    assert mlp1.loss_curve_ == mlp2.loss_curve_


# --- Hyperparameter validation ----------------------------------------------


def test_invalid_activation_raises(moons_data):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(hidden_layer_sizes=(16,), activation="bogus")
    with pytest.raises(ValueError, match="activation"):
        mlp.fit(X_train, y_train)


@pytest.mark.parametrize("bad", [0, -3, (0,), (16, -1), (16.5,), "16"])
def test_invalid_hidden_layer_sizes_raises(moons_data, bad):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(hidden_layer_sizes=bad)
    with pytest.raises(ValueError, match="hidden_layer_sizes"):
        mlp.fit(X_train, y_train)


def test_invalid_alpha_raises(moons_data):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(hidden_layer_sizes=(16,), alpha=-0.5)
    with pytest.raises(ValueError, match="alpha"):
        mlp.fit(X_train, y_train)


def test_invalid_max_iter_raises(moons_data):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(hidden_layer_sizes=(16,), max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        mlp.fit(X_train, y_train)


def test_invalid_tol_raises(moons_data):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(hidden_layer_sizes=(16,), tol=-1e-3)
    with pytest.raises(ValueError, match="tol"):
        mlp.fit(X_train, y_train)


def test_invalid_n_iter_no_change_raises(moons_data):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(hidden_layer_sizes=(16,), n_iter_no_change=0)
    with pytest.raises(ValueError, match="n_iter_no_change"):
        mlp.fit(X_train, y_train)


def test_invalid_learning_rate_raises(moons_data):
    X_train, _, y_train, _ = moons_data
    mlp = MLPClassifierScratch(hidden_layer_sizes=(16,), learning_rate_init=0.0)
    with pytest.raises(ValueError, match="learning_rate_init"):
        mlp.fit(X_train, y_train)
