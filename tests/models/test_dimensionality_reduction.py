"""
Behavioral tests for the from-scratch PCA (src/models/dimensionality_reduction.py).

Each test targets one observable behavior of the public API: attribute
values vs. sklearn's reference implementation, projection/reconstruction
algebra, hyperparameter validation, and determinism.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.models.dimensionality_reduction import PCAScratch


@pytest.fixture
def X_corr():
    """240x4 correlated gaussian mirroring the app's pca_correlated_4f dataset."""
    rng = np.random.default_rng(0)
    cov = [
        [1.0, 0.9, 0.7, 0.4],
        [0.9, 1.0, 0.8, 0.5],
        [0.7, 0.8, 1.0, 0.6],
        [0.4, 0.5, 0.6, 1.0],
    ]
    return rng.multivariate_normal(np.zeros(4), cov, size=240)


def _sign_aligned(a, b, axis):
    """Flip ``a``'s row/column vectors (along ``axis``) so each shot points
    the same way as ``b``: eigen directions are defined only up to sign,
    so comparisons must align signs before asserting closeness."""
    dots = np.sum(a * b, axis=axis, keepdims=True)
    return a * np.where(dots < 0, -1.0, 1.0)


# --- Fit: attributes against the sklearn reference -------------------------


def test_explained_variance_ratio_matches_sklearn(X_corr):
    scratch = PCAScratch(n_components=2).fit(X_corr)
    reference = PCA(n_components=2).fit(X_corr)
    # Ratios are ordered (largest first) on both sides, so they compare
    # element-wise without permutation or sign alignment.
    assert np.allclose(
        scratch.explained_variance_ratio_,
        reference.explained_variance_ratio_,
        atol=1e-8,
    )


def test_explained_variance_matches_sklearn(X_corr):
    scratch = PCAScratch(n_components=2).fit(X_corr)
    reference = PCA(n_components=2).fit(X_corr)
    assert np.allclose(
        scratch.explained_variance_, reference.explained_variance_, atol=1e-8
    )


def test_components_are_unit_norm_and_match_sklearn_up_to_sign(X_corr):
    scratch = PCAScratch(n_components=2).fit(X_corr)
    reference = PCA(n_components=2).fit(X_corr)

    assert np.allclose(np.linalg.norm(scratch.components_, axis=1), 1.0)
    aligned = _sign_aligned(scratch.components_, reference.components_, axis=1)
    assert np.allclose(aligned, reference.components_, atol=1e-8)


def test_transform_matches_sklearn_up_to_sign(X_corr):
    scratch = PCAScratch(n_components=2).fit(X_corr)
    reference = PCA(n_components=2).fit(X_corr)

    projected = scratch.transform(X_corr)
    assert projected.shape == (240, 2)

    aligned = _sign_aligned(projected, reference.transform(X_corr), axis=0)
    assert np.allclose(aligned, reference.transform(X_corr), atol=1e-8)


# --- Projection algebra -----------------------------------------------------


def test_full_rank_inverse_reconstructs_data(X_corr):
    scratch = PCAScratch(n_components=4).fit(X_corr)
    coordinates = scratch.fit_transform(X_corr)
    assert np.allclose(scratch.inverse_transform(coordinates), X_corr, atol=1e-8)


def test_k1_transform_and_inverse_shapes(X_corr):
    scratch = PCAScratch(n_components=1).fit(X_corr)
    coordinates = scratch.transform(X_corr)
    assert coordinates.shape == (240, 1)
    assert scratch.inverse_transform(coordinates).shape == (240, 4)


def test_fit_returns_self_for_chaining(X_corr):
    scratch = PCAScratch()
    assert scratch.fit(X_corr) is scratch


# --- Ratios are a property of the dataset, not of k -------------------------


def test_kept_ratios_never_exceed_total_variance(X_corr):
    scratch = PCAScratch(n_components=2).fit(X_corr)
    assert scratch.explained_variance_ratio_.sum() <= 1.0 + 1e-9


def test_rank_deficient_data_yields_vanished_last_component(X_corr):
    X_rank_deficient = X_corr.copy()
    X_rank_deficient[:, 3] = X_rank_deficient[:, 0]  # duplicate -> rank 3 of 4

    scratch = PCAScratch(n_components=4).fit(X_rank_deficient)
    vanished = scratch.explained_variance_ratio_[3]
    # Numerical eigenvalues can be slightly negative near zero; the
    # implementation clamps them to 0 like sklearn's covariance solver.
    assert vanished >= -1e-10
    assert vanished < 1e-8


def test_pc1_explains_most_variance_on_correlated_data(X_corr):
    scratch = PCAScratch(n_components=1).fit(X_corr)
    assert scratch.explained_variance_ratio_[0] > 0.5


# --- Defaults and determinism ------------------------------------------------


def test_default_keeps_all_components(X_corr):
    scratch = PCAScratch().fit(X_corr)
    assert scratch.n_components_ == min(len(X_corr), X_corr.shape[1])
    assert scratch.components_.shape == (4, 4)


def test_mean_is_the_per_feature_sample_mean(X_corr):
    scratch = PCAScratch(n_components=2).fit(X_corr)
    assert np.allclose(scratch.mean_, X_corr.mean(axis=0))


def test_repeated_fits_are_identical(X_corr):
    first = PCAScratch(n_components=2).fit(X_corr)
    second = PCAScratch(n_components=2).fit(X_corr)
    assert np.array_equal(first.components_, second.components_)
    assert np.array_equal(
        first.explained_variance_ratio_, second.explained_variance_ratio_
    )


# --- Validation ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 5, 1.5])
def test_invalid_n_components_raises(X_corr, bad):
    with pytest.raises(ValueError, match="n_components"):
        PCAScratch(n_components=bad).fit(X_corr)


def test_single_sample_raises(X_corr):
    with pytest.raises(ValueError, match="at least 2 samples"):
        PCAScratch().fit(X_corr[:1])
