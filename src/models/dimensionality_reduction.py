"""
Dimensionality reduction implemented from scratch using NumPy.

Includes:
- PCAScratch: Principal Component Analysis

PCA (Pearson, 1901; Hotelling, 1933) compresses correlated features onto
the directions along which the data actually spreads. After centering the
rows, those directions are the eigenvectors of the sample covariance
matrix and the variance along each one is the matching eigenvalue
(Sigma v = lambda v). Keeping the k leading eigenvectors turns each
n_features-long row into k coordinates while preserving most of the
spread. The eigendecomposition is exact and deterministic — no iterative
solver and no random_state.

t-SNE-style nonlinear methods are out of scope of this study repo.

Simplifications vs. scikit-learn (documented on purpose):
- One code path: eigendecomposition of the ddof=1 sample covariance.
  scikit-learn decomposes the centered data with SVD instead; the
  resulting components/estimated variance agree to float precision for
  n_samples >= 2, but the two are not bit-identical algorithms.
- Eigenvalues can come out slightly negative for numerically
  rank-deficient data; they are clamped to 0 like sklearn's
  covariance-eigendecomposition solver (its SVD path cannot produce
  negatives by construction).
- The component sign convention is fixed per row (largest-absolute
  coefficient positive) so scratch and sklearn loadings are comparable,
  but genuine ties on that coefficient leave the sign data-dependent —
  same as sklearn's svd_flip.
- No whiten, svd_solver, tol, or randomized-solver knobs: the exact
  decomposition needs none of them.
- Minimal validation surface: dtypes are cast to float64 and shape /
  n_components are checked in fit only; transform and inverse_transform
  let NumPy raise on feature-count mismatches.
"""

from typing import Optional

import numpy as np


class PCAScratch:
    """
    Principal Component Analysis from scratch.

    Fits the covariance eigendecomposition once and in full; the k kept
    components are only sliced out afterwards. ``transform`` centers rows
    and projects them onto the kept axes, ``inverse_transform`` maps
    coordinates back to feature space.

    Parameters:
        n_components: Number of components to keep. None keeps
            min(n_samples, n_features). Validated at fit time.

    Attributes after fit:
        components_: Principal axes of shape (n_components, n_features),
            unit-norm rows, sign-adjusted so each row's
            largest-absolute coefficient is positive
        explained_variance_: Variance per kept component,
            shape (n_components,)
        explained_variance_ratio_: explained_variance_ as a share of the
            total variance across ALL components (the covariance trace)
        mean_: Per-feature mean of the fitted X, shape (n_features,)
        n_components_: Resolved number of kept components (int)
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCAScratch":
        """
        Fit PCA: center the rows, eigendecompose the sample covariance,
        keep the leading eigenpairs.

        Parameters:
            X: Data of shape (n_samples, n_features); needs at least
                2 samples for the ddof=1 covariance estimate.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError(
                f"PCA needs at least 2 samples to estimate the covariance "
                f"with ddof=1; got X with shape {X.shape}"
            )
        self.n_components_ = self._resolve_n_components(n_samples, n_features)

        X_centered = X - X.mean(axis=0)
        covariance = np.cov(X_centered, rowvar=False)

        # ddof=1 (np.cov's default) matches sklearn's SVD estimator for
        # n >= 2 samples: its singular values satisfy S^2 / (n - 1).
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # eigh returns ascending order — reverse to descending. Clamp
        # negatives to 0 before use: numerically rank-deficient data
        # yields ~1e-16 eigenvalues of either sign, and sklearn's
        # covariance solver clamps the same way.
        eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        kept = self.n_components_
        self.explained_variance_ = eigenvalues[:kept].copy()
        self.components_ = eigenvectors[:, :kept].T.copy()

        # sklearn's svd_flip-style convention: sign each axis so its
        # largest-absolute coefficient is positive, making scratch and
        # sklearn loadings directly comparable.
        for i in range(kept):
            lead = np.argmax(np.abs(self.components_[i]))
            if self.components_[i, lead] < 0:
                self.components_[i] = -self.components_[i]

        # Ratios divide by the total variance across ALL components (the
        # covariance trace), not just the kept ones: ratios are dataset
        # properties, so the kept numbers do not depend on k.
        total_variance = float(np.trace(covariance))
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )

        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project rows onto the kept components.

        Parameters:
            X: Data of shape (n_samples, n_features)

        Returns:
            Coordinates of shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on X and return the projection of the same rows.

        Parameters:
            X: Data of shape (n_samples, n_features)

        Returns:
            Coordinates of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map kept-component coordinates back to feature space.

        Parameters:
            X: Coordinates of shape (n_samples, n_components)

        Returns:
            Reconstruction of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        return X @ self.components_ + self.mean_

    def _resolve_n_components(self, n_samples: int, n_features: int) -> int:
        """Validate ``n_components`` against X's shape; None means full rank."""
        limit = min(n_samples, n_features)
        requested = self.n_components

        if requested is None:
            return limit

        if not isinstance(requested, (int, np.integer)):
            raise ValueError(
                f"n_components must be None or an int; got {requested!r}"
            )
        if not 1 <= requested <= limit:
            raise ValueError(
                f"n_components={requested} is out of range: must be in "
                f"[1, min(n_samples={n_samples}, n_features={n_features})]"
                f"={limit}"
            )
        return int(requested)
