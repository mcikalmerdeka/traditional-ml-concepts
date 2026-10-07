"""
Naive Bayes classifiers implemented from scratch using NumPy.

Includes:
- GaussianNaiveBayesScratch: Gaussian Naive Bayes for classification

''Naive'' Bayes assumes features are conditionally independent given the
class, so the joint probability factorises into a prior times one
per-feature term:

    P(y | x) ∝ P(y) · Π_j P(x_j | y)

For the Gaussian variant each feature is modelled per class as a normal
N(theta[y, j], var[y, j]), which turns that product into closed-form
means/variances during fit and a simple Gaussian log-density sum during
predict. All prediction math stays in log space because multiplying
many small densities would underflow in linear space, and log-space
subtraction of the row maximum (logsumexp semantics) makes the final
normalisation numerically safe.

Only the Gaussian variant is implemented. Multinomial/Bernoulli Naive
Bayes target count/binary feature data and are out of scope for this
study repo.

Simplifications vs. scikit-learn (documented on purpose):
- predict/predict_proba work from a single matrix of log posteriors,
  while sklearn accumulates class-by-class via a private
  _joint_log_likelihood — same numbers, simpler plumbing.
- No partial_fit / incremental updates; fit is a single closed form
  over the whole batch.
- No sample_weight support.
"""

from typing import Union

import numpy as np


class GaussianNaiveBayesScratch:
    """
    Gaussian Naive Bayes classifier from scratch.

    Fit estimates a prior, a mean, and a population variance per class
    per feature, then predicts by comparing log posteriors: the log
    prior plus one Gaussian log-density term per feature.

    Parameters:
        var_smoothing: Additive share of the largest feature variance,
            added to every per-class variance for numerical stability
            (sklearn-compatible; the absolute amount added is
            ``var_smoothing * X.var(axis=0).max()``)

    Attributes after fit:
        classes_: Sorted unique class labels
        class_prior_: Class frequencies, shape (n_classes,)
        theta_: Per-class per-feature means, shape (n_classes, n_features)
        var_: Per-class per-feature population variances plus the
            smoothing epsilon, shape (n_classes, n_features)
        epsilon_: Absolute smoothing term added to each variance
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesScratch":
        """
        Estimate class priors, per-class means, and smoothed variances.

        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,)

        Returns:
            self
        """
        # NaN would silently pass a ``<= 0`` check, so invert the test
        if not self.var_smoothing > 0:
            raise ValueError(
                f"var_smoothing must be > 0 to stabilise variance "
                f"estimates; got {self.var_smoothing!r}"
            )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Map arbitrary labels to 0..C-1; never assume labels are 0/1.
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_classes = len(self.classes_)

        self.class_prior_ = np.bincount(y_idx, minlength=n_classes) / len(y)

        n_features = X.shape[1]
        self.theta_ = np.zeros((n_classes, n_features))
        raw_var = np.zeros((n_classes, n_features))
        for c in range(n_classes):
            X_c = X[y_idx == c]
            self.theta_[c] = X_c.mean(axis=0)
            # Population variance (ddof=0) to match sklearn's GaussianNB
            raw_var[c] = X_c.var(axis=0)

        # sklearn's smoothing formula: share of the largest feature
        # variance, so scale is relative to the data and never zero
        # unless the data itself is constant.
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        # var_ is stored smoothed: that is what predict evaluates
        self.var_ = raw_var + self.epsilon_

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Log prior + summed Gaussian log-densities; shape (n, n_classes)."""
        X = np.asarray(X, dtype=float)
        n_samples = len(X)

        # Looping over the (few) classes keeps the read-friendly
        # one-class-per-row formula without a bulky 3-D broadcast.
        jll = np.empty((n_samples, len(self.classes_)))
        for c in range(len(self.classes_)):
            # log N(x_j | theta, var) = -0.5*log(2*pi*var)
            #                          - (x_j - theta)^2 / (2*var)
            log_pdf = (
                -0.5 * np.log(2.0 * np.pi * self.var_[c])
                - (X - self.theta_[c]) ** 2 / (2.0 * self.var_[c])
            )
            jll[:, c] = np.log(self.class_prior_[c]) + log_pdf.sum(axis=1)
        return jll

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior class probabilities.

        Computed in log space and normalised with logsumexp semantics:
        subtract the row max before exponentiating so that even very
        negative log likelihoods cannot overflow or underflow.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Probabilities of shape (n_samples, n_classes), rows sum to 1
        """
        jll = self._joint_log_likelihood(X)
        log_norm = jll.max(axis=1, keepdims=True)
        posteriors = np.exp(jll - log_norm)
        return posteriors / posteriors.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class with the highest log posterior.

        Ties broken toward the lower class, as in sklearn.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[jll.argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Mean accuracy on the given data.

        Parameters:
            X: Features of shape (n_samples, n_features)
            y: True class labels of shape (n_samples,)

        Returns:
            Fraction of correctly predicted labels
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))
