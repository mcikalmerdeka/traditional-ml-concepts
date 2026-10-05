"""Curated toy datasets for the playgrounds.

Deterministic (fixed seeds) and tiny (≤ 300 rows, ≤ 4 features) so slider
moves re-fit fast. Hand-crafted regression sets use known coefficients so
"scratch vs sklearn" coefficient comparisons are meaningful; classification
and clustering sets come from sklearn's make_* family with pinned seeds.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

from app.core.card import Data

_REGRESSION_NOTE_CLEAN = "clean linear data — both engines should agree"
_REGRESSION_NOTE_NOISY = "noisy linear data — noise floor limits R² for both engines"
_REGRESSION_NOTE_OUTLIERS = "heavy outliers — watch ridge/lasso shrug them off"


def _linear(rng: np.random.Generator, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(-3, 3, size=(120, 1))
    y = 3 * X[:, 0] + 2 + rng.normal(0, sigma, 120)
    return X, y


def _lin_clean_1f() -> Data:
    X, y = _linear(np.random.default_rng(0), 0.5)
    return Data(X=X, y=y, note=_REGRESSION_NOTE_CLEAN, family="regression")


def _lin_noisy_1f() -> Data:
    X, y = _linear(np.random.default_rng(1), 3.0)
    return Data(X=X, y=y, note=_REGRESSION_NOTE_NOISY, family="regression")


def _lin_outliers_1f() -> Data:
    rng = np.random.default_rng(2)
    X, y = _linear(rng, 0.5)
    idx = rng.choice(120, 10, replace=False)
    y[idx] += rng.uniform(20, 40, 10)
    return Data(X=X, y=y, note=_REGRESSION_NOTE_OUTLIERS, family="regression")


def _moons() -> Data:
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    return Data(X=X, y=y, note="two moons — non-linear boundary needed", family="classification")


def _circles() -> Data:
    X, y = make_circles(n_samples=240, factor=0.5, noise=0.08, random_state=0)
    return Data(X=X, y=y, note="concentric circles — radial structure", family="classification")


def _lin_separable() -> Data:
    X, y = make_blobs(n_samples=240, centers=[(-2, -2), (2, 2)], cluster_std=0.8, random_state=0)
    return Data(X=X, y=y, note="linearly separable blobs — easy case", family="classification")


def _kmeans_4blobs() -> Data:
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=0)
    return Data(X=X, y=None, note="four blobs — K-Means happy case", family="clustering")


def _kmeans_rings() -> Data:
    rng = np.random.default_rng(3)
    n = 150
    theta_outer = rng.uniform(0, 2 * np.pi, n)
    theta_inner = rng.uniform(0, 2 * np.pi, n)
    outer = np.c_[5 * np.cos(theta_outer), 5 * np.sin(theta_outer)] + rng.normal(0, 0.3, (n, 2))
    inner = np.c_[2 * np.cos(theta_inner), 2 * np.sin(theta_inner)] + rng.normal(0, 0.3, (n, 2))
    X = np.vstack([outer, inner])
    return Data(X=X, y=None, note="concentric rings — where K-Means fails", family="clustering")


_REGISTRY = {
    "lin_clean_1f": _lin_clean_1f,
    "lin_noisy_1f": _lin_noisy_1f,
    "lin_outliers_1f": _lin_outliers_1f,
    "moons": _moons,
    "circles": _circles,
    "lin_separable": _lin_separable,
    "kmeans_4blobs": _kmeans_4blobs,
    "kmeans_rings": _kmeans_rings,
}

_DATASETS: dict[str, Data] = {k: f() for k, f in _REGISTRY.items()}


def get_dataset(dataset_id: str) -> Data:
    return _DATASETS[dataset_id]


def dataset_ids() -> tuple[str, ...]:
    return tuple(_DATASETS)
