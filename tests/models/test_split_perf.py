"""Interactive-page performance budgets for the heavy scratch engines.

The Streamlit playground fragment re-fits both engines on every slider move,
so a default-param scratch fit must stay well under a couple of seconds.
These tests pin that property against future regressions.

Watched RED pre-fix on this machine:
  RF  scratch default fit (moons):     7.549 s  >  5 s budget
  GB  scratch default fit (moons):    28.177 s  >  5 s budget
Green after the split-search vectorization (tree split parity is pinned in
test_split_parity.py — speed must not change picked splits).
"""

import time

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

from src.models.ensemble_models import (
    GradientBoostingClassifierScratch,
    RandomForestClassifierScratch,
)

BUDGET_S = 5.0


def _moons():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    return X, y


def test_random_forest_scratch_default_fit_within_budget():
    X, y = _moons()
    rf = RandomForestClassifierScratch(
        n_estimators=100, max_depth=5, random_state=0
    )
    t0 = time.perf_counter()
    rf.fit(X, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < BUDGET_S, f"RF scratch default fit took {elapsed:.2f}s"


def test_gradient_boosting_scratch_default_fit_within_budget():
    X, y = _moons()
    gb = GradientBoostingClassifierScratch(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
    )
    t0 = time.perf_counter()
    gb.fit(X, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < BUDGET_S, f"GB scratch default fit took {elapsed:.2f}s"


def test_forest_regressor_scratch_fit_within_budget():
    # regression twins sit behind the RF card's variant select — guaranteed by
    # the same vectorized base tree
    X, _ = make_blobs(n_samples=240, centers=3, cluster_std=2.2, random_state=0)
    rng = np.random.default_rng(0)
    y = rng.normal(X[:, 0] * 2 + X[:, 1], 0.3)
    from src.models.ensemble_models import RandomForestRegressorScratch

    reg = RandomForestRegressorScratch(n_estimators=100, max_depth=5, random_state=0)
    t0 = time.perf_counter()
    reg.fit(X, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < BUDGET_S, f"RF-regressor scratch fit took {elapsed:.2f}s"
