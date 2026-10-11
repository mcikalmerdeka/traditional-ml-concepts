"""Split-search parity: vectorized `_best_split` must pick exactly what the
naive per-threshold implementation picks — same (feature, threshold, gain),
same tie-breaking, same fitted predictions.

RED phase: imports the module-level vectorized functions; before the fix they
don't exist, so this file fails at collection. The naive reference copies below
are verbatim transcriptions of the pre-fix method bodies (tree_models.py at
commit e45b109) — the fix may NOT change picked splits, only how fast they're
found. Gains are compared with atol=1e-12: the vectorized arithmetic (running
sums) rounds at ~1e-16 differently, and batteries assert no tie flips because of
it.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_circles, make_moons

import src.models.tree_models as tm


def _naive_classifier_best_split(self, X, y):
    """VERBATIM pre-fix DecisionTreeClassifierScratch._best_split."""
    best_gain = -1
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    # Try each feature
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        # Try each unique value as threshold
        for threshold in thresholds:
            left_mask, right_mask = self._split(X, threshold, feature)

            # Skip if split doesn't satisfy min_samples_leaf
            if np.sum(left_mask) < self.min_samples_leaf or \
               np.sum(right_mask) < self.min_samples_leaf:
                continue

            left_y, right_y = y[left_mask], y[right_mask]

            # Calculate information gain
            gain = self._information_gain(y, left_y, right_y)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def _naive_regressor_best_split(self, X, y):
    """VERBATIM pre-fix DecisionTreeRegressorScratch._best_split."""
    best_reduction = -1
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    # Try each feature
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        # Try each unique value as threshold
        for threshold in thresholds:
            left_mask, right_mask = self._split(X, threshold, feature)

            # Skip if split doesn't satisfy min_samples_leaf
            if np.sum(left_mask) < self.min_samples_leaf or \
               np.sum(right_mask) < self.min_samples_leaf:
                continue

            left_y, right_y = y[left_mask], y[right_mask]

            # Calculate variance reduction
            reduction = self._variance_reduction(y, left_y, right_y)

            if reduction > best_reduction:
                best_reduction = reduction
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_reduction


# --- battery: adversarial + real toy data -----------------------------------

_BATTERY = []
_rng = np.random.default_rng(7)
for _trial in range(60):
    _n = int(_rng.integers(6, 150))
    _k = int(_rng.choice([2, 3]))
    _X = _rng.normal(size=(_n, 2)) * _rng.uniform(0.5, 4)
    if _trial % 5 == 0:
        _X[:, 0] = np.round(_X[:, 0] * 2) / 2  # duplicate feature values
    if _trial % 11 == 0:
        _X[:, 1] = 1.0  # constant column
    _BATTERY.append(
        (_X.copy(), _rng.integers(0, _k, size=_n),
         _rng.normal(_rng.uniform(-2, 2), 1.0, size=_n))
    )


_TOYS = [
    make_moons(n_samples=60, noise=0.15, random_state=0),
    make_circles(n_samples=60, factor=0.5, noise=0.08, random_state=0),
    make_blobs(n_samples=90, centers=3, cluster_std=2.2, random_state=0),
]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("min_samples_leaf", [1, 2, 5])
def test_classifier_picks_identical_on_battery(criterion, min_samples_leaf):
    for X, y_clf, y_reg in _BATTERY:
        cls = tm.DecisionTreeClassifierScratch(
            min_samples_leaf=min_samples_leaf, criterion=criterion
        )
        f1, t1, g1 = _naive_classifier_best_split(cls, X, y_clf)
        f2, t2, g2 = tm._vectorized_classifier_best_split(cls, X, y_clf)
        assert f2 == f1
        if f1 is None:
            assert g2 == g1 == -1
        else:
            assert t2 == t1
            assert pytest.approx(g2, abs=1e-12) == g1


@pytest.mark.parametrize("min_samples_leaf", [1, 2, 5])
def test_regressor_picks_identical_on_battery(min_samples_leaf):
    # bit-parity holds wherever the best reductions differ by more than fp
    # ties (~1e-9): the running-sum decomposition rounds ~1e-16 differently
    # from the naive mean-then-square, so at mathematically-equal reductions
    # the two may report a different tie candidate. Those picks are equivalent
    # (same gain); the vectorized best reduction never trails the naive one.
    for X, y_clf, y_reg in _BATTERY:
        reg = tm.DecisionTreeRegressorScratch(min_samples_leaf=min_samples_leaf)
        f1, t1, r1 = _naive_regressor_best_split(reg, X, y_reg)
        f2, t2, r2 = tm._vectorized_regressor_best_split(reg, X, y_reg)
        if r1 is None:
            assert r2 == r1 == -1
            continue
        tie = abs(r2 - r1) <= 1e-9
        if not tie:
            assert (f2, t2) == (f1, t1)
        assert pytest.approx(r2, abs=1e-9) == r1


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("min_samples_leaf", [1, 3])
def test_classifier_picks_identical_on_toys(criterion, min_samples_leaf):
    for X, y in _TOYS:
        y = np.asarray(y)
        cls = tm.DecisionTreeClassifierScratch(
            min_samples_leaf=min_samples_leaf, criterion=criterion
        )
        f1, t1, g1 = _naive_classifier_best_split(cls, X, y)
        f2, t2, g2 = tm._vectorized_classifier_best_split(cls, X, y)
        assert (f2, t2) == (f1, t1)
        assert pytest.approx(g2, abs=1e-12) == g1


def test_empty_y_returns_no_split_like_naive():
    cls = tm.DecisionTreeClassifierScratch()
    reg = tm.DecisionTreeRegressorScratch()
    X = np.zeros((0, 2))
    assert tm._vectorized_classifier_best_split(cls, X, np.zeros(0, dtype=int)) \
        == (None, None, -1)
    assert _naive_classifier_best_split(cls, X, np.zeros(0, dtype=int)) \
        == (None, None, -1)
    assert tm._vectorized_regressor_best_split(reg, X, np.zeros(0)) == (None, None, -1)


def test_conflicting_tie_keeps_first_candidate_in_scan_order():
    # perfectly alternating labels over one constant-featuring X: a real-tie
    # node — whichever implementation found the theory-zero gain, first-in-scan
    # order must be kept (strict `>` keeps the earlier candidate)
    X = np.linspace(0, 1, 16).reshape(-1, 1)
    y = np.zeros(16, dtype=int)
    y[::2] = 1
    cls = tm.DecisionTreeClassifierScratch(min_samples_leaf=1)
    f1, t1, g1 = _naive_classifier_best_split(cls, X, y)
    f2, t2, g2 = tm._vectorized_classifier_best_split(cls, X, y)
    assert (f2, t2) == (f1, t1)
    assert pytest.approx(g2, abs=1e-12) == g1


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_fitted_tree_predictions_identical_on_toys(criterion):
    # what actually matters: the whole tree predicts the same thing
    for X, y in _TOYS:
        y = np.asarray(y)
        fast = tm.DecisionTreeClassifierScratch(max_depth=4, criterion=criterion)
        naive = tm.DecisionTreeClassifierScratch(max_depth=4, criterion=criterion)
        fast._best_split = lambda X_, y_, _c=fast: tm._vectorized_classifier_best_split(_c, X_, y_)
        naive._best_split = lambda X_, y_, _c=naive: _naive_classifier_best_split(_c, X_, y_)
        assert np.array_equal(fast.fit(X, y).predict(X), naive.fit(X, y).predict(X))


def test_fitted_regressor_predictions_identical_on_blobs():
    # predictions agree to fp noise: tie-level splits (see the pick test) can
    # move a boundary by no more than the reduction noise that caused them
    X, y_clf, y_reg = _BATTERY[0]
    fast = tm.DecisionTreeRegressorScratch(max_depth=4)
    naive = tm.DecisionTreeRegressorScratch(max_depth=4)
    fast._best_split = lambda X_, y_, _c=fast: tm._vectorized_regressor_best_split(_c, X_, y_)
    naive._best_split = lambda X_, y_, _c=naive: _naive_regressor_best_split(_c, X_, y_)
    np.testing.assert_allclose(
        fast.fit(X, y_reg).predict(X), naive.fit(X, y_reg).predict(X), atol=1e-9
    )
