"""
Ensemble methods implemented from scratch using NumPy.

Includes:
- RandomForestClassifierScratch: Random Forest for classification
- RandomForestRegressorScratch: Random Forest for regression

Random Forest (Breiman, 2001) = bagging + random feature subsampling:
each tree fits a bootstrap sample of the rows, and at every node only a
random subset of the features (``max_features``) is considered for the
best split. The two sources of randomness decorrelate the trees, and
averaging/voting over them reduces variance far below a single tree.

Base learners are the scratch decision trees from ``tree_models.py``,
extended here with per-node feature subsampling and per-split recording
(needed for feature importances). ``Gradient Boosting`` and other
ensemble methods remain placeholders for future work.

Simplifications vs. scikit-learn (documented on purpose):
- ``predict_proba`` returns the fraction of trees voting for each class,
  while sklearn averages per-leaf class distributions.
- Feature importances follow sklearn's mean-decrease-in-impurity
  recipe (per-tree normalized, then averaged across trees) but weight
  by node sample counts only.
- Single-threaded: no ``n_jobs`` parallelism.
"""

from typing import Optional, Union

import numpy as np

from .tree_models import DecisionTreeClassifierScratch, DecisionTreeRegressorScratch

MaxFeatures = Union[int, float, str, None]


class _RandomFeatureSplitMixin:
    """
    Adds Breiman-style per-node feature subsampling to a scratch tree.

    The base tree searches ALL features at each node. This mixin restricts
    the search to a random subset (``max_features``) drawn fresh at every
    node, and records ``(feature, gain, n_samples)`` for each accepted
    split so the forest can compute impurity-based feature importances.
    """

    def _resolve_n_features(self, n_features: int) -> int:
        """Convert ``max_features`` into an absolute feature count."""
        mf = self.max_features

        if mf is None:
            return n_features

        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(
                f"max_features must be 'sqrt', 'log2', an int, a float in "
                f"(0, 1], or None; got {mf!r}"
            )

        if isinstance(mf, float):
            if not 0 < mf <= 1:
                raise ValueError(
                    f"float max_features must be in (0, 1]; got {mf}"
                )
            return max(1, int(mf * n_features))

        if isinstance(mf, int):
            if not 1 <= mf <= n_features:
                raise ValueError(
                    f"int max_features must be in [1, n_features={n_features}]; "
                    f"got {mf}"
                )
            return mf

        raise ValueError(
            f"max_features must be 'sqrt', 'log2', an int, a float in (0, 1], "
            f"or None; got {mf!r}"
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best split among a random feature subset.

        Draws ``max_features`` columns, delegates to the base tree's
        exhaustive search on the reduced matrix, then remaps the winning
        local column index back to the global feature index so the parent
        ``_build_tree`` can split the full X correctly.
        """
        n_features = X.shape[1]
        m = self._resolve_n_features(n_features)

        features = self._rng.choice(n_features, size=m, replace=False)
        local_feature, threshold, gain = super()._best_split(X[:, features], y)

        if local_feature is None:
            return None, None, -1

        feature = int(features[local_feature])
        self._split_records.append((feature, gain, len(y)))
        return feature, threshold, gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Delegate to the base builder; reset split records at the root."""
        if depth == 0:
            self._split_records = []
        return super()._build_tree(X, y, depth)

    def impurity_decreases(self, n_features: int) -> Optional[np.ndarray]:
        """
        Per-feature total impurity decrease for this tree, normalized to
        sum to 1. Returns None if the tree never split (degenerate fit).
        """
        decreases = np.zeros(n_features)
        for feature, gain, n_node_samples in self._split_records:
            decreases[feature] += gain * n_node_samples

        total = decreases.sum()
        if total <= 0:
            return None
        return decreases / total


class _ForestClassifierTree(_RandomFeatureSplitMixin, DecisionTreeClassifierScratch):
    """Base-learner tree for the random forest classifier."""

    def __init__(
        self,
        *,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        criterion: str,
        max_features: MaxFeatures,
        rng: np.random.RandomState,
    ):
        # The base tree compares depth >= max_depth, which breaks on None;
        # an unbounded forest tree is expressed as infinity instead.
        super().__init__(
            max_depth=np.inf if max_depth is None else max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
        )
        self.max_features = max_features
        self._rng = rng
        self._split_records = []


class _ForestRegressorTree(_RandomFeatureSplitMixin, DecisionTreeRegressorScratch):
    """Base-learner tree for the random forest regressor."""

    def __init__(
        self,
        *,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: MaxFeatures,
        rng: np.random.RandomState,
    ):
        super().__init__(
            max_depth=np.inf if max_depth is None else max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.max_features = max_features
        self._rng = rng
        self._split_records = []


class RandomForestClassifierScratch:
    """
    Random Forest classifier from scratch.

    Fits ``n_estimators`` decision trees, each on a bootstrap sample of
    the training rows, with a fresh random feature subset considered at
    every node. Predictions aggregate the trees by majority vote.

    Parameters:
        n_estimators: Number of trees in the forest
        criterion: Split quality measure ('gini' or 'entropy')
        max_depth: Maximum tree depth (None = grow unbounded)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        max_features: Features considered per node ('sqrt', 'log2',
            an int, a float in (0, 1], or None for all)
        bootstrap: Whether to sample rows with replacement per tree
        oob_score: Whether to estimate held-out accuracy on out-of-bag rows
        random_state: Seed for reproducible bootstrap/feature sampling

    Attributes after fit:
        classes_: Sorted unique class labels
        estimators_: The fitted base-learner trees
        feature_importances_: Mean-decrease-in-impurity importances,
            normalized to sum to 1
        oob_score_: Out-of-bag accuracy (when oob_score=True)
        oob_decision_function_: Per-sample OOB vote fractions; rows for
            samples that were never out of bag are NaN (when oob_score=True)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierScratch":
        """
        Fit the forest.

        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,)

        Returns:
            self
        """
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy'; got {self.criterion!r}"
            )
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score requires bootstrap=True")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Map arbitrary labels to 0..C-1; the base trees use np.bincount,
        # which only accepts non-negative contiguous integers.
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_classes = len(self.classes_)

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        oob_masks = []

        for _ in range(self.n_estimators):
            tree_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            tree = _ForestClassifierTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=self.max_features,
                rng=tree_rng,
            )

            if self.bootstrap:
                indices = rng.randint(0, n_samples, size=n_samples)
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[indices] = True
                oob_mask = ~in_bag
            else:
                indices = np.arange(n_samples)
                oob_mask = np.zeros(n_samples, dtype=bool)

            tree.fit(X[indices], y_idx[indices])
            self.estimators_.append(tree)
            oob_masks.append(oob_mask)

        self.feature_importances_ = self._compute_importances(n_features)

        if self.oob_score:
            self._compute_oob(X, y_idx, oob_masks, n_classes)

        return self

    def _aggregate_votes(self, X: np.ndarray) -> np.ndarray:
        """Tally one vote per tree per sample; shape (n_samples, n_classes)."""
        votes = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            votes[np.arange(len(X)), tree.predict(X)] += 1
        return votes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by majority vote across trees.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        return self.classes_[self._aggregate_votes(X).argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities as vote fractions across trees.

        Note: returns the fraction of trees voting for each class, whereas
        scikit-learn averages the per-leaf class distributions.
        """
        X = np.asarray(X, dtype=float)
        return self._aggregate_votes(X) / self.n_estimators

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _compute_importances(self, n_features: int) -> np.ndarray:
        """Average per-tree normalized impurity decreases across the forest."""
        per_tree = [
            tree.impurity_decreases(n_features) for tree in self.estimators_
        ]
        usable = [imp for imp in per_tree if imp is not None]

        if not usable:
            return np.full(n_features, 1.0 / n_features)

        importances = np.mean(usable, axis=0)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        return importances

    def _compute_oob(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        oob_masks: list,
        n_classes: int,
    ) -> None:
        """Estimate held-out accuracy from out-of-bag rows."""
        votes = np.zeros((len(X), n_classes))
        for tree, oob_mask in zip(self.estimators_, oob_masks):
            oob_indices = np.where(oob_mask)[0]
            if len(oob_indices) == 0:
                continue
            votes[oob_indices, tree.predict(X[oob_indices])] += 1

        covered = votes.sum(axis=1) > 0
        oob_pred = votes.argmax(axis=1)

        self.oob_decision_function_ = np.divide(
            votes,
            votes.sum(axis=1, keepdims=True),
            out=np.full_like(votes, np.nan),
            where=covered[:, None],
        )
        self.oob_score_ = float(np.mean(oob_pred[covered] == y_idx[covered]))


class RandomForestRegressorScratch:
    """
    Random Forest regressor from scratch.

    Same construction as the classifier, but trees fit on squared-error
    splits and predictions aggregate by averaging.

    Parameters:
        n_estimators: Number of trees in the forest
        max_depth: Maximum tree depth (None = grow unbounded)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        max_features: Features considered per node ('sqrt', 'log2',
            an int, a float in (0, 1], or None for all; the sklearn
            regressor default of 1.0 means all features)
        bootstrap: Whether to sample rows with replacement per tree
        oob_score: Whether to estimate held-out R² on out-of-bag rows
        random_state: Seed for reproducible bootstrap/feature sampling

    Attributes after fit:
        estimators_: The fitted base-learner trees
        feature_importances_: Mean-decrease-in-impurity importances,
            normalized to sum to 1
        oob_score_: Out-of-bag R² (when oob_score=True)
        oob_prediction_: Out-of-bag predictions; entries for samples that
            were never out of bag are NaN (when oob_score=True)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        """
        Fit the forest.

        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Regression targets of shape (n_samples,)

        Returns:
            self
        """
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score requires bootstrap=True")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        oob_masks = []

        for _ in range(self.n_estimators):
            tree_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            tree = _ForestRegressorTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=tree_rng,
            )

            if self.bootstrap:
                indices = rng.randint(0, n_samples, size=n_samples)
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[indices] = True
                oob_mask = ~in_bag
            else:
                indices = np.arange(n_samples)
                oob_mask = np.zeros(n_samples, dtype=bool)

            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)
            oob_masks.append(oob_mask)

        self.feature_importances_ = self._compute_importances(n_features)

        if self.oob_score:
            self._compute_oob(X, y, oob_masks)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values as the mean across trees.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        predictions = np.column_stack(
            [tree.predict(X) for tree in self.estimators_]
        )
        return predictions.mean(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score on the given data."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)

    def _compute_importances(self, n_features: int) -> np.ndarray:
        """Average per-tree normalized impurity decreases across the forest."""
        per_tree = [
            tree.impurity_decreases(n_features) for tree in self.estimators_
        ]
        usable = [imp for imp in per_tree if imp is not None]

        if not usable:
            return np.full(n_features, 1.0 / n_features)

        importances = np.mean(usable, axis=0)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        return importances

    def _compute_oob(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oob_masks: list,
    ) -> None:
        """Estimate held-out R² from out-of-bag rows."""
        oob_sum = np.zeros(len(X))
        oob_count = np.zeros(len(X))
        for tree, oob_mask in zip(self.estimators_, oob_masks):
            oob_indices = np.where(oob_mask)[0]
            if len(oob_indices) == 0:
                continue
            oob_sum[oob_indices] += tree.predict(X[oob_indices])
            oob_count[oob_indices] += 1

        covered = oob_count > 0
        self.oob_prediction_ = np.divide(
            oob_sum,
            oob_count,
            out=np.full(len(X), np.nan),
            where=covered,
        )

        y_pred = self.oob_prediction_[covered]
        ss_res = np.sum((y[covered] - y_pred) ** 2)
        ss_tot = np.sum((y[covered] - np.mean(y[covered])) ** 2)
        self.oob_score_ = float(1 - ss_res / ss_tot)
