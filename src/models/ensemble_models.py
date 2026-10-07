"""
Ensemble methods implemented from scratch using NumPy.

Includes:
- RandomForestClassifierScratch: Random Forest for classification
- RandomForestRegressorScratch: Random Forest for regression
- GradientBoostingClassifierScratch: Gradient Boosting for classification
- VotingEnsembleClassifierScratch: hard/soft voting over heterogeneous members

Random Forest (Breiman, 2001) = bagging + random feature subsampling:
each tree fits a bootstrap sample of the rows, and at every node only a
random subset of the features (``max_features``) is considered for the
best split. The two sources of randomness decorrelate the trees, and
averaging/voting over them reduces variance far below a single tree.

Gradient Boosting (Friedman, 2001) is the variance-reducing method's
opposite: a stagewise additive model F_m = F_{m-1} + nu * h_m where each
tree fits the negative gradient of the loss (residuals for squared error,
y - p for log-loss) and nu (``learning_rate``) shrinks every step.

Voting merges independently trained members of different bias families by
label majority (hard) or averaged probabilities (soft).

Base learners are the scratch decision trees from ``tree_models.py``,
extended there with per-leaf ``proba`` storage and an ``apply`` accessor;
the forest adds per-node feature subsampling via ``_RandomFeatureSplitMixin``.

Simplifications vs. scikit-learn (documented on purpose):
- ``predict_proba`` returns the fraction of trees voting for each class,
  while sklearn averages per-leaf class distributions.
- Feature importances follow sklearn's mean-decrease-in-impurity
  recipe (per-tree normalized, then averaged across trees) but weight
  by node sample counts only.
- Single-threaded: no ``n_jobs`` parallelism.
- Gradient boosting refits each stage's trees on gradient residuals and
  (like sklearn) then replaces leaf values with the Newton step per leaf,
  but skips Friedman's line search, subsample<1.0, and histogram trees.
"""

from copy import deepcopy
from typing import Any, Optional, Union

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


# --- Gradient Boosting (classifier) ------------------------------------------


class GradientBoostingClassifierScratch:
    """
    Gradient Boosting classifier from scratch.

    A stagewise additive model: start from an initial guess F0 (log-odds of
    the base rate for binary, log class priors for multiclass) and grow

        F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)

    where each regression tree h_m fits the negative gradient of the loss —
    the residuals y - p for log-loss. As in scikit-learn, after a tree is
    fitted on the raw gradient its leaf values are replaced with the Newton
    step per leaf (sum of gradients / sum of second derivatives), which uses
    the loss's curvature to size each leaf's update.

    Binary problems stage ONE stump-tree per iteration on the log-odds;
    multiclass problems stage K trees per iteration (softmax deviance), one
    per class.

    Parameters:
        n_estimators: Number of boosting stages
        learning_rate: Shrinkage applied to every tree's contribution (nu)
        max_depth: Maximum depth of each stage's regression tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        random_state: Accepted for API parity; unused — no stochastic
            subsampling is implemented (sklearn's ``subsample`` is always 1.0)

    Attributes after fit:
        classes_: Sorted unique class labels
        estimators_: List over stages; stage m holds the K regression trees
            fitted at that stage (1 tree for binary, one per class otherwise)
        loss_curve_: Not tracked (sklearn's per-stage train deviance is
            out of scope here)
    """

    _LOGIT_CLIP = 30.0  # sigmoid/softmax inputs beyond +-30 are numerically saturated

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _check_params(self) -> None:
        if not (isinstance(self.n_estimators, int) and self.n_estimators >= 1):
            raise ValueError(
                f"n_estimators must be a positive int; got {self.n_estimators!r}"
            )
        if not (self.learning_rate > 0):
            raise ValueError(
                f"learning_rate must be > 0; got {self.learning_rate!r}"
            )
        if not (isinstance(self.max_depth, int) and self.max_depth >= 1):
            raise ValueError(f"max_depth must be a positive int; got {self.max_depth!r}")

    def _base_tree(self) -> DecisionTreeRegressorScratch:
        """The regression tree fitted to a stage's gradient residuals."""
        return DecisionTreeRegressorScratch(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

    @classmethod
    def _newton_leaf_values(
        cls,
        tree: DecisionTreeRegressorScratch,
        X: np.ndarray,
        grad: np.ndarray,
        second: np.ndarray,
    ) -> None:
        """
        Overwrite the fitted tree's leaf values with the loss's Newton step.

        A tree fitted by least squares on the gradient has leaf means; the
        log/deviance losses instead want each leaf's Taylor step
        = sum(grad) / sum(second). Grouping samples by leaf needs the tree's
        ``apply`` accessor (leaf Node references).
        """
        leaves = tree.apply(X)
        buckets: dict = {}
        for i, leaf in enumerate(leaves):
            buckets.setdefault(id(leaf), []).append(i)
        for indices in buckets.values():
            idx = np.asarray(indices)
            numer = float(grad[idx].sum())
            denom = float(second[idx].sum())
            # pure leaves carry no gradient mass; the floor only prevents 0/0
            leaves[idx[0]].value = float(numer / max(denom, 1e-12))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifierScratch":
        """
        Fit the stagewise additive model.

        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,)

        Returns:
            self
        """
        self._check_params()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = X.shape[0]

        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_classes = len(self.classes_)

        # A single training class: the trivial constant classifier (sklearn
        # would raise here; the teaching surfaces prefer a working model).
        if n_classes == 1:
            self.estimators_ = []
            self._single_class = True
            return self
        self._single_class = False

        self.estimators_ = []

        if n_classes == 2:
            # F0: log-odds of the positive class's base rate
            pos_rate = float(np.clip((y_idx == 1).mean(), 1e-15, 1 - 1e-15))
            self._f0_logit = float(np.log(pos_rate / (1 - pos_rate)))
            F = np.full(n_samples, self._f0_logit)

            for _ in range(self.n_estimators):
                p = self._sigmoid(F)
                grad = y_idx.astype(float) - p       # negative log-loss gradient
                second = p * (1.0 - p)               # its curvature

                tree = self._base_tree().fit(X, grad)
                self._newton_leaf_values(tree, X, grad, second)
                F += self.learning_rate * tree.predict(X)

                self.estimators_.append([tree])
        else:
            # F0: log class priors (a zero-mass prior floors at a safe eps;
            # prior sums stay 1 — only absent classes are floored)
            prior = np.bincount(y_idx, minlength=n_classes) / n_samples
            self._class_prior = prior
            F = np.tile(np.log(np.maximum(prior, 1e-15)), (n_samples, 1))

            onehot = np.eye(n_classes)[y_idx]
            for _ in range(self.n_estimators):
                P = self._softmax(np.clip(F, -self._LOGIT_CLIP, self._LOGIT_CLIP))
                stage = []
                for k in range(n_classes):
                    grad = onehot[:, k] - P[:, k]    # negative gradient, class k
                    second = P[:, k] * (1.0 - P[:, k])

                    tree = self._base_tree().fit(X, grad)
                    self._newton_leaf_values(tree, X, grad, second)
                    F[:, k] += self.learning_rate * tree.predict(X)

                    stage.append(tree)
                self.estimators_.append(stage)

        return self

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        """Logistic function, clipped input keeps exp() in float64 range."""
        clipped = np.clip(logits, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Row-wise softmax, max-shifted for numerical stability."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    def _decision(self, X: np.ndarray) -> np.ndarray:
        """
        Replay the staged model over raw features -> the score F.

        Binary problems get a 1-D log-odds vector; multiclass problems get
        the (n_samples, n_classes) matrix whose softmax IS the model.
        """
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        if self._single_class:
            return np.zeros(n_samples)

        n_classes = len(self.classes_)
        if n_classes == 2:
            F = np.full(n_samples, self._f0_logit)
            for stage in self.estimators_:
                for tree in stage:
                    F += self.learning_rate * tree.predict(X)
            return F

        F = np.tile(
            np.log(np.maximum(self._class_prior, 1e-15)), (n_samples, 1)
        )
        for stage in self.estimators_:
            for k, tree in enumerate(stage):
                F[:, k] += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities from the staged model.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Probabilities of shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=float)

        if self._single_class:
            return np.ones((len(X), 1))

        F = self._decision(X)
        if len(self.classes_) == 2:
            p1 = self._sigmoid(F)
            return np.column_stack([1.0 - p1, p1])
        return self._softmax(np.clip(F, -self._LOGIT_CLIP, self._LOGIT_CLIP))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by the staged model's argmax.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


# --- Voting Ensemble (classifier) --------------------------------------------


class VotingEnsembleClassifierScratch:
    """
    Hard/soft voting classifier over heterogeneous SciKit-style members.

    Trains a panel of models (each given as a ``(name, estimator)`` template
    tuple, like sklearn's ``VotingClassifier``) on fresh copies and merges
    their verdicts:

    - hard voting: majority of the members' predicted labels
    - soft voting: argmax of the AVERAGED class probabilities

    Members must implement ``fit`` and ``predict``; soft voting additionally
    requires ``predict_proba`` returning columns in the member's sorted class
    order (an observable ``classes_`` attribute is used to remap columns when
    a member declares it — e.g. a basket of scratch and sklearn models).

    Parameters:
        estimators: List of (name, estimator-template) tuples; templates are
            deep-copied at fit time so each member trains from scratch state
        voting: 'hard' or 'soft'

    Attributes after fit:
        classes_: Sorted unique class labels seen across training
        member_names_: The member names, in declaration order
        estimators_: The fitted member models, in declaration order
        named_estimators_: dict name -> fitted member
    """

    def __init__(
        self,
        estimators: Optional[list] = None,
        voting: str = "hard",
    ):
        self.estimators = estimators
        self.voting = voting

    def _vote_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Tally every member's hard votes into a (n_samples, n_classes) matrix.

        Label values outside ``classes_`` are a member contract violation —
        raised loudly rather than silently mis-bucketed by searchsorted.
        """
        votes = np.zeros((len(X), len(self.classes_)))
        for name, est in zip(self.member_names_, self.estimators_):
            labels = np.asarray(est.predict(X))
            if not np.isin(labels, self.classes_).all():
                raise ValueError(
                    f"member '{name}' predicted labels outside the ensemble's "
                    f"classes_ {self.classes_.tolist()}: "
                    f"{sorted(set(labels) - set(self.classes_.tolist()))}"
                )
            columns = np.searchsorted(self.classes_, labels)
            np.add.at(votes, (np.arange(len(X)), columns), 1.0)
        return votes

    def _proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Average every member's probabilities into one (n, C) matrix."""
        avg = np.zeros((len(X), len(self.classes_)))
        for name, est in zip(self.member_names_, self.estimators_):
            proba = np.asarray(est.predict_proba(X), dtype=float)
            member_classes = getattr(est, "classes_", self.classes_)
            if proba.shape[0] != len(X) or proba.shape[1] != len(member_classes):
                raise ValueError(
                    f"member '{name}' returned proba shaped {proba.shape}; "
                    f"expected ({len(X)}, {len(member_classes)})"
                )
            for col, value in enumerate(member_classes):
                pos = int(np.searchsorted(self.classes_, value))
                avg[:, pos] += proba[:, col]
        return avg / len(self.estimators_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsembleClassifierScratch":
        """
        Fit every member on fresh copies of the template estimators.

        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,)

        Returns:
            self
        """
        if not self.estimators:
            raise ValueError(
                "estimators must be a non-empty list of (name, estimator) tuples"
            )
        if any(not isinstance(name, str) for name, _ in self.estimators):
            raise ValueError("every estimator must be given a string name")
        if len({name for name, _ in self.estimators}) != len(self.estimators):
            raise ValueError("duplicate member names are not allowed")
        if self.voting not in ("hard", "soft"):
            raise ValueError(
                f"voting must be 'hard' or 'soft'; got {self.voting!r}"
            )
        if self.voting == "soft" and any(
            not hasattr(est, "predict_proba") for _, est in self.estimators
        ):
            raise ValueError(
                "soft voting requires every member to implement predict_proba"
            )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.member_names_ = tuple(name for name, _ in self.estimators)

        # deep-copy the (unfitted) templates: each member trains its own state
        self.estimators_ = [deepcopy(est).fit(X, y) for _, est in self.estimators]
        self.named_estimators_ = dict(zip(self.member_names_, self.estimators_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by hard or soft voting.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        if self.voting == "hard":
            votes = self._vote_matrix(X)
        else:
            votes = self._proba_matrix(X)
        # argmax takes the FIRST max: vote/probability ties resolve to the
        # lowest class index (the same order-dependent rule sklearn uses)
        return self.classes_[votes.argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities as the members' averaged probabilities.

        Only meaningful for voting='soft' — hard-voting ensembles have no
        probability model (the votes themselves are the raw material).

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Averaged probabilities of shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=float)
        return self._proba_matrix(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
