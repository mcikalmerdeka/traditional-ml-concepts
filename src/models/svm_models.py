"""
Support Vector Machine classifier implemented from scratch using NumPy.

Includes:
- SVMClassifierScratch: soft-margin SVC, binary via the simplified SMO
  algorithm and multiclass via one-vs-one (exactly like sklearn's SVC)

Core idea. The soft-margin SVM finds the hyperplane f(x) = w·x + b whose
margin (the band y·f(x) >= 1 on the training data) is maximal, while
letting points inside/violating the band pay a slack penalty ``C``. It
is solved through its DUAL: maximize

    W(α) = Σ_i α_i − ½ Σ_ij y_i y_j α_i α_j K(x_i, x_j)
    s.t. Σ_i y_i α_i = 0,  0 <= α_i <= C,

where K is the (possibly non-linear) kernel: RBF and polynomial kernels
make the margin maximal in the kernel-induced feature space. Only rows
with α_i > 0 contribute to the decision function
f(x) = Σ_sv y_sv α_sv K(sv, x) + b — the support vectors, i.e. the points
at or inside the margin. The dual is attacked with the SIMPLIFIED SMO
algorithm (Platt 1998; the CS229 two-variable variant): sweep the rows,
for each row violating its KKT condition at tolerance ``tol`` pair it
with the row maximizing |E_i − E_j| (E = signed prediction error), and
solve that two-variable subproblem in closed form.

SVR (support vector regression) is intentionally out of scope: this
module stays classifier-only, i.e. an SVC twin, not an SVR one
(see also the "no predict_proba" bullet below).

Simplifications vs. scikit-learn (documented on purpose):
- The full dense Gram matrix K is precomputed once per fit (per
  one-vs-one pair on its row subset). That is O(n²) memory but fine for
  the few-hundred-row fits this codebase targets; libsvm instead swaps
  kernel rows in and out of a cache.
- Plain simplified SMO: deterministic full sweeps over every row, no
  working-set selection heuristics, no error cache, no shuffling.
  ``max_iter`` bounds full PASSES over the data, not solver epochs, and
  the optimizer may end early on heavily overlapping data without
  reaching full KKT convergence — a scratch-SMO honesty, not a bug.
- The second alpha j is chosen deterministically: primary pick is
  argmax |E_i − E_j| (ties resolve to the lowest index; the violating row
  itself is excluded), and when that pair is degenerate (a frozen equal-
  label interval L == H, non-negative eta, or a zero-movement clip) the
  remaining candidates are tried in descending gap order until one
  applies a real step — Platt's simplified-SMO fallback ladder with the
  randomness removed. Without the ladder, sweeps can end with zero
  applied steps while genuine KKT violations remain, stalling the fit.
  ``random_state`` is accepted for API parity but currently unused
  because nothing in the training path is random.
- No ``predict_proba``: sklearn's SVC only exposes it behind
  ``probability=True`` (internal Platt scaling), which is out of scope.
- Binary ``dual_coef_`` is a flat signed vector (y_sv * alpha_sv, length
  n_support); sklearn stores shape (1, n_support). Binary ``intercept_``
  is a plain float; sklearn stores shape (1,).
- Multiclass ``support_``/``support_vectors_``/``dual_coef_`` are the
  concatenation of the per-pair blocks in fit order (a row that is a
  support vector of two pairs appears twice); sklearn keeps the
  DE-duplicated rows plus a block-matrix dual layout over them.
- Multiclass ``n_support_`` reports per-pair counts (one row count per
  binary problem, summing to len(support_)); sklearn reports per-class
  counts of its deduplicated support set.
- Multiclass ``decision_function`` and the vote tie-break use
  confidence-weighted scores: a pair (a, b) with decision d adds +|d| to
  the class it favors and −|d| to the other (equivalently
  ``scores[:, b] += d; scores[:, a] -= d``); ``predict`` returns the
  argmax of these sums among the max-vote classes, remaining ties going
  to the first class in sorted order. sklearn counts plain unit votes
  (``break_ties=False``) and aggregates differently when
  ``break_ties=True``.
- ``gamma='scale'`` is resolved once from the RAW training X with
  sklearn's exact formula 1 / (n_features * X.var()) BEFORE any
  one-vs-one subsetting (zero-variance data falls back to 1 /
  n_features, mirroring sklearn's var == 0 handling); ``gamma='auto'``
  means 1 / n_features. The resolved value is kept private as ``_gamma``
  (sklearn keeps it private too).
"""

from typing import Optional

import numpy as np

_VALID_KERNELS = ("linear", "poly", "rbf")

# Alphas at or below this magnitude count as zero when collecting support
# vectors. Alpha updates are clipped to the exact [L, H] bounds, so genuine
# zeros land on 0.0 exactly; the epsilon only filters float dust such as
# -1e-17-style residue and keeps n_support_ stable.
_SV_EPS = 1e-8

# A pair step smaller than this is treated as "no change" (state is frozen
# rather than applied) so that a full sweep can honestly produce zero
# changes and the sweep loop terminates.
_STEP_EPS = 1e-10


def _smo_solve(K: np.ndarray, y: np.ndarray, C: float, tol: float,
               max_iter: int) -> tuple:
    """
    Solve one soft-margin binary dual with the simplified SMO algorithm.

    Parameters:
        K: Precomputed Gram matrix of the (subset of) training rows,
            shape (n, n). Precomputed because every step reads K[i, :],
            K[j, :] and the pivot entries many times over.
        y: Binary targets of shape (n,), values exactly -1.0 / +1.0
            (+1 is the higher sorted class in the one-vs-one setup).
        C: Soft-margin penalty (dual upper bound on every alpha)
        tol: KKT-violation tolerance used by the sweeps
        max_iter: Maximum number of full PASSES over all rows

    Returns:
        (alphas, b, n_iter): dual coefficients (n,), intercept, and the
        number of sweeps executed (<= max_iter).
    """
    n = len(y)
    alphas = np.zeros(n)
    b = 0.0
    # f[k] holds the current f(x_k) = Σ_l y_l α_l K_kl + b for ALL rows.
    # Maintaining it incrementally after every pair update (instead of
    # recomputing from cached errors) guarantees the error vector E is
    # always consistent with the stored (alphas, b) — the classic stale-E
    # SMO bug cannot occur.
    f = np.zeros(n)
    E = f - y  # E[k] = f(x_k) - y_k; equals -y_k at the all-zero start

    n_iter = 0
    while n_iter < max_iter:
        num_changed = 0
        for i in range(n):
            # KKT violation at tolerance tol: either the point sits beyond
            # the margin while its alpha can still grow, or it sits inside
            # while its alpha can still shrink.
            if not (
                (y[i] * E[i] < -tol and alphas[i] < C)
                or (y[i] * E[i] > tol and alphas[i] > 0.0)
            ):
                continue

            # Second-choice ladder (Platt's simplified-SMO fallback, made
            # fully deterministic): try the partner with the largest
            # |E_i - E_j| first (stable argsort resolves ties to the lowest
            # index), then the remaining rows in descending gap order until
            # one yields a real step. The biggest-gap partner is frequently
            # a same-label row pinned at alpha 0 — an equal-label pair at
            # (0, 0) is frozen (its feasible interval collapses to L == H) —
            # so without the ladder a sweep can finish with zero applied
            # steps while genuine KKT violations remain and the fit stalls.
            e_gaps = np.abs(E[i] - E)
            e_gaps[i] = -1.0  # rule out j == i: any real gap beats -1.0
            for j in np.argsort(-e_gaps, kind="stable"):
                j = int(j)
                if j == i:
                    continue

                a_i, a_j = alphas[i], alphas[j]
                if y[i] == y[j]:
                    # Equal labels move together: α_i + α_j is the frozen
                    # sum, so feasibility for α_i is
                    # [max(0, sum - C), min(C, sum)].
                    lo = max(0.0, a_i + a_j - C)
                    hi = min(C, a_i + a_j)
                else:
                    # Opposite labels: α_i - α_j is frozen, giving
                    # [max(0, a_i - a_j), min(C, C + a_i - a_j)] for α_i.
                    lo = max(0.0, a_i - a_j)
                    hi = min(C, C + a_i - a_j)
                if lo == hi:
                    continue

                # Partition with respect to the equality constraint.
                # Negative eta <=> the restricted dual is strictly concave
                # along the constraint line, i.e. the closed-form step below
                # is a true maximum; eta >= 0 means degenerate/flat and is
                # skipped (limit case: identical kernel rows).
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0.0:
                    continue

                # Stationary point of the restricted dual along the feasible
                # line, then clipped to the feasible interval. (Sign check on
                # a hand-solved two-point problem: x = (-1, +1), y = (-1, +1),
                # linear kernel, C large => alpha step of +1/2, the analytic
                # optimum.)
                a_i_new = a_i + y[i] * (E[i] - E[j]) / eta
                a_i_new = min(hi, max(lo, a_i_new))
                # α_j follows from the constraint y_i α_i + y_j α_j = const;
                # the [lo, hi] clip on α_i is chosen so α_j stays in [0, C].
                a_j_new = a_j + y[i] * y[j] * (a_i - a_i_new)
                if abs(a_j_new - a_j) <= _STEP_EPS:
                    continue

                # Intercepts that make f(x_i) == y_i (b1) or f(x_j) == y_j
                # (b2), computed with the FINAL clipped alphas: using
                # pre-clip deltas here is one of the classic SMO bugs.
                b1 = b - E[i] - y[i] * (a_i_new - a_i) * K[i, i] - y[j] * (
                    a_j_new - a_j) * K[i, j]
                b2 = b - E[j] - y[i] * (a_i_new - a_i) * K[i, j] - y[j] * (
                    a_j_new - a_j) * K[j, j]
                # An alpha strictly between its bounds corresponds to a
                # support vector exactly ON the margin, where f is pinned to
                # ±1 by KKT — then the matching b is exact. If both clipping
                # boundaries were hit, only the average b is consistent.
                b_old = b
                if 0.0 < a_i_new < C:
                    b = b1
                elif 0.0 < a_j_new < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                # Fold the pair update into every row's decision value at
                # once, keeping E consistent with the stored (alphas, b).
                delta_i = y[i] * (a_i_new - a_i)
                delta_j = y[j] * (a_j_new - a_j)
                f += delta_i * K[i] + delta_j * K[j] + (b - b_old)
                E = f - y
                alphas[i] = a_i_new
                alphas[j] = a_j_new
                num_changed += 1
                break

        n_iter += 1
        if num_changed == 0:
            break

    return alphas, float(b), n_iter


class SVMClassifierScratch:
    """
    Soft-margin Support Vector classifier from scratch (SVC twin).

    Solves the soft-margin kernel dual with simplified SMO: binary
    problems directly, multiclass problems with one binary SMO per pair
    of classes (sklearn-style) aggregated into per-class scores.

    Parameters:
        C: Soft-margin penalty (slack cost; larger = fewer margin
            violations, harder margin)
        kernel: 'linear', 'poly', or 'rbf'
        degree: Polynomial kernel degree (used when kernel='poly')
        gamma: Kernel width parameter for RBF / scale for poly:
            'scale' = 1 / (n_features * X.var()) on the raw training X
            (sklearn's formula), 'auto' = 1 / n_features, or a positive
            float used directly
        coef0: Constant term of the polynomial kernel
        tol: KKT-violation tolerance that ends the SMO sweeps
        max_iter: Maximum number of full passes over the data (bounds
            passes, not solver epochs; overlapping data may terminate
            early without full convergence)
        random_state: Kept for API parity; currently unused because the
            specified SMO (deterministic j selection, fixed sweep order)
            involves no randomness

    Attributes after fit:
        classes_: Sorted unique class labels
        support_: Indices of the training rows that are support vectors;
            sorted for the binary case, the per-pair concatenation for
            multiclass (a row may appear once per pair it serves)
        support_vectors_: The support rows themselves (n_sv, n_features)
        dual_coef_: Signed support coefficients y_sv * alpha_sv,
            shape (n_sv,) — flat for binary, per-pair concatenation for
            multiclass
        intercept_: Binary: float b. Multiclass: per-pair intercepts,
            shape (n_pairs,)
        n_support_: Per-class counts for binary (shape (2,)); per-pair
            counts for multiclass (shape (n_pairs,))
        n_iter_: Sweeps executed (int for binary; per-pair array for
            multiclass)
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma="scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    # -- validation and kernels -------------------------------------------

    def _validate_hyperparams(self) -> None:
        """Validate hyperparameters; raises ValueError on any bad value."""
        if self.kernel not in _VALID_KERNELS:
            raise ValueError(
                f"kernel must be one of 'linear', 'poly', 'rbf'; got "
                f"{self.kernel!r}"
            )
        if not isinstance(self.C, (int, float, np.integer, np.floating)) or self.C <= 0:
            raise ValueError(f"C must be a positive number; got {self.C!r}")
        if not isinstance(self.degree, (int, np.integer)) or self.degree < 1:
            raise ValueError(
                f"degree must be an integer >= 1; got {self.degree!r}"
            )
        if not isinstance(self.tol, (int, float, np.integer, np.floating)) or self.tol <= 0:
            raise ValueError(f"tol must be a positive number; got {self.tol!r}")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be an integer >= 1; got {self.max_iter!r}"
            )

    def _resolve_gamma(self, n_features: int, x_var: float) -> float:
        """Turn the ``gamma`` option into the float actually used by kernels."""
        g = self.gamma
        if isinstance(g, str):
            if g == "scale":
                # sklearn collapses zero variance to 1.0 before dividing.
                var = x_var if x_var > 0.0 else 1.0
                return 1.0 / (n_features * var)
            if g == "auto":
                return 1.0 / n_features
            raise ValueError(
                f"gamma must be 'scale', 'auto', or a positive number; got "
                f"{g!r}"
            )
        if isinstance(g, (int, float, np.integer, np.floating)):
            if g <= 0:
                raise ValueError(
                    f"gamma must be 'scale', 'auto', or a positive number; got {g!r}"
                )
            return float(g)
        raise ValueError(
            f"gamma must be 'scale', 'auto', or a positive number; got {g!r}"
        )

    def _gram(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Kernel matrix between two coordinate blocks under this fit's kernel."""
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "poly":
            return (self._gamma * (A @ B.T) + self.coef0) ** self._degree

        # RBF: ||a - b||^2 expanded, with tiny negative round-off clipped
        # before the exp so the diagonal is exactly 1.0.
        deltas = (
            (A ** 2).sum(axis=1)[:, None]
            + (B ** 2).sum(axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )
        np.maximum(deltas, 0.0, out=deltas)
        return np.exp(-self._gamma * deltas)

    # -- fitting ------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifierScratch":
        """
        Fit the SVM to the training data.

        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,) (any values; mapped via
                ``classes_`` so labels need not be 0/1)

        Returns:
            self
        """
        self._validate_hyperparams()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        if y.shape != (n_samples,):
            raise ValueError(
                f"X and y have inconsistent lengths: X is {X.shape}, "
                f"y is {y.shape}"
            )

        # Arbitrary labels -> positions 0..C-1 via the sorted class list.
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                f"SVM needs at least 2 classes to define a margin; got "
                f"labels {list(self.classes_)}"
            )
        y_idx = np.searchsorted(self.classes_, y)

        # gamma='scale' must be computed on the RAW training X (sklearn's
        # exact formula) before any one-vs-one subsetting, so every pair's
        # kernels see the same gamma sklearn would resolve.
        self._gamma = self._resolve_gamma(n_features, float(X.var()))
        self._degree = int(self.degree)
        self._C = float(self.C)
        self._tol = float(self.tol)
        self._max_iter = int(self.max_iter)

        # Binary is the single pair (class 0 negative, class 1 positive,
        # where +1 is the higher sorted class). Multiclass expands to all
        # C(n, 2) pairs in lexicographic order, exactly like sklearn's OvO.
        if n_classes == 2:
            self._pair_roles = [(0, 1)]
            masks = [np.ones(n_samples, dtype=bool)]
        else:
            self._pair_roles = [
                (i0, i1)
                for i0 in range(n_classes)
                for i1 in range(i0 + 1, n_classes)
            ]
            masks = [
                (y_idx == i0) | (y_idx == i1) for i0, i1 in self._pair_roles
            ]

        support_parts, dual_parts, intercept_parts = [], [], []
        count_parts, iter_parts = [], []

        for (neg, pos), mask in zip(self._pair_roles, masks):
            sub_X = X[mask]
            signs = np.where(y_idx[mask] == pos, 1.0, -1.0)
            alphas, b, n_iter = self._solve_pair(sub_X, signs)

            sv_local = np.flatnonzero(alphas > _SV_EPS)
            rows = np.flatnonzero(mask)
            block_rows = rows[sv_local]
            block_dual = signs[sv_local] * alphas[sv_local]
            if len(block_rows) > 1:
                # Keep each pair's block sorted by original row index; the
                # signed coefs must follow the same permutation.
                order = np.argsort(block_rows)
                block_rows = block_rows[order]
                block_dual = block_dual[order]

            support_parts.append(block_rows)
            dual_parts.append(block_dual)
            intercept_parts.append(b)
            count_parts.append(len(block_rows))
            iter_parts.append(n_iter)

        if support_parts:
            self.support_ = np.concatenate(support_parts).astype(np.intp)
            self.dual_coef_ = np.concatenate(dual_parts).astype(np.float64)
        else:
            self.support_ = np.array([], dtype=np.intp)
            self.dual_coef_ = np.array([], dtype=np.float64)
        self.support_vectors_ = X[self.support_]

        self._pair_intercepts = np.asarray(intercept_parts, dtype=np.float64)
        # Slice boundaries of each pair's block inside support_/dual_coef_.
        # Kept separate from n_support_ because the binary n_support_ counts
        # PER CLASS (like sklearn), not per pair.
        self._pair_offsets = np.concatenate(
            [
                np.zeros(1, dtype=np.intp),
                np.cumsum(np.asarray(count_parts, dtype=np.intp)),
            ]
        )
        if n_classes == 2:
            # sklearn-mirroring flat shapes for the binary case.
            self.intercept_ = float(self._pair_intercepts[0])
            self.n_support_ = np.asarray(
                [
                    int((self.dual_coef_ < 0).sum()),  # class 0: y = -1
                    int((self.dual_coef_ > 0).sum()),  # class 1: y = +1
                ],
                dtype=np.intp,
            )
            self.n_iter_ = int(iter_parts[0])
        else:
            self.intercept_ = self._pair_intercepts
            self.n_support_ = np.asarray(count_parts, dtype=np.intp)
            self.n_iter_ = np.asarray(iter_parts, dtype=np.intp)
        return self

    def _solve_pair(self, X_sub: np.ndarray, signs: np.ndarray) -> tuple:
        """
        One binary SMO problem.

        Parameters:
            X_sub: Rows of the two classes (m, n_features)
            signs: Targets converted to exactly -1.0 / +1.0

        Returns:
            (alphas, b, n_iter) in the subset's local indexing
        """
        K = self._gram(X_sub, X_sub)
        return _smo_solve(K, signs, self._C, self._tol, self._max_iter)

    # -- prediction ----------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise like sklearn does on prediction before fit."""
        if not hasattr(self, "classes_"):
            raise ValueError(
                "This SVMClassifierScratch instance is not fitted yet; "
                "call fit() first."
            )

    def _pair_decisions(self, X: np.ndarray) -> np.ndarray:
        """
        Raw per-pair decision values, shape (n_samples, n_pairs). Sign
        convention: d > 0 favors the pair's higher sorted class.
        """
        # One cross-kernel between ALL support rows and X; pair blocks are
        # contiguous slices of it, so no per-pair kernel recomputation.
        K = self._gram(self.support_vectors_, X)
        columns = []
        for k in range(len(self._pair_roles)):
            block = slice(
                int(self._pair_offsets[k]), int(self._pair_offsets[k + 1])
            )
            columns.append(
                self.dual_coef_[block] @ K[block] + self._pair_intercepts[k]
            )
        if not columns:
            return np.zeros((len(X), 0))
        return np.column_stack(columns)

    def _aggregate_scores(self, D: np.ndarray) -> np.ndarray:
        """
        Turn raw pair decisions into per-class aggregated scores
        (n_samples, n_classes): each pair adds +|d| to the class it
        favors and -|d| to the other.
        """
        scores = np.zeros((len(D), len(self.classes_)))
        for k, (neg, pos) in enumerate(self._pair_roles):
            scores[:, pos] += D[:, k]
            scores[:, neg] -= D[:, k]
        return scores

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Signed margin against the hyperplane(s).

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Binary: signed margin f(x), shape (n_samples,); positive
            means the classes_[1] side. Multiclass: per-class aggregated
            scores, shape (n_samples, n_classes).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        D = self._pair_decisions(X)
        if len(self.classes_) == 2:
            return D[:, 0]
        return self._aggregate_scores(D)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)

        Notes:
            Binary: sign of the margin (a margin of exactly 0 falls to
            classes_[0]). Multiclass: each pair votes for one of its two
            classes; the max-vote classes are ranked by the aggregated
            score sums with remaining ties resolved to the first class in
            sorted order.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        D = self._pair_decisions(X)

        if len(self.classes_) == 2:
            return self.classes_[(D[:, 0] > 0).astype(int)]

        votes = np.zeros((len(D), len(self.classes_)), dtype=np.intp)
        row_ids = np.arange(len(D))
        for k, (neg, pos) in enumerate(self._pair_roles):
            votes[row_ids, np.where(D[:, k] > 0, pos, neg)] += 1

        best = votes.max(axis=1, keepdims=True)
        # Among the max-vote classes prefer the largest aggregated score
        # sum; np.argmax's first-occurrence rule settles the last ties in
        # ascending class order.
        picked = np.where(votes == best, self._aggregate_scores(D),
                          -np.inf).argmax(axis=1)
        return self.classes_[picked]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
