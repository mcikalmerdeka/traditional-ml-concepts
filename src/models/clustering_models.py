"""
Clustering algorithms implemented from scratch using NumPy.

Includes:
- KMeansScratch: Lloyd's algorithm with k-means++ seeding and n_init restarts
- DBSCANScratch: density-based clustering grown from core points
- AgglomerativeClusteringScratch: bottom-up merging with ward / complete /
  average / single linkage via Lance-Williams updates

K-Means alternates two steps until stable: assign every point to its
nearest centroid, then move each centroid to the mean of its points.
The full loop restarts ``n_init`` times from different seeds and the
lowest-inertia run wins, which is what protects the result from bad
seedings. k-means++ seeding picks the first center uniformly at random
and every next one with probability proportional to the squared
distance to the nearest center chosen so far, so the initial centroids
spread across the data instead of collapsing onto one blob.

DBSCAN grows clusters from core points (rows with at least
``min_samples`` neighbours within ``eps``, counting the point itself).
Every core point that can be reached from a cluster's cores through a
chain of core-neighbours joins that cluster, non-core points within
reach of it become border members, and rows no core ever reaches stay
noise (-1).

Agglomerative clustering starts every point as its own cluster and
repeatedly merges the two closest clusters until ``n_clusters`` remain.
Pairwise point distances are computed once; cluster-to-cluster
distances are then maintained with the classic Lance-Williams
recurrences instead of re-measuring.

Simplifications vs. scikit-learn (documented on purpose):
- K-Means always runs plain Lloyd's (no 'elkan' algorithm switch), no
  chunked distance computation, and no ``sample_weight`` support.
- DBSCAN uses one brute-force pairwise distance matrix; sklearn can
  fall back to faster neighbour structures on large data. The labeling
  semantics (index-order expansion, border points claimed by the first
  cluster to reach them) match sklearn's.
- Agglomerative clustering is euclidean-only (``metric`` is fixed), with
  no ``distance_threshold`` early stop and no connectivity constraints.
"""

import numpy as np

_INT = (int, np.integer)
_REAL = (int, float, np.integer, np.floating)


def _as_matrix(X, name="X") -> np.ndarray:
    """Coerce input to a 2-D float64 matrix or raise an informative error."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, n_features); "
            f"got shape {arr.shape}"
        )
    return arr


def _squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise squared euclidean distances between the rows of A and B.

    Uses the ||a||² + ||b||² − 2a·b identity so the whole (nA, nB) matrix
    is one matrix product; small negative values from float round-off are
    clipped because squared distances cannot be negative.
    """
    sq = (
        np.sum(A * A, axis=1)[:, None]
        + np.sum(B * B, axis=1)[None, :]
        - 2.0 * (A @ B.T)
    )
    return np.maximum(sq, 0.0)


def _not_fitted(class_name: str) -> ValueError:
    return ValueError(
        f"This {class_name} instance is not fitted yet; call fit() first."
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value) -> bool:
    return isinstance(value, _INT) and not isinstance(value, (bool, np.bool_))


class KMeansScratch:
    """
    K-Means clustering from scratch (Lloyd's algorithm, k-means++ seeding).

    Parameters:
        n_clusters: Number of clusters to form (int >= 1, <= n_samples)
        init: 'k-means++' (spread seeding) or 'random' (uniform rows)
        n_init: Independent runs of the full algorithm; the lowest-inertia
            run is kept
        max_iter: Maximum Lloyd iterations per run
        tol: Convergence threshold on the Frobenius norm of the centroid
            shift between iterations
        random_state: Seed for reproducible seeding across runs

    Attributes after fit:
        cluster_centers_: Centroid coordinates, shape (n_clusters, n_features)
        labels_: Cluster index of every training row, shape (n_samples,)
        inertia_: Sum of squared distances of each point to its centroid
        n_iter_: Iterations executed by the best run
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        """
        Run the algorithm ``n_init`` times and keep the best solution.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            self
        """
        X = _as_matrix(X)
        n_samples = len(X)
        self._validate_params(n_samples)

        # One RNG drives everything; each restart gets a unique sub-seed so
        # runs never repeat a seeding by accident.
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**31 - 1, size=self.n_init)

        best = None
        for seed in seeds:
            result = self._single_run(X, int(seed))
            if best is None or result[3] < best[3]:
                best = result

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each row to its nearest fitted centroid.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Cluster indices of shape (n_samples,)
        """
        if not hasattr(self, "cluster_centers_"):
            raise _not_fitted("KMeansScratch")
        X = _as_matrix(X)
        return _squared_distances(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on ``X`` and return the training labels (cluster indices).
        """
        self.fit(X)
        return self.labels_

    def _validate_params(self, n_samples: int) -> None:
        k = self.n_clusters
        _require(
            _is_int(k) and k >= 1,
            f"n_clusters must be an int >= 1; got {k!r}",
        )
        _require(
            k <= n_samples,
            f"n_clusters={k} cannot be larger than n_samples={n_samples}",
        )
        _require(
            self.init in ("k-means++", "random"),
            f"init must be 'k-means++' or 'random'; got {self.init!r}",
        )
        for name, value in (("n_init", self.n_init), ("max_iter", self.max_iter)):
            _require(
                _is_int(value) and value >= 1,
                f"{name} must be an int >= 1; got {value!r}",
            )

    def _initial_centers(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Seed ``k`` centroids with k-means++ or by uniform row sampling."""
        k = self.n_clusters
        n_samples = len(X)

        if self.init == "random":
            indices = rng.choice(n_samples, size=k, replace=False)
            return X[indices].copy()

        # k-means++: first center uniform, then sample rows with
        # probability proportional to their squared distance to the
        # nearest center already chosen.
        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rng.randint(n_samples)]
        nearest_sq = _squared_distances(X, centers[:1])[:, 0]
        for c in range(1, k):
            total = nearest_sq.sum()
            if total <= 0:
                # Every row already sits on a chosen center; an exact
                # duplicate keeps the algorithm valid (empty-cluster
                # reseeding resolves collisions later).
                centers[c] = X[rng.randint(n_samples)]
            else:
                centers[c] = X[rng.choice(n_samples, p=nearest_sq / total)]
            next_sq = _squared_distances(X, centers[c : c + 1])[:, 0]
            nearest_sq = np.minimum(nearest_sq, next_sq)
        return centers

    def _single_run(self, X: np.ndarray, seed: int) -> tuple:
        """One full Lloyd's loop; returns (centers, labels, inertia, n_iter)."""
        rng = np.random.RandomState(seed)
        centers = self._initial_centers(X, rng)
        k = self.n_clusters
        n_samples = len(X)

        previous_labels = None
        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            sq_to_centers = _squared_distances(X, centers)
            labels = sq_to_centers.argmin(axis=1)

            # Converged: the assignment is a fixed point of the loop, so
            # the current centroids already are the cluster means.
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            previous_labels = labels

            # Empty cluster: reseed it on the row farthest from its own
            # centroid; that row moves over immediately.
            counts = np.bincount(labels, minlength=k)
            for c in np.flatnonzero(counts == 0):
                own_sq = sq_to_centers[np.arange(n_samples), labels]
                farthest = int(own_sq.argmax())
                centers[c] = X[farthest]
                labels[farthest] = c
                own_sq[farthest] = -1.0  # seed each empty cluster on a distinct row

            new_centers = self._cluster_means(X, labels, k)
            shift = np.sqrt(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            if shift <= self.tol:
                break

        # Recompute against the final centroids so labels_ and inertia_
        # are always exactly consistent with predict() on X.
        final_sq = _squared_distances(X, centers)
        final_labels = final_sq.argmin(axis=1)
        inertia = float(final_sq[np.arange(n_samples), final_labels].sum())
        return centers, final_labels, inertia, n_iter

    def _cluster_means(
        self, X: np.ndarray, labels: np.ndarray, k: int
    ) -> np.ndarray:
        """Centroid of each cluster as the mean of its member rows."""
        sums = np.zeros((k, X.shape[1]))
        # Scatter-add rows; unavoidable pointwise accumulation but only
        # O(n_samples) work overall.
        np.add.at(sums, labels, X)
        counts = np.bincount(labels, minlength=k)
        return sums / counts[:, None]


# --- DBSCAN -----------------------------------------------------------------


class DBSCANScratch:
    """
    Density-based spatial clustering from scratch.

    A point is a core point when at least ``min_samples`` rows (including
    itself) lie within ``eps``. Clusters expand greedily: an unassigned
    core point starts a cluster, and every core point within ``eps`` of a
    member core joins it, recursively (breadth-first, neighbours visited
    in index order). Non-core points reached by the expansion become
    border members; everything else is noise (-1).

    There is deliberately no ``predict`` method — DBSCAN is not a
    predictive model, it only labels the rows it was fitted on
    (scikit-learn's DBSCAN is the same).

    Parameters:
        eps: Neighbourhood radius; two rows within ``eps`` are neighbours
        min_samples: Core-point threshold, counting the point itself

    Attributes after fit:
        labels_: Cluster id per row; 0..k-1 for members, -1 for noise
        core_sample_indices_: Sorted row indices of the core points
        components_: Copies of the core rows, shape (m, n_features)
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X: np.ndarray) -> "DBSCANScratch":
        """
        Label every row as cluster member, border point, or noise.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            self
        """
        self._validate_params()
        X = _as_matrix(X)
        n_samples = len(X)

        # One brute-force pass: neighbourhood(i) = {j : dist(i, j) <= eps},
        # which includes i itself, matching sklearn's min_samples counting.
        distances = np.sqrt(_squared_distances(X, X))
        neighborhoods = distances <= self.eps
        core_mask = neighborhoods.sum(axis=1) >= self.min_samples

        labels = np.full(n_samples, -1, dtype=np.int64)
        cluster_id = 0
        for i in range(n_samples):
            if not core_mask[i] or labels[i] != -1:
                continue

            # Breadth-first expansion over density-connected cores using a
            # plain list + read pointer (no recursion, no collections).
            labels[i] = cluster_id
            frontier = [i]
            ptr = 0
            while ptr < len(frontier):
                core = frontier[ptr]
                ptr += 1
                for j in np.flatnonzero(neighborhoods[core]):
                    if labels[j] != -1:
                        continue
                    if core_mask[j]:
                        labels[j] = cluster_id
                        frontier.append(int(j))
                    else:
                        # Border point: the first cluster whose core
                        # reaches it claims it, and it never expands.
                        labels[j] = cluster_id
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core_mask)
        self.components_ = X[self.core_sample_indices_].copy()
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on ``X`` and return the training labels (noise is -1).
        """
        self.fit(X)
        return self.labels_

    def _validate_params(self) -> None:
        eps = self.eps
        _require(
            isinstance(eps, _REAL) and not isinstance(eps, (bool, np.bool_)) and eps > 0,
            f"eps must be a positive number; got {eps!r}",
        )
        _require(
            _is_int(self.min_samples) and self.min_samples >= 1,
            f"min_samples must be an int >= 1; got {self.min_samples!r}",
        )


# --- Agglomerative -----------------------------------------------------------


class AgglomerativeClusteringScratch:
    """
    Bottom-up (agglomerative) hierarchical clustering from scratch.

    Every point starts as its own cluster; the two closest clusters are
    merged repeatedly until ``n_clusters`` remain. Distances between
    clusters follow one of four linkages and are updated in place with
    the Lance-Williams recurrences, Ward's among them: merging A and B
    costs ``(nA * nB / (nA + nB)) * ||mean_A - mean_B||²`` (kept and
    compared in squared-distance space).

    Clustering is transductive: labels exist only for the fitted rows,
    so there is deliberately no ``predict`` method (scikit-learn's
    AgglomerativeClustering likewise). Distances are euclidean;
    alternatives ('metric') are out of scope here.

    Parameters:
        n_clusters: Number of clusters to stop at (1 <= n_clusters < n_samples)
        linkage: 'ward', 'complete', 'single', or 'average' — the rule that
            measures the distance between two clusters

    Attributes after fit:
        labels_: Cluster id per fitted row, 0..k-1, shape (n_samples,)
        n_clusters_: Number of clusters found (equals ``n_clusters``)
    """

    #: Lance-Williams combination rule for squared distances, as a function
    #: of (na, nb, nk, the two old distances to K, and the merging distance).
    _WARD_INCREASE = staticmethod(
        lambda na, nb, nk, dak, dbk, dab: (
            (na + nk) * dak + (nb + nk) * dbk - nk * dab
        )
        / (na + nb + nk)
    )

    def __init__(self, n_clusters: int = 2, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X: np.ndarray) -> "AgglomerativeClusteringScratch":
        """
        Merge closest clusters until ``n_clusters`` remain, then relabel.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            self
        """
        X = _as_matrix(X)
        n_samples = len(X)
        self._validate_params(n_samples)
        target = self.n_clusters

        # Point-to-point distances are computed exactly once, then only
        # Lance-Williams row updates maintain cluster-to-cluster distances.
        point_sq = _squared_distances(X, X)
        if self.linkage == "ward":
            # Ward's criterion lives in squared-distance space.
            cluster_sq = point_sq.copy()
        else:
            cluster_sq = np.sqrt(point_sq)

        cluster_sq[np.diag_indices(n_samples)] = np.inf
        sizes = np.ones(n_samples)
        representative = np.arange(n_samples)  # cluster id each row belongs to
        active = np.ones(n_samples, dtype=bool)
        remaining = n_samples

        while remaining > target:
            # Smallest pair wins; argmin over the flattened matrix
            # resolves ties by the lowest index, deterministically.
            masked = np.where(active[:, None] & active[None, :], cluster_sq, np.inf)
            a, b = np.unravel_index(np.argmin(masked), cluster_sq.shape)
            combined = self._combine_rows(cluster_sq, sizes, a, b)

            # The merged cluster keeps the lower representative index.
            representative[representative == b] = a
            cluster_sq[a, :] = combined
            cluster_sq[:, a] = combined
            cluster_sq[a, a] = np.inf
            sizes[a] += sizes[b]
            sizes[b] = 0
            active[b] = False
            cluster_sq[b, :] = np.inf
            cluster_sq[:, b] = np.inf
            remaining -= 1

        # Brief, contiguous ids: only the partition matters, not the
        # representative indices themselves.
        _, labels = np.unique(representative, return_inverse=True)
        self.labels_ = labels.astype(np.int64)
        self.n_clusters_ = int(remaining)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on ``X`` and return the fitted rows' labels (no generalization).
        """
        self.fit(X)
        return self.labels_

    def _validate_params(self, n_samples: int) -> None:
        _require(
            self.linkage in ("ward", "average", "complete", "single"),
            f"linkage must be 'ward', 'average', 'complete', or 'single'; "
            f"got {self.linkage!r}",
        )
        k = self.n_clusters
        _require(
            _is_int(k) and 1 <= k,
            f"n_clusters must be an int >= 1; got {k!r}",
        )
        _require(
            k < n_samples,
            f"n_clusters={k} must be strictly smaller than "
            f"n_samples={n_samples}",
        )

    def _combine_rows(
        self, cluster_sq: np.ndarray, sizes: np.ndarray, a: int, b: int
    ) -> np.ndarray:
        """
        Lance-Williams row for the merged pair (A, B) against every other
        active cluster K, applied across the whole distance matrix at once.
        """
        nk = sizes
        na, nb = sizes[a], sizes[b]
        dak, dbk = cluster_sq[a, :], cluster_sq[b, :]
        dab = cluster_sq[a, b]

        if self.linkage == "single":
            combined = np.minimum(dak, dbk)
        elif self.linkage == "complete":
            combined = np.maximum(dak, dbk)
        elif self.linkage == "average":
            combined = (na * dak + nb * dbk) / (na + nb)
        else:  # ward
            combined = self._WARD_INCREASE(na, nb, nk, dak, dbk, dab)
        return combined
