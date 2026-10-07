"""
Behavioral tests for the from-scratch clustering algorithms
(src/models/clustering_models.py).

Each test targets one observable behavior of the public API: fit
shapes and attributes, label handling, agreement with scikit-learn
on easy problems, degenerate-size behavior (k=1, k=n, all-noise eps),
hyperparameter validation, and determinism.
"""

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from src.models.clustering_models import (
    AgglomerativeClusteringScratch,
    DBSCANScratch,
    KMeansScratch,
)


@pytest.fixture
def kmeans_blobs():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    return X


@pytest.fixture
def rings():
    """Concentric rings built exactly like app/core/datasets.py _kmeans_rings."""
    rng = np.random.default_rng(3)
    n = 150
    theta_outer = rng.uniform(0, 2 * np.pi, n)
    theta_inner = rng.uniform(0, 2 * np.pi, n)
    outer = (
        np.c_[5 * np.cos(theta_outer), 5 * np.sin(theta_outer)]
        + rng.normal(0, 0.3, (n, 2))
    )
    inner = (
        np.c_[2 * np.cos(theta_inner), 2 * np.sin(theta_inner)]
        + rng.normal(0, 0.3, (n, 2))
    )
    X = np.vstack([outer, inner])
    truth = np.array([0] * n + [1] * n)
    return X, truth


@pytest.fixture
def noisy_blobs():
    """Three blobs plus low-density uniform rows that DBSCAN should call noise."""
    X, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.9, random_state=7)
    rng = np.random.RandomState(7)
    outliers = rng.uniform(X.min(axis=0) - 1.0, X.max(axis=0) + 1.0, size=(25, 2))
    return np.vstack([X, outliers])


@pytest.fixture
def agglom_blobs():
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=1.0, random_state=42)
    return X


def bridge_data():
    """Two dense blobs joined by a sparse chain of points.

    Single linkage chains the bridge into one cluster; complete linkage
    keeps the two blob halves apart. All randomness is pinned so the
    expected partitions are stable, and a tiny vertical jitter keeps the
    bridge distances distinct — an exact uniform chain has duplicate
    merge distances, where the greedy merge order is ambiguous and even
    sklearn's own tree would be arbitrary.
    """
    rng = np.random.RandomState(11)
    blob_a = rng.randn(30, 2) * 0.5 + [-4.0, 0.0]
    blob_b = rng.randn(30, 2) * 0.5 + [4.0, 0.0]
    xs = np.linspace(-3.0, 3.0, 11)
    bridge = np.c_[xs, np.zeros_like(xs)]
    bridge = bridge + rng.uniform(-0.05, 0.05, size=bridge.shape)
    return np.vstack([blob_a, bridge, blob_b])


# --- K-Means ---------------------------------------------------------------


def test_kmeans_shapes_and_attributes(kmeans_blobs):
    X = kmeans_blobs
    km = KMeansScratch(n_clusters=4, random_state=42)
    km.fit(X)
    assert km.cluster_centers_.shape == (4, X.shape[1])
    assert km.cluster_centers_.dtype == np.float64
    assert km.labels_.shape == (len(X),)
    assert np.array_equal(np.unique(km.labels_), [0, 1, 2, 3])
    assert 1 <= km.n_iter_ <= km.max_iter
    assert isinstance(km.inertia_, float)
    assert km.inertia_ > 0


def test_kmeans_labels_equal_predict(kmeans_blobs):
    X = kmeans_blobs
    km = KMeansScratch(n_clusters=4, random_state=42)
    km.fit(X)
    assert np.array_equal(km.labels_, km.predict(X))


def test_kmeans_agreement_with_sklearn(kmeans_blobs):
    X = kmeans_blobs
    km = KMeansScratch(n_clusters=4, n_init=10, random_state=42)
    km.fit(X)
    ref = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
    assert adjusted_rand_score(km.labels_, ref.labels_) >= 0.9


def test_kmeans_same_seed_gives_identical_result(kmeans_blobs):
    X = kmeans_blobs
    km1 = KMeansScratch(n_clusters=4, random_state=7)
    km2 = KMeansScratch(n_clusters=4, random_state=7)
    km1.fit(X)
    km2.fit(X)
    assert np.array_equal(km1.cluster_centers_, km2.cluster_centers_)
    assert km1.inertia_ == km2.inertia_


def test_kmeans_cross_seed_same_partition(kmeans_blobs):
    """On easy well-separated blobs any seed must recover the same grouping."""
    X = kmeans_blobs
    km1 = KMeansScratch(n_clusters=4, random_state=1)
    km2 = KMeansScratch(n_clusters=4, random_state=2)
    km1.fit(X)
    km2.fit(X)
    assert adjusted_rand_score(km1.labels_, km2.labels_) == 1.0


def test_kmeans_random_init_fits_and_is_deterministic(kmeans_blobs):
    X = kmeans_blobs
    km1 = KMeansScratch(n_clusters=4, init="random", random_state=42)
    km1.fit(X)
    km2 = KMeansScratch(n_clusters=4, init="random", random_state=42)
    km2.fit(X)
    assert km1.cluster_centers_.shape == (4, X.shape[1])
    assert np.array_equal(km1.cluster_centers_, km2.cluster_centers_)


def test_kmeans_elbow_signal(kmeans_blobs):
    X = kmeans_blobs
    inertia_k4 = KMeansScratch(n_clusters=4, random_state=0).fit(X).inertia_
    inertia_k2 = KMeansScratch(n_clusters=2, random_state=0).fit(X).inertia_
    assert inertia_k4 < 0.7 * inertia_k2


def test_kmeans_k_equals_one(kmeans_blobs):
    X = kmeans_blobs
    km = KMeansScratch(n_clusters=1, random_state=0)
    km.fit(X)
    assert np.array_equal(km.labels_, np.zeros(len(X), dtype=int))
    assert km.cluster_centers_.shape == (1, X.shape[1])
    assert np.allclose(km.cluster_centers_[0], X.mean(axis=0))


def test_kmeans_k_equals_n_samples(kmeans_blobs):
    X = kmeans_blobs
    km = KMeansScratch(n_clusters=len(X), n_init=1, random_state=0)
    km.fit(X)
    assert km.inertia_ < 1e-12
    assert np.array_equal(np.sort(km.labels_), np.arange(len(X)))


def test_kmeans_n_clusters_larger_than_samples_raises(kmeans_blobs):
    X = kmeans_blobs
    km = KMeansScratch(n_clusters=len(X) + 1)
    with pytest.raises(ValueError, match="n_clusters"):
        km.fit(X)


def test_kmeans_bad_init_raises(kmeans_blobs):
    km = KMeansScratch(n_clusters=4, init="bogus")
    with pytest.raises(ValueError, match="init"):
        km.fit(kmeans_blobs)


def test_kmeans_n_init_must_be_at_least_one_raises(kmeans_blobs):
    km = KMeansScratch(n_clusters=4, n_init=0)
    with pytest.raises(ValueError, match="n_init"):
        km.fit(kmeans_blobs)


def test_kmeans_max_iter_must_be_at_least_one_raises(kmeans_blobs):
    km = KMeansScratch(n_clusters=4, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        km.fit(kmeans_blobs)


# --- DBSCAN ----------------------------------------------------------------


def test_dbscan_two_blobs_gives_two_clusters_no_noise():
    X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.6, random_state=1)
    db = DBSCANScratch(eps=1.0, min_samples=5)
    labels = db.fit_predict(X)
    assert sorted(set(labels)) == [0, 1]
    assert (labels == -1).sum() == 0


def test_dbscan_solves_rings_where_kmeans_cannot(rings):
    """Two concentric rings: the headline case K-Means cannot represent."""
    X, truth = rings
    db = DBSCANScratch(eps=1.2, min_samples=5)
    db.fit(X)
    assert sorted(set(db.labels_)) == [0, 1]
    assert (db.labels_ == -1).sum() == 0
    assert adjusted_rand_score(db.labels_, truth) == 1.0


def test_dbscan_all_noise_when_eps_tiny(rings):
    X, _ = rings
    db = DBSCANScratch(eps=0.05, min_samples=5)
    db.fit(X)
    assert np.array_equal(db.labels_, np.full(len(X), -1))
    assert db.core_sample_indices_.shape == (0,)
    assert db.components_.shape == (0, X.shape[1])


def test_dbscan_huge_eps_gives_single_cluster(rings):
    X, _ = rings
    db = DBSCANScratch(eps=100.0, min_samples=5)
    db.fit(X)
    assert np.array_equal(db.labels_, np.zeros(len(X), dtype=int))
    assert np.array_equal(db.core_sample_indices_, np.arange(len(X)))
    assert np.array_equal(db.components_, X)


def test_dbscan_core_attributes_are_consistent(noisy_blobs):
    X = noisy_blobs
    db = DBSCANScratch(eps=0.7, min_samples=6)
    db.fit(X)
    assert np.all(np.diff(db.core_sample_indices_) > 0)  # sorted ascending
    assert np.array_equal(db.components_, X[db.core_sample_indices_])
    # every core point is in a cluster, never noise
    assert (db.labels_[db.core_sample_indices_] >= 0).all()


def test_dbscan_agreement_with_sklearn(noisy_blobs):
    X = noisy_blobs
    db = DBSCANScratch(eps=0.7, min_samples=6)
    db.fit(X)
    ref = DBSCAN(eps=0.7, min_samples=6).fit(X)
    ref_labels = ref.labels_
    # same rows flagged as noise; label ids may permute over the rest
    assert np.array_equal(db.labels_ == -1, ref_labels == -1)
    keep = ref_labels != -1
    assert adjusted_rand_score(db.labels_[keep], ref_labels[keep]) == 1.0


def test_dbscan_fit_predict_matches_labels(noisy_blobs):
    X = noisy_blobs
    db = DBSCANScratch(eps=0.7, min_samples=6)
    labels = db.fit_predict(X)
    assert np.array_equal(labels, db.labels_)


def test_dbscan_has_no_predict_method(noisy_blobs):
    """DBSCAN is not a predictive model: no labels for unseen rows."""
    assert not hasattr(DBSCANScratch(), "predict")


@pytest.mark.parametrize("bad_eps", [0, -0.5])
def test_dbscan_eps_must_be_positive_raises(noisy_blobs, bad_eps):
    db = DBSCANScratch(eps=bad_eps, min_samples=5)
    with pytest.raises(ValueError, match="eps"):
        db.fit(noisy_blobs)


@pytest.mark.parametrize("bad_ms", [0, -1, 2.5])
def test_dbscan_min_samples_must_be_positive_int_raises(noisy_blobs, bad_ms):
    db = DBSCANScratch(eps=0.7, min_samples=bad_ms)
    with pytest.raises(ValueError, match="min_samples"):
        db.fit(noisy_blobs)


# --- Agglomerative ---------------------------------------------------------


@pytest.mark.parametrize("linkage", ["ward", "average", "complete", "single"])
def test_agglomerative_agreement_with_sklearn_all_linkages(agglom_blobs, linkage):
    X = agglom_blobs
    model = AgglomerativeClusteringScratch(n_clusters=3, linkage=linkage)
    model.fit(X)
    ref = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(X)
    assert adjusted_rand_score(model.labels_, ref.labels_) == 1.0


def test_agglomerative_ward_agrees_with_sklearn_on_moons():
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    model = AgglomerativeClusteringScratch(n_clusters=2, linkage="ward")
    model.fit(X)
    ref = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X)
    assert adjusted_rand_score(model.labels_, ref.labels_) >= 0.9


def test_agglomerative_labels_shape_and_ids(agglom_blobs):
    X = agglom_blobs
    model = AgglomerativeClusteringScratch(n_clusters=3, linkage="ward")
    model.fit(X)
    assert model.labels_.shape == (len(X),)
    assert np.array_equal(np.unique(model.labels_), [0, 1, 2])
    assert model.n_clusters_ == 3


def test_agglomerative_k_equals_one(agglom_blobs):
    X = agglom_blobs
    model = AgglomerativeClusteringScratch(n_clusters=1, linkage="average")
    model.fit(X)
    assert model.n_clusters_ == 1
    assert np.array_equal(model.labels_, np.zeros(len(X), dtype=int))


def test_agglomerative_n_clusters_must_be_below_n_samples_raises(agglom_blobs):
    X = agglom_blobs
    for bad in [0, -1, 120, 121, 2.5]:
        model = AgglomerativeClusteringScratch(n_clusters=bad, linkage="ward")
        with pytest.raises(ValueError, match="n_clusters"):
            model.fit(X)


def test_agglomerative_bad_linkage_raises(agglom_blobs):
    model = AgglomerativeClusteringScratch(n_clusters=3, linkage="centroid")
    with pytest.raises(ValueError, match="linkage"):
        model.fit(agglom_blobs)


def test_agglomerative_single_chains_bridge_complete_separates():
    X = bridge_data()
    single = AgglomerativeClusteringScratch(n_clusters=2, linkage="single")
    complete = AgglomerativeClusteringScratch(n_clusters=2, linkage="complete")
    single.fit(X)
    complete.fit(X)
    # bridge rows are indices 30..40
    assert len(np.unique(single.labels_[30:41])) == 1  # bridge chains into one cluster
    assert len(np.unique(complete.labels_[30:41])) == 2  # bridge split between blobs
    assert adjusted_rand_score(single.labels_, complete.labels_) < 0.8


def test_agglomerative_matches_sklearn_on_chained_structure():
    X = bridge_data()
    for linkage in ["single", "complete"]:
        model = AgglomerativeClusteringScratch(n_clusters=2, linkage=linkage)
        model.fit(X)
        ref = AgglomerativeClustering(n_clusters=2, linkage=linkage).fit(X)
        assert adjusted_rand_score(model.labels_, ref.labels_) == 1.0


def test_agglomerative_deterministic_no_randomness(agglom_blobs):
    X = agglom_blobs
    m1 = AgglomerativeClusteringScratch(n_clusters=3, linkage="ward")
    m2 = AgglomerativeClusteringScratch(n_clusters=3, linkage="ward")
    m1.fit(X)
    m2.fit(X)
    assert np.array_equal(m1.labels_, m2.labels_)
