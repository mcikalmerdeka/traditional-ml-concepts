import numpy as np
import pytest

from app.core.datasets import dataset_ids, get_dataset


def test_all_ids_present():
    assert set(dataset_ids()) == {
        "lin_clean_1f",
        "lin_noisy_1f",
        "lin_outliers_1f",
        "lin_2f",
        "moons",
        "circles",
        "lin_separable",
        "blobs_noisy",
        "kmeans_4blobs",
        "kmeans_rings",
        "var_blobs",
        "pca_correlated_4f",
    }


def test_unknown_id_raises():
    with pytest.raises(KeyError):
        get_dataset("nope")


@pytest.mark.parametrize("ds_id", list(dataset_ids()))
def test_shapes_and_determinism(ds_id):
    d1, d2 = get_dataset(ds_id), get_dataset(ds_id)
    assert d1.X.shape[0] <= 300 and d1.X.shape[1] <= 4
    assert np.array_equal(d1.X, d2.X)  # deterministic
    if d1.y is not None:
        assert d1.y.shape[0] == d1.X.shape[0]
    assert d1.note and d1.family


def test_clustering_sets_have_no_labels():
    for ds_id in ("kmeans_4blobs", "kmeans_rings"):
        assert get_dataset(ds_id).y is None


def test_dim_reduction_sets_have_no_labels():
    assert get_dataset("pca_correlated_4f").y is None
