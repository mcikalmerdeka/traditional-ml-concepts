"""Compare-page pure logic (slice 3 Task 1): family sections, dataset
union/default, headline metric extraction. TDD: written BEFORE
app/compare_logic.py (spec: docs/superpowers/specs/2026-10-06-...-slice3-design §2/§4).
"""

import pytest

from app.compare_logic import (
    FAMILY_ORDER, HEADLINE_LABEL, NOT_DECLARED,
    dataset_union, default_dataset, family_sections, headline,
)
from app.registry.discovery import all_cards

ALL = all_cards()


def test_family_sections_cover_all_cards_in_spec_order():
    sections = family_sections(ALL)
    assert [f for f, _ in sections] == [f for f in FAMILY_ORDER]
    assert sum(len(cards) for _, cards in sections) == len(ALL)
    sizes = {f: len(cs) for f, cs in sections}
    assert sizes["regression"] == 1            # linear-regression
    assert sizes["classification"] == 6        # dt, knn, logistic, svm, nb, nn
    assert sizes["clustering"] == 3            # hierarchical, dbscan, kmeans
    assert sizes["dimensionality-reduction"] == 1  # pca
    assert sizes["ensembles"] == 3             # gb, rf, voting


def test_classification_union_and_default():
    cards = [c for c in ALL if c.family == "classification"]
    assert dataset_union(cards) == ("moons", "circles", "lin_separable", "blobs_noisy")
    assert default_dataset(cards) == "moons"   # declared by all 6 — unique max


def test_clustering_tie_breaks_to_earliest_union_position():
    cards = [c for c in ALL if c.family == "clustering"]
    # each of the 3 ids is declared by exactly 2 of 3 cards — a genuine tie
    # (union order = dataset-registry declaration order — see ledger ruling
    # Task 1; default breaks to the earliest union position)
    assert dataset_union(cards) == ("kmeans_4blobs", "kmeans_rings", "var_blobs")
    assert default_dataset(cards) == "kmeans_4blobs"  # earliest in the union


def test_headline_picks_family_label_and_falls_back():
    accuracy = [("Accuracy", 0.91)]
    assert headline(accuracy, "classification") == 0.91
    clustered = [("Clusters", 3.0), ("Noise", 0.02), ("Silhouette", 0.55)]
    assert headline(clustered, "clustering") == 0.55
    all_noise = [("Clusters", 0.0), ("Noise", 1.0)]          # no Silhouette
    assert headline(all_noise, "clustering") == 0.0           # fallback: first
    assert headline([], "clustering") is None
    assert NOT_DECLARED == "n/a (not declared)"
    assert HEADLINE_LABEL["regression"] == "R²"
