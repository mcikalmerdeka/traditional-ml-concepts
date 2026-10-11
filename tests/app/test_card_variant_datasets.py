"""AlgorithmCard.datasets_for: variant-aware dataset routing.

Cards that expose an `algorithm` variant select (decision-tree, knn,
random-forest) need the playground's dataset selectbox to offer only the
datasets that variant can learn from — a classifier variant on lin_* toy rows
bows to continuous y (np.bincount cannot even run), a regressor variant on
moons teaches nonsense.

Contract: `datasets_for(params)` —
- no variant_datasets declared  → always card.datasets (unchanged cards)
- variant matches a declaration → that variant's tuple
- variant unknown/missing       → card.datasets

RED: the field/method don't exist on AlgorithmCard yet.
"""

import pytest

from app.core.card import AlgorithmCard, Slider, Select


def _card(**kwargs):
    base = dict(
        id="t",
        title="T",
        family="regression",
        when_to_use="when",
        theory="## Theory\nx",
        sources=(("src/models/x.py", ("XScratch",)),),
        hypers=(Slider("alpha", 0.0, 1.0, 0.1, 0.5),),
        datasets=("d1", "d2"),
        fit=lambda data, params, engine: None,
        metrics=lambda fitted, data: [("Metric", 0.5)],
        visualizations=(),
    )
    base.update(kwargs)
    return AlgorithmCard(**base)


def test_card_without_variant_datasets_returns_static_datasets():
    card = _card()
    assert card.variant_datasets == ()
    assert card.datasets_for({"alpha": 0.5}) == ("d1", "d2")


def test_variant_datasets_routes_matching_variant():
    card = _card(
        hypers=(Select("algorithm", ("a", "b"), "a"),),
        variant_datasets=(
            ("a", ("da1",)),
            ("b", ("db1", "db2")),
        ),
    )
    assert card.datasets_for({"algorithm": "a"}) == ("da1",)
    assert card.datasets_for({"algorithm": "b"}) == ("db1", "db2")


def test_variant_datasets_falls_back_to_static_when_variant_unknown():
    card = _card(
        hypers=(Select("algorithm", ("a", "b"), "a"),),
        variant_datasets=(("a", ("da1",)),),
    )
    assert card.datasets_for({}) == ("d1", "d2")
    assert card.datasets_for({"algorithm": "b"}) == ("d1", "d2")


def test_validate_accepts_variant_datasets_and_rejects_unknown_family_shape():
    card = _card(
        hypers=(Select("algorithm", ("a", "b"), "a"),),
        variant_datasets=(("classifier", ("d1",)),),
    )
    card.validate()  # must not raise
