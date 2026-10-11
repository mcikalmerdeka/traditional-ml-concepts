"""Every card is dual-engine: the 10 sklearn-only placeholders are unplugged.

The design (2026-10-11 integration): once their scratch twins landed in
src/models/, the slice-3 placeholders upgrade to dual-engine following the
knn/decision-tree pattern — no `sklearn_only=True`, a non-empty `sources`
declaration, and a working scratch fit per card.

RED: all ten cards currently carry sklearn_only=True and empty sources.
The generic contract suite (test_cards_contract) then auto-expands to the
scratch engine for each upgraded card, card by card.
"""

import pytest

from app.registry.discovery import all_cards, get_card

WAS_SKLEARN_ONLY = (
    "svm",
    "naive-bayes",
    "kmeans",
    "dbscan",
    "hierarchical-clustering",
    "pca",
    "neural-network",
    "random-forest",
    "gradient-boosting",
    "voting-ensemble",
)


@pytest.mark.parametrize("card_id", WAS_SKLEARN_ONLY)
def test_placeholder_card_is_now_dual_engine(card_id):
    card = get_card(card_id)
    assert not card.sklearn_only, f"{card_id} still sklearn-only"
    assert card.sources, f"{card_id} declares no scratch sources"


@pytest.mark.parametrize("card_id", WAS_SKLEARN_ONLY)
def test_placeholder_card_scratch_fit_runs(card_id):
    # one scratch fit at declared defaults on the card's first dataset —
    # deeper coverage (every dataset, metrics, viz) is the contract suite's job
    from app.core.datasets import get_dataset
    from app.core.engines import run

    card = all_cards() and get_card(card_id)
    data = get_dataset(card.datasets[0])
    fitted = run(card, data, {h.name: h.default for h in card.hypers}, "scratch")
    assert fitted is not None
