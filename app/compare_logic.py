"""Compare-page pure logic (spec: slice-3 design §2/§4).

Family sectioning, per-family dataset union/default selection, and headline
metric extraction — over AlgorithmCard declarations only. No Streamlit
imports: the compare page consumes these functions; tests run them bare.
"""

from app.core.card import AlgorithmCard
from app.core.datasets import dataset_ids

# Section order on the Compare page (spec §4); families with no cards are
# skipped by family_sections.
FAMILY_ORDER: tuple[str, ...] = (
    "regression",
    "classification",
    "clustering",
    "dimensionality-reduction",
    "ensembles",
)

# The metric label that headlines each family (spec §2.3). Read from the
# card's own metrics() output by label match; family_sections' cards supply
# the fallback behavior in headline().
HEADLINE_LABEL: dict[str, str] = {
    "regression": "R²",
    "classification": "Accuracy",
    "clustering": "Silhouette",
    "dimensionality-reduction": "Explained variance",
    "ensembles": "Accuracy",
}

# An undeclared card×dataset cell — shown, never skipped silently (spec §2.2:
# a gap is information).
NOT_DECLARED = "n/a (not declared)"


def family_sections(cards: list) -> list[tuple[str, list]]:
    """Group cards by family in FAMILY_ORDER order, non-empty sections only;
    cards keep their (discovery) order within a family."""
    return [
        (family, [c for c in cards if c.family == family])
        for family in FAMILY_ORDER
        if any(c.family == family for c in cards)
    ]


def dataset_union(cards: list) -> tuple[str, ...]:
    """Dataset ids across the section's cards, first-declaration order —
    i.e. the dataset registry's canonical order (core/datasets.py §10 list),
    filtered to the ids these cards declare. Ruled in slice-3 Task 1: the
    clustering tie-break ("earliest position in the union") must produce the
    spec §2.2 default kmeans_4blobs, which only holds under this reading."""
    declared = {ds for card in cards for ds in card.datasets}
    return tuple(ds for ds in dataset_ids() if ds in declared)


def default_dataset(cards: list) -> str:
    """The union id declared by the most cards; ties break to the earliest
    position in the union (spec §2.2: 'ties/near-ties break on first
    declaration order')."""
    union = dataset_union(cards)
    counts = {ds: sum(ds in c.datasets for c in cards) for ds in union}
    return max(union, key=lambda ds: counts[ds])


def headline(values: list[tuple[str, float]], family: str) -> float | None:
    """First tuple whose label equals the family's HEADLINE_LABEL; else the
    first tuple's value; None on an empty list (spec §2.3)."""
    if not values:
        return None
    for label, value in values:
        if label == HEADLINE_LABEL[family]:
            return value
    return values[0][1]  # fallback: the card's own first metric
