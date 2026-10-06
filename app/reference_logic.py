"""Quick-reference pure logic (spec: slice-3 design §5).

One declaration row per card — engines, datasets, hyperparameter defaults,
when-to-use — straight from AlgorithmCard fields. No Streamlit imports: the
quick-reference page consumes these functions; tests run them bare.
"""

from app.core.card import AlgorithmCard


def engines_label(card: AlgorithmCard) -> str:
    """Which engines the card's page actually runs (spec §5)."""
    return "sklearn only" if card.sklearn_only else "scratch + sklearn"


def hypers_label(card: AlgorithmCard) -> str:
    """Declared hypers as `name=default`, comma-joined in declaration order."""
    return ", ".join(f"{h.name}={h.default}" for h in card.hypers)


def reference_rows(cards: list) -> list[dict]:
    """One row per card, discovery order; keys are the reference table's
    column order (spec §5). Page-link cells are carried by st.page_link in
    the page layer — the `page` key holds the card id (url_path)."""
    return [
        {
            "card": card.title,
            "page": card.id,
            "family": card.family,
            "engines": engines_label(card),
            "datasets": ", ".join(card.datasets),
            "hypers": hypers_label(card),
            "when_to_use": card.when_to_use,
        }
        for card in cards
    ]
