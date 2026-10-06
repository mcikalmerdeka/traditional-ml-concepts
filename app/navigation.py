"""Single builder of the app's st.Page objects; Home.py and the page scripts
all navigate through these same objects (spec: slice-3 design §3; Streamlit
v1.65: st.page_link to a callable page needs the Page object itself, so
compare/reference jump-links reuse the entries placed in st.navigation).
"""

import streamlit as st

from app.core.page import render_card
from app.registry.discovery import discover_cards

# Landing copy (cleanup minor #8): describes both page kinds honestly —
# 4 of 14 cards run scratch + sklearn side by side; the other 10 run
# scikit-learn alone until their scratch twin lands in src/models/.
LANDING_INTRO = (
    "Pick an algorithm from the sidebar to open its study page: the core "
    "mathematics, a playground where toy datasets plus hyperparameter "
    "sliders re-run the fits. Some algorithms run *your* implementation and "
    "scikit-learn side by side — watch where the two engines agree and "
    "where they diverge: that gap is where hyperparameter understanding "
    "lives. The rest run scikit-learn alone until their scratch twin lands "
    "in `src/models/`. The **Compare** page puts every algorithm's headline "
    "metric in one table per family; **Quick Reference** is a cheat-sheet "
    "over all cards."
)


def render_home() -> None:
    st.title("Traditional ML Concepts")
    st.markdown(LANDING_INTRO)
    st.markdown("## Algorithms")
    by_family: dict[str, list] = {}
    for card in discover_cards():
        by_family.setdefault(card.family, []).append(card)
    for family, cards in by_family.items():
        st.badge(family)
        for card in cards:
            st.markdown(f"**{card.title}** — {card.when_to_use}")


def card_pages() -> list[st.Page]:
    """One st.Page per discovered card, discovery order (G2: adding a card
    file is the whole job)."""
    return [
        st.Page(lambda c=c: render_card(c), title=c.title, icon="📘", url_path=c.id)
        for c in discover_cards()
    ]


def build_nav() -> list[st.Page]:
    """Home (default) + Compare + Quick Reference + every card page."""
    return [
        st.Page(render_home, title="Home", icon="🏠", default=True),
        st.Page(_render_compare_page, title="Compare", icon="📊", url_path="compare"),
        st.Page(
            _render_reference_page,
            title="Quick Reference",
            icon="🗂",
            url_path="quick-reference",
        ),
    ] + card_pages()


def _render_compare_page() -> None:
    from app.compare_page import render_compare

    render_compare()


def _render_reference_page() -> None:
    from app.reference_page import render_reference

    render_reference()
