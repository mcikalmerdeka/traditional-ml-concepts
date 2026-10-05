import sys
from pathlib import Path

# Launch-contract bootstrap: `streamlit run` puts ONLY the script's directory
# (app/) on sys.path — no cwd, no PYTHONPATH — so the repo root must be added
# here, before any app.* import can resolve. ensure_root_on_path() below is
# then a no-op safety net.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.registry.discovery import discover_cards
from app.core.page import render_card
from app.paths import ROOT, ensure_root_on_path

ensure_root_on_path()

st.set_page_config(page_title="Traditional ML Concepts", layout="wide")

LANDING_INTRO = (
    "Pick an algorithm from the sidebar to open its study page: the core "
    "mathematics, the from-scratch NumPy implementation read live from `src/`, "
    "and a playground where toy datasets plus hyperparameter sliders re-run "
    "both your implementation and scikit-learn side by side. Watch where the "
    "two engines agree — and where they diverge: that gap is where "
    "hyperparameter understanding lives."
)


def render_home():
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


cards = discover_cards()
st.navigation(
    [st.Page(render_home, title="Home", icon="🏠", default=True)]
    + [
        st.Page(lambda c=c: render_card(c), title=c.title, icon="📘", url_path=c.id)
        for c in cards
    ]
).run()
