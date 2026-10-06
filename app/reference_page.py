"""Slice-3 Quick Reference page (spec §5): one dense declaration table over
every discovered card — scan the whole app in one screen, jump anywhere.
No widgets, no fits — pure declaration rendering (G2: auto-corrects as cards
come and go). Page-module body only defines; rendering happens via the
__main__ guard or navigation (import-time rendering forbidden).
"""

import sys
from pathlib import Path

# Launch-contract bootstrap (guarded — idempotent), before any app.* import.
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import streamlit as st

from app.navigation import card_pages
from app.paths import ensure_root_on_path
from app.reference_logic import reference_rows
from app.registry.discovery import all_cards

ensure_root_on_path()

INTRO = (
    "Every discovered card in one table — family, engines, datasets, "
    "hyperparameter defaults, when to use. Jump-link buttons below the table "
    "use the same Page objects as the sidebar (st.dataframe cells cannot "
    "hold page links)."
)


def render_reference():
    st.title("Quick Reference")
    st.markdown(INTRO)
    st.dataframe(reference_rows(all_cards()), hide_index=True)
    for page in card_pages():
        st.page_link(page=page, label=page.title)


if __name__ == "__main__":
    render_reference()
