"""Slice-3 Compare page (spec §4): every family's cards as one row of a
metric table — engines side by side, the app's governing idea applied across
algorithms. Fits happen at declared defaults only; failures leave the page
usable (spec §11). Page-module body only defines; rendering happens via the
__main__ guard or navigation (import-time rendering is forbidden — Review
Focus #5).
"""

import logging
import sys
from pathlib import Path

# Launch-contract bootstrap (guarded — idempotent across reruns), before any
# app.* import; ensure_root_on_path() afterwards is a no-op safety net.
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import streamlit as st

from app.compare_logic import (
    NOT_DECLARED,
    dataset_union,
    default_dataset,
    family_sections,
    headline,
)
from app.core.datasets import get_dataset
from app.core.engines import run
from app.paths import ensure_root_on_path
from app.registry.discovery import all_cards
from app.navigation import card_pages

ensure_root_on_path()

logger = logging.getLogger(__name__)

INTRO = (
    "Every card fit at its declared defaults on the selected dataset — one "
    "row per algorithm, engines side by side. A card that does not declare "
    "the selected dataset shows `n/a (not declared)`; a card whose scratch "
    "twin has not landed yet shows `—` under scratch."
)


def render_compare():
    st.title("Compare")
    st.markdown(INTRO)
    for family, cards in family_sections(all_cards()):
        st.markdown(f"### {family}")
        options = list(dataset_union(cards))
        ds = st.selectbox(
            "dataset",
            options,
            index=options.index(default_dataset(cards)),
            key=f"compare::{family}",
            format_func=lambda s: s.replace("_", " "),
        )
        data = get_dataset(ds)
        st.caption(data.note)
        rows = []
        for card in cards:
            row = {"card": card.title, "page": card.id}
            engines = ["sklearn"] if card.sklearn_only else ["scratch", "sklearn"]
            if ds not in card.datasets:
                row["scratch"] = row["sklearn"] = NOT_DECLARED
                rows.append(row)
                continue
            try:
                fitteds = {
                    eng: run(card, data, {h.name: h.default for h in card.hypers}, eng)
                    for eng in engines
                }
                for eng in ("scratch", "sklearn"):
                    if eng in fitteds:
                        value = headline(card.metrics(fitteds[eng], data), card.family)
                        row[eng] = f"{value:.3f}"
                    else:
                        row[eng] = "—"  # sklearn-only card's empty engine cell
            except Exception as exc:
                logger.exception("compare fit failed for %s on %s", card.id, ds)
                st.toast(f"Fit failed for {card.title} — details in the console.")
                for eng in ("scratch", "sklearn"):
                    row[eng] = f"error: {exc}"
            rows.append(row)
        st.dataframe(rows, hide_index=True)
        failed = [
            r["card"]
            for r in rows
            if str(r.get("scratch", "")).startswith("error")
            or str(r.get("sklearn", "")).startswith("error")
        ]
        if failed:
            st.error(
                f"Fit failed for: {', '.join(failed)} — see the cells above; "
                "every other row still renders."
            )
        for page in [p for p in card_pages() if p.url_path in {c.id for c in cards}]:
            st.page_link(page=page, label=f"open {page.title}")


if __name__ == "__main__":
    render_compare()
