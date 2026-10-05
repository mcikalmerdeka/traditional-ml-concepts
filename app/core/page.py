"""The generic page renderer: turns any AlgorithmCard into a full Streamlit page.

Every algorithm page has the identical rhythm (spec §8):
header → theory → code → playground fragment → metrics → notes.
Only the playground lives inside the fragment, so slider moves re-run fits
and figures — not theory rendering.
"""

import inspect

import streamlit as st

from app.core.card import AlgorithmCard, PlayContext, Select, Slider, Toggle
from app.core.datasets import get_dataset
from app.core.engines import run
from app.core.source_code import get_class_source

st.cache_data(get_dataset)  # dataset access is cached (spec §12)


def resolve_hypers(hypers: tuple, values: dict) -> dict:
    """Merge resolved widget values over declared defaults (pure)."""
    return {h.name: values.get(h.name, h.default) for h in hypers}


def render_hypers(hypers: tuple) -> dict:
    """Render each spec to its Streamlit widget; return resolved params."""
    params = {}
    for h in hypers:
        if isinstance(h, Slider):
            params[h.name] = st.sidebar.slider(
                h.name,
                min_value=h.min,
                max_value=h.max,
                value=h.default,
                step=h.step,
                help=h.help,
            )
        elif isinstance(h, Select):
            params[h.name] = st.sidebar.selectbox(
                h.name,
                options=list(h.options),
                index=list(h.options).index(h.default),
                help=h.help,
            )
        elif isinstance(h, Toggle):
            params[h.name] = st.sidebar.toggle(h.name, value=h.default, help=h.help)
        else:
            raise TypeError(f"unknown hyperparameter spec: {h!r}")
    return params


def _render_code_section(card: AlgorithmCard) -> None:
    st.markdown("## The code")
    st.markdown("The from-scratch implementation this page runs, read live from `src/`:")
    if card.sklearn_only:
        st.expander(f"sklearn setup glue for `{card.id}`", expanded=False).code(
            inspect.getsource(card.fit), language="python"
        )
        return
    for rel_path, class_names in card.sources:
        for class_name in class_names:
            with st.expander(f"`{rel_path}` → `{class_name}`", expanded=False):
                st.code(get_class_source(rel_path, class_name), language="python")


@st.fragment
def _playground(card: AlgorithmCard) -> None:
    st.markdown("## Playground")
    dataset_id = st.sidebar.selectbox(
        "Dataset", list(card.datasets), format_func=lambda s: s.replace("_", " ")
    )
    data = get_dataset(dataset_id)
    st.sidebar.caption(data.note)
    params = render_hypers(card.hypers)

    engines = ["sklearn"] if card.sklearn_only else ["scratch", "sklearn"]
    fitteds: dict[str, object] = {}
    try:
        for eng in engines:
            fitteds[eng] = run(card, data, params, eng)
    except Exception as exc:  # bad hyperparameter combos are a learning moment
        st.error(
            f"Fit failed with `{params}` — `{exc}`. "
            "Try different hyperparameter values."
        )
        return

    # metrics row: one column per (engine, metric)
    rows = [
        (eng, label, value)
        for eng in engines
        for label, value in card.metrics(fitteds[eng], data)
    ]
    cols = st.columns(len(rows))
    for col, (eng, label, value) in zip(cols, rows):
        col.metric(label=f"{label} — {eng}", value=f"{value:.3f}")

    for viz in card.visualizations:
        try:
            fig = viz(
                PlayContext(
                    data, params, fitteds.get("scratch"), fitteds["sklearn"]
                )
            )
            st.plotly_chart(fig)  # width defaults to "stretch"
        except Exception as exc:
            st.error(f"Visualization failed with `{params}` — `{exc}`")


def render_card(card: AlgorithmCard) -> None:
    st.title(card.title)
    st.badge(card.family)
    st.markdown(f"**When to use:** {card.when_to_use}")

    st.markdown("## Theory")
    st.markdown(card.theory)

    _render_code_section(card)
    _playground(card)

    if card.notes:
        st.markdown("## What to try")
        st.markdown("\n".join(f"- {note}" for note in card.notes))


def run_card_page(card_id: str) -> None:
    """Entry used by Home.py and the smoke runner: bootstrap, fetch, render."""
    from app.registry.discovery import get_card

    render_card(get_card(card_id))
