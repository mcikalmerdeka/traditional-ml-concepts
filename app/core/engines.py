"""Run an algorithm card's fit under one of the two engines.

The only place that knows which engine is which: cards receive `engine` and
return the raw fitted model; this module wraps it as the engine-agnostic
`Fitted` adapter so visualization code never branches on the engine.
"""

from typing import Literal

from app.core.card import AlgorithmCard, Data, Fitted


def run(
    card: AlgorithmCard,
    data: Data,
    params: dict,
    engine: Literal["scratch", "sklearn"],
) -> Fitted:
    """Fit via the card's glue and wrap as Fitted. Fit errors propagate —
    the page layer catches them and renders a friendly st.error."""
    raw = card.fit(data, params, engine)
    kind = "transform" if card.family == "dimensionality-reduction" else "predict"
    return Fitted(raw=raw, kind=kind)
