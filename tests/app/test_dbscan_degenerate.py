"""DBSCAN degenerate pin — Review Focus #1.

Unlike slice 1's cards, DBSCAN's slider bounds *can* reach a legitimate
failure state: at the smallest eps every point is noise and zero clusters
form. The card must degrade gracefully there — metrics return a value (no
crash, no NaN) and the page stays usable (spec §11).
"""

from app.core.datasets import get_dataset
from app.core.engines import run
from app.registry.discovery import get_card


def test_all_noise_combo_degrades_gracefully():
    card = get_card("dbscan")
    params = {h.name: h.default for h in card.hypers} | {"eps": 0.1}
    fitted = run(card, get_dataset("kmeans_rings"), params, "sklearn")
    values = dict(card.metrics(fitted, get_dataset("kmeans_rings")))
    assert values["Clusters"] == 0.0  # every point is noise; no crash, no NaN
