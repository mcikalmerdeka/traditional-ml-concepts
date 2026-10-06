"""DBSCAN degenerate pin — Review Focus #1.

Unlike slice 1's cards, DBSCAN's slider bounds *can* reach a legitimate
failure state: at the smallest eps every point is noise and zero clusters
form. The card must degrade gracefully there — metrics return a value (no
crash, no NaN) and the page stays usable (spec §11).
"""

import numpy as np
from sklearn.metrics import silhouette_score

from app.core.card import Data
from app.core.datasets import get_dataset
from app.core.engines import run
from app.registry.discovery import get_card


def test_all_noise_combo_degrades_gracefully():
    card = get_card("dbscan")
    params = {h.name: h.default for h in card.hypers} | {"eps": 0.1}
    fitted = run(card, get_dataset("kmeans_rings"), params, "sklearn")
    values = dict(card.metrics(fitted, get_dataset("kmeans_rings")))
    assert values["Clusters"] == 0.0  # every point is noise; no crash, no NaN


def test_silhouette_excludes_noise_points():
    # final-review Important #2: scattered noise must not pose as a
    # pseudo-"cluster -1" inside the headline silhouette — the metric's basis
    # must match the sibling clustering pages (non-noise rows only)
    X = np.array(
        [[0, 0], [0, 0.4], [0.4, 0], [10, 10], [10, 10.4], [10.4, 10], [5, 5]],
        dtype=float,
    )
    data = Data(X=X, y=None, note="pin — two dense blobs + one isolated point",
                family="clustering")
    card = get_card("dbscan")
    params = {h.name: h.default for h in card.hypers} | {"eps": 0.9, "min_samples": 2}
    fitted = run(card, data, params, "sklearn")
    labels = fitted.raw.labels_
    assert -1 in labels  # the isolated point is noise
    assert len(set(labels)) - (1 if -1 in labels else 0) == 2  # two real clusters
    values = dict(card.metrics(fitted, data))
    mask = labels != -1
    expected = float(silhouette_score(X[mask], labels[mask]))
    assert values["Silhouette"] == expected



def test_all_noise_page_stays_usable():
    # cleanup minor #5: page-level variant (spec §11) — eps=0.1 is IN-bounds
    # and legal, so the page renders an all-noise lesson, not an error.
    import os

    from streamlit.testing.v1 import AppTest

    from app.paths import ROOT

    os.environ["SMOKE_CARD_ID"] = "dbscan"
    at = AppTest.from_file(str(ROOT / "tests" / "app" / "smoke_runner.py"), default_timeout=120)
    at.run()  # first run instantiates the widgets (sidebar fragment)
    at.sidebar.slider[0].set_value(0.1)  # eps slider (key dbscan::eps)
    at.run()
    assert not at.exception, at.exception
    assert len(at.error) == 0  # eps=0.1 is IN-bounds and legal — not an error, a lesson
