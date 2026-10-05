"""K-Means card guards: sklearn-only identity + degenerate-combo error path.

AppTest cannot deliver a hyperparameter value outside a widget's declared
bounds (streamlit 1.65 validates widget values on render — a pre-run
session-state write of 400 against a 1..10 slider surfaces as an app
exception inside the widget, not as a catchable fit failure). So the
renderer's spec-§11 error path is pinned by forcing the card's fit glue to
raise — the same code path an out-of-bounds k would take — while the engine
test below proves a genuinely degenerate k does raise at the engine layer.
"""

import logging
import os

import pytest
from streamlit.testing.v1 import AppTest

from app.paths import ROOT
from app.core.datasets import get_dataset
from app.core.engines import run
from app.registry.discovery import get_card

SMOKE_RUNNER = str(ROOT / "tests" / "app" / "smoke_runner.py")


def test_kmeans_card_is_sklearn_only():
    card = get_card("kmeans")
    assert card.sklearn_only is True
    assert card.sources == ()


def test_degenerate_k_raises_at_engine_layer():
    # n_clusters=400 > n_samples(300): sklearn raises, engines must propagate
    # so the page layer can turn it into st.error. (n_clusters == n_samples
    # is legal — sklearn accepts it — so 400, not 300, is the degenerate k.)
    card = get_card("kmeans")
    params = {h.name: h.default for h in card.hypers} | {"n_clusters": 400}
    with pytest.raises(ValueError):
        run(card, get_dataset("kmeans_rings"), params, "sklearn")


def test_fit_failure_becomes_error_and_page_stays_usable(caplog):
    """Renderer contract (spec §11): a failing fit renders st.error with the
    params snapshot; the rest of the page (theory, sidebar) stays usable."""
    card = get_card("kmeans")
    original_fit = card.fit

    def boom(data, params, engine):
        raise ValueError("degenerate n_clusters")

    object.__setattr__(card, "fit", boom)  # frozen dataclass — bypass for tests
    try:
        os.environ["SMOKE_CARD_ID"] = "kmeans"
        at = AppTest.from_file(SMOKE_RUNNER, default_timeout=120)
        with caplog.at_level(logging.ERROR, logger="app.core.page"):
            at.run()
        assert not at.exception, at.exception
        assert len(at.error) >= 1
        # params snapshot present in the error message
        assert "n_clusters" in at.error[0].value
        # page stays usable: theory section still rendered
        assert any("Theory" in md.value for md in at.markdown)
        # spec §11: unexpected fit failures also reach the console with a
        # traceback (not just the on-page st.error)
        assert any(r.exc_info for r in caplog.records), caplog.text
    finally:
        object.__setattr__(card, "fit", original_fit)
