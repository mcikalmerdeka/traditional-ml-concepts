import os

import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from app.core.card import PlayContext
from app.core.datasets import get_dataset
from app.core.engines import run
from app.paths import ROOT
from app.registry.discovery import all_cards

ALL = all_cards()

SMOKE_RUNNER = str(ROOT / "tests" / "app" / "smoke_runner.py")


def test_at_least_one_card():
    assert len(ALL) >= 1


def test_ids_unique_and_slugs():
    ids = [c.id for c in ALL]
    assert len(ids) == len(set(ids))
    assert all(i.replace("-", "").isalnum() for i in ids)


@pytest.mark.parametrize("engine", ["scratch", "sklearn"])
@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_fit_and_metrics_work_on_every_dataset(card, engine):
    if card.sklearn_only and engine == "scratch":
        pytest.skip("sklearn-only card")
    for ds_id in card.datasets:
        data = get_dataset(ds_id)
        fitted = run(card, data, {h.name: h.default for h in card.hypers}, engine)
        values = card.metrics(fitted, data)
        assert values and all(isinstance(v, float) for _, v in values)


@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_every_viz_returns_figure(card):
    data = get_dataset(card.datasets[0])
    params = {h.name: h.default for h in card.hypers}
    fits = {
        e: run(card, data, params, e)
        for e in ["sklearn"] + ([] if card.sklearn_only else ["scratch"])
    }
    ctx = PlayContext(data, params, fits.get("scratch"), fits["sklearn"])
    for viz in card.visualizations:
        assert isinstance(viz(ctx), go.Figure)


@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_contract_invariants(card):
    card.validate()  # raises on any violation


@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_card_page_smoke(card):
    os.environ["SMOKE_CARD_ID"] = card.id
    at = AppTest.from_file(SMOKE_RUNNER, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
