"""Variant cards: decision-tree, knn, random-forest gain classifier/regressor
brains behind one page — the linear-regression card's `algorithm` select
pattern.

Contract per card:
- an `algorithm` Select with options ("classifier", "regressor"), default
  "classifier"
- `variant_datasets` routes the regression toys to the regressor variant
  (classifiers must not train on continuous y — Gini's bincount cannot run)
- run() works for BOTH engines under BOTH variants on the variant's first
  dataset; every viz returns a Figure for both variants.

RED: the cards currently expose no `algorithm` select and no
variant_datasets.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from app.core.card import PlayContext
from app.core.datasets import get_dataset
from app.core.engines import run
from app.registry.discovery import get_card

VARIANT_CARDS = ("decision-tree", "knn", "random-forest")


@pytest.mark.parametrize("card_id", VARIANT_CARDS)
def test_card_declares_variant_select_and_routes_datasets(card_id):
    card = get_card(card_id)
    algo = next((h for h in card.hypers if h.name == "algorithm"), None)
    assert algo is not None, f"{card_id} declares no 'algorithm' select"
    assert set(algo.options) == {"classifier", "regressor"}
    assert algo.default == "classifier"

    cls_ds = card.datasets_for({"algorithm": "classifier"})
    reg_ds = card.datasets_for({"algorithm": "regressor"})
    assert cls_ds and reg_ds and set(cls_ds).isdisjoint(reg_ds)
    assert "moons" in cls_ds


@pytest.mark.parametrize("card_id", VARIANT_CARDS)
@pytest.mark.parametrize("engine", ["scratch", "sklearn"])
def test_variant_fits_run_on_both_engines(card_id, engine):
    card = get_card(card_id)
    for variant in ("classifier", "regressor"):
        params = {h.name: h.default for h in card.hypers} | {"algorithm": variant}
        ds_id = card.datasets_for(params)[0]
        data = get_dataset(ds_id)
        fitted = run(card, data, params, engine)
        values = card.metrics(fitted, data)
        assert values and all(isinstance(v, float) for _, v in values)


@pytest.mark.parametrize("card_id", VARIANT_CARDS)
def test_variant_viz_returns_figures(card_id):
    card = get_card(card_id)
    for variant in ("classifier", "regressor"):
        params = {h.name: h.default for h in card.hypers} | {"algorithm": variant}
        ds_id = card.datasets_for(params)[0]
        data = get_dataset(ds_id)
        fits = {e: run(card, data, params, e) for e in ("scratch", "sklearn")}
        ctx = PlayContext(data, params, fits["scratch"], fits["sklearn"])
        for viz in card.visualizations:
            fig = viz(ctx)
            assert isinstance(fig, go.Figure)


def test_random_forest_regressor_scratch_recovers_a_linear_target():
    # the regressor twin must actually regress (not just run): a noisy
    # linear surface with tree ensembles should land well above a
    # constant-mean baseline R²
    from app.core.card import Data

    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(160, 2))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 1 + rng.normal(0, 0.5, 160)
    data = Data(X=X, y=y, note="t", family="regression")
    card = get_card("random-forest")
    params = {h.name: h.default for h in card.hypers} | {"algorithm": "regressor"}
    fitted = run(card, data, params, "scratch")
    values = dict(card.metrics(fitted, data))
    assert values["R²"] > 0.7
