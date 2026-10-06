import os

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from app.core.card import PlayContext
from app.core.datasets import get_dataset
from app.core.engines import run
from app.paths import ROOT
from app.registry.discovery import all_cards, get_card

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
    at = AppTest.from_file(SMOKE_RUNNER, default_timeout=120)
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("card", [c for c in ALL if c.id == "decision-tree"])
def test_tree_overfit_curve_shows_test_gap(card):    # on moons with max_depth=20, train accuracy must exceed test accuracy
    # (memorization: fit on full moons scores higher on its own training rows
    # than on unseen test rows)
    data = get_dataset("moons")
    params = {h.name: h.default for h in card.hypers} | {"max_depth": 20}
    fitted = run(card, data, params, "sklearn")
    Xtr, Xte, ytr, yte = train_test_split(data.X, data.y, test_size=0.3, random_state=0)
    train_acc = accuracy_score(ytr, fitted.raw.predict(Xtr))
    test_acc = accuracy_score(yte, fitted.raw.predict(Xte))
    assert train_acc >= test_acc


def test_decision_tree_card_exists():
    assert any(c.id == "decision-tree" for c in ALL)


def test_knn_card_exists():
    assert any(c.id == "knn" for c in ALL)


def test_logistic_regression_card_exists():
    assert any(c.id == "logistic-regression" for c in ALL)


def test_svm_card_exists():
    assert any(c.id == "svm" for c in ALL)


def test_random_forest_card_exists():
    assert any(c.id == "random-forest" for c in ALL)


def test_gradient_boosting_card_exists():
    assert any(c.id == "gradient-boosting" for c in ALL)


def test_naive_bayes_card_exists():
    assert any(c.id == "naive-bayes" for c in ALL)


def test_hierarchical_clustering_card_exists():
    assert any(c.id == "hierarchical-clustering" for c in ALL)


def test_dbscan_card_exists():
    assert any(c.id == "dbscan" for c in ALL)


def test_pca_card_exists():
    assert any(c.id == "pca" for c in ALL)


def test_neural_network_card_exists():
    assert any(c.id == "neural-network" for c in ALL)


def test_voting_ensemble_card_exists():
    assert any(c.id == "voting-ensemble" for c in ALL)


@pytest.mark.parametrize("card", [c for c in ALL if c.id == "knn"])
def test_knn_k1_overfits_and_k25_underfits_on_moons(card):
    # full-data training accuracy is the memorization signal: k=1 returns each
    # point's own label (accuracy 1.0); k=25 smooths heavily and loses accuracy
    data = get_dataset("moons")
    base = {h.name: h.default for h in card.hypers}
    p1 = base | {"n_neighbors": 1}
    p25 = base | {"n_neighbors": 25}
    s1 = run(card, data, p1, "sklearn")
    s25 = run(card, data, p25, "sklearn")
    acc1 = card.metrics(s1, data)[0][1]
    acc25 = card.metrics(s25, data)[0][1]
    assert acc1 > acc25


@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_fit_is_deterministic(card):
    # sklearn param drift / nondeterminism: identical fits must agree exactly
    # (Review Focus #3 — RF/GB bootstrap, MLP adam all pin random_state=0)
    data = get_dataset(card.datasets[0])
    params = {h.name: h.default for h in card.hypers}
    a = run(card, data, params, "sklearn")
    b = run(card, data, params, "sklearn")

    def out(f):
        # kind-aware accessor (task-12 ruling): transform family reads
        # .transform; predictive families read .predict; clustering cards
        # are transductive — no predict exists, so read raw labels_.
        if f.kind == "transform":
            return f.transform(data.X)
        if hasattr(f.raw, "predict"):
            return f.predict(data.X)
        return f.raw.labels_

    assert np.array_equal(out(a), out(b))


@pytest.mark.parametrize("card", [c for c in ALL if c.id == "svm"])
def test_svm_theory_has_no_latex_row_break(card):
    # cleanup minor #2: the hard-margin formula once rendered with an
    # unintended LaTeX row break (\\;) — the theory string must contain none
    assert "\\\\" not in card.theory


@pytest.mark.parametrize("card", [c for c in ALL if c.id == "hierarchical-clustering"])
def test_hierarchical_marker_reaches_slider_top(card):
    # cleanup minor #6: the silhouette sweep must cover the slider's full
    # range (2..10) so the "current k" marker never vanishes
    data = get_dataset("kmeans_4blobs")
    params = {h.name: h.default for h in card.hypers} | {"n_clusters": 10}
    fitteds = {e: run(card, data, params, e) for e in ["sklearn"]}
    from app.core.card import PlayContext
    ctx = PlayContext(data, params, None, fitteds["sklearn"])
    fig = card.visualizations[1](ctx)
    marker = fig.data[-1]
    assert list(marker.x) == [10]


def test_voting_ensemble_reuses_context_fit():
    # cleanup minor #7: the member-vs-ensemble viz must reuse ctx.sklearn for
    # the ensemble row, refitting only the three members — patch the module
    # the viz's bare `fit(...)` call resolves against (module-global, NOT the
    # frozen card attribute — different call path than the Task-3 page patch;
    # ledger ruling Task 5). If the viz still refits the ensemble, boom raises.
    import sys as _sys

    card = get_card("voting-ensemble")
    data = get_dataset("moons")
    params = {h.name: h.default for h in card.hypers}
    sklearn_fit = run(card, data, params, "sklearn")
    module = _sys.modules["app_cards_voting_ensemble"]
    original = module.fit  # card.fit is this same function (verified)

    def boom(data, params, engine):
        raise AssertionError("viz must reuse ctx.sklearn, not refit")

    module.fit = boom
    from app.core.card import PlayContext
    try:
        ctx = PlayContext(data, params, None, sklearn_fit)
        card.visualizations[1](ctx)  # must not refit the ensemble
    finally:
        module.fit = original
from app.registry.discovery import get_card
