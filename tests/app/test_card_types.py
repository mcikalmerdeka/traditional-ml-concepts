import numpy as np
import pytest

from app.core.card import (
    AlgorithmCard,
    Data,
    Fitted,
    PlayContext,
    Select,
    Slider,
    Toggle,
)


def make_card(**overrides):
    fields = dict(
        id="dummy",
        title="Dummy",
        family="classification",
        when_to_use="testing",
        theory="## Theory\n$$y = Xw$$",
        sources=(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),),
        hypers=(Slider("max_depth", 1, 20, 1, 3),),
        datasets=("moons",),
        fit=lambda data, params, engine: None,
        metrics=lambda fitted, data: [("acc", 1.0)],
        visualizations=(),
    )
    fields.update(overrides)
    return AlgorithmCard(**fields)


def test_fitted_delegates_predict():
    class M:
        def predict(self, X):
            return np.sum(X, axis=1)

    f = Fitted(raw=M(), kind="classification")
    assert list(f.predict(np.array([[1.0, 2.0]]))) == [3.0]


def test_validate_rejects_slider_default_out_of_bounds():
    with pytest.raises(ValueError, match="bounds"):
        make_card(hypers=(Slider("max_depth", 1, 20, 1, 99),)).validate()


def test_validate_rejects_select_default_not_in_options():
    with pytest.raises(ValueError, match="options"):
        make_card(hypers=(Select("crit", ("gini", "entropy"), "chi2"),)).validate()


def test_validate_enforces_sklearn_only_sources_invariant():
    with pytest.raises(ValueError, match="sklearn_only"):
        make_card(sources=(), sklearn_only=False).validate()
    with pytest.raises(ValueError, match="sklearn_only"):
        make_card(sklearn_only=True).validate()


def test_validate_rejects_duplicate_hyper_names():
    with pytest.raises(ValueError, match="duplicate"):
        make_card(
            hypers=(Slider("a", 0, 1, 1, 0), Slider("a", 0, 2, 1, 1))
        ).validate()


def test_select_accepts_non_string_options():
    card = make_card(hypers=(Select("n_init", (1, 10), 10),))
    card.validate()  # must not raise


def test_playcontext_is_frozen():
    ctx = PlayContext(
        data=Data(X=np.zeros((2, 2)), y=None, note="n", family="clustering"),
        params={},
        scratch=None,
        sklearn=None,
    )
    with pytest.raises(Exception):
        ctx.params = {}
