import numpy as np
import pytest

from app.core.card import AlgorithmCard, Data
from app.core.engines import run


def dummy_card(fit_fn, family="classification"):
    return AlgorithmCard(
        id="d",
        title="D",
        family=family,
        when_to_use="",
        theory="t",
        sources=(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),),
        hypers=(),
        datasets=(),
        fit=fit_fn,
        metrics=lambda f, d: [("m", 1.0)],
        visualizations=(),
    )


def data():
    return Data(X=np.zeros((4, 2)), y=None, note="n", family="clustering")


def test_run_wraps_raw_model_and_delegates_predict():
    class M:
        def predict(self, X):
            return X[:, 0]

    card = dummy_card(lambda d, p, e: M())
    f = run(card, data(), {}, "scratch")
    assert isinstance(f.raw, M)
    assert f.kind == "predict"
    assert list(f.predict(np.array([[7.0, 0.0]]))) == [7.0]


def test_run_sets_transform_kind_for_dim_reduction():
    class M:
        def transform(self, X):
            return X

    card = dummy_card(lambda d, p, e: M(), family="dimensionality-reduction")
    assert run(card, data(), {}, "sklearn").kind == "transform"


def test_run_propagates_fit_errors():
    def boom(d, p, e):
        raise ValueError("bad combo")

    card = dummy_card(boom)
    with pytest.raises(ValueError, match="bad combo"):
        run(card, data(), {}, "sklearn")
