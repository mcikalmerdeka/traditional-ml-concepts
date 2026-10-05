import numpy as np
import plotly.graph_objects as go

from app.core.card import AlgorithmCard, Data, Slider
from app.core.page import resolve_hypers


def viz_card():
    def viz(ctx):
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1])
        return fig

    return AlgorithmCard(
        id="viz-dummy",
        title="Viz",
        family="classification",
        when_to_use="",
        theory="t",
        sources=(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),),
        hypers=(Slider("max_depth", 1, 20, 1, 3),),
        datasets=("moons",),
        fit=lambda d, p, e: type(
            "M", (), {"predict": staticmethod(lambda X: X[:, 0])}
        )(),
        metrics=lambda f, d: [("acc", 0.9)],
        visualizations=(viz,),
    )


def test_resolve_hypers_defaults_and_overrides():
    card = viz_card()
    assert resolve_hypers(card.hypers, {}) == {"max_depth": 3}
    assert resolve_hypers(card.hypers, {"max_depth": 9}) == {"max_depth": 9}


def test_viz_is_pure_function_of_context():
    card = viz_card()
    data = Data(
        X=np.array([[0.0, 0.0], [1.0, 1.0]]),
        y=np.array([0, 1]),
        note="n",
        family="classification",
    )
    from app.core.card import PlayContext
    from app.core.engines import run

    s = run(card, data, {}, "scratch")
    k = run(card, data, {}, "sklearn")
    fig = card.visualizations[0](PlayContext(data, {}, s, k))
    assert isinstance(fig, go.Figure)
