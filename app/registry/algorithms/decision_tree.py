"""Card: Decision Tree Classifier.

Dual-engine: from-scratch DecisionTreeClassifierScratch vs sklearn's
DecisionTreeClassifier. The playground shows decision boundaries per engine
and the train/test accuracy-vs-depth curves that expose overfitting.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from app.core.card import AlgorithmCard, Select, Slider, Toggle
from src.models.tree_models import DecisionTreeClassifierScratch

THEORY = """## Theory

A decision tree predicts by asking a sequence of yes/no feature questions.
Each split partitions the training rows to reduce **impurity**. Entropy-based
impurity (used by the scratch implementation's `entropy` option):

$$H(y) = -\\sum_c p_c \\log_2 p_c$$

Gini impurity (`gini` option):

$$G(y) = 1 - \\sum_c p_c^2$$

A candidate split on feature $j$ at threshold $t$ is scored by the impurity
drop — **information gain**:

$$IG = H(parent) - \\sum_{branches} \\tfrac{n_b}{n} \\, H(branch_b)$$

The tree greedily picks the split with the highest gain, recursively, until a
stopping criterion fires: `max_depth` reached, a node has fewer rows than
`min_samples_split`, or a node is pure. With no depth limit a tree can keep
splitting until every leaf is a single training row — memorizing the training
set and generalizing badly. Regularize with `max_depth` and
`min_samples_split`; criterion choice (gini vs entropy) rarely matters much.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return DecisionTreeClassifierScratch(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            criterion=params["criterion"],
        ).fit(data.X, data.y)
    return DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        criterion=params["criterion"],
        random_state=0,
    ).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundaries(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    scratch_fig = labeled_scatter(ctx.data.X, ctx.data.y, "scratch")
    decision_boundary(scratch_fig, ctx.scratch.predict, ctx.data.X, "scratch")
    sklearn_fig = labeled_scatter(ctx.data.X, ctx.data.y, "sklearn")
    decision_boundary(sklearn_fig, ctx.sklearn.predict, ctx.data.X, "sklearn")

    fig = go.Figure()
    fig.add_traces([t.update(xaxis="x", yaxis="y") for t in scratch_fig.data])
    fig.add_traces([t.update(xaxis="x2", yaxis="y2") for t in sklearn_fig.data])
    fig.update_layout(
        title="Decision regions — scratch (left) vs sklearn (right)",
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _accuracy_vs_depth(ctx):
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    depths = list(range(1, 21))
    series = {"scratch": {}, "sklearn": {}}
    for depth in depths:
        params = dict(ctx.params, max_depth=depth)
        for name, engine in [("scratch", "scratch"), ("sklearn", "sklearn")]:
            fitted = fit(sub, params, engine)
            series[name][depth] = (
                accuracy_score(ytr, fitted.predict(Xtr)),
                accuracy_score(yte, fitted.predict(Xte)),
            )
    fig = go.Figure()
    for name, color, dash in [
        ("scratch", "#EF553B", "solid"),
        ("sklearn", "#636EFA", "solid"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=depths,
                y=[series[name][d][0] for d in depths],
                name=f"{name} train",
                line=dict(color=color, dash=dash, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=depths,
                y=[series[name][d][1] for d in depths],
                name=f"{name} test",
                line=dict(color=color, dash="dot", width=2),
            )
        )
    fig.update_layout(
        title="Accuracy vs max_depth — train and test split apart as the tree memorizes",
        template="plotly_white",
        xaxis_title="max_depth",
        yaxis_title="accuracy",
    )
    return fig


card = AlgorithmCard(
    id="decision-tree",
    title="Decision Tree",
    family="classification",
    when_to_use="non-linear rules with human-readable logic — watch it overfit",
    theory=THEORY,
    sources=(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),),
    hypers=(
        Slider("max_depth", 1, 20, 1, 3, "maximum tree depth"),
        Slider("min_samples_split", 2, 20, 1, 2, "minimum rows to consider a split"),
        Select("criterion", ("gini", "entropy"), "gini", "split quality measure"),
    ),
    datasets=("moons", "circles", "lin_separable"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_depth),
    notes=(
        "on `moons`, drag max_depth 3 → 20: train accuracy → 1.0 while test "
        "collapses — memorization",
        "min_samples_split fights overfitting from the other side — raise it "
        "with a deep tree",
        "switch criterion gini ↔ entropy: boundaries barely move — try it and see",
        "compare scratch vs sklearn regions: small differences near the moons' "
        "overlap reveal tie-breaking differences",
    ),
)
