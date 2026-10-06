"""Card: Logistic Regression (the last dual-engine card).

Dual-engine: from-scratch LogisticRegressionScratch (plain batch gradient
descent on the sigmoid) vs sklearn's LogisticRegression (L2-regularized
log-loss via a quasi-Newton solver). The divergence is the lesson: C has no
scratch counterpart, so that slider moves only one boundary.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from app.core.card import AlgorithmCard, Slider
from src.models.linear_models import LogisticRegressionScratch

THEORY = """## Theory

Logistic regression predicts the *probability* that a row belongs to class 1.
Keep the linear model, wrap it in the **sigmoid**:

$$\\sigma(z) = \\tfrac{1}{1 + e^{-z}}$$

so the prediction is:

$$\\hat{p} = \\sigma(w^\\top x + b)$$

Training minimizes the **log-loss** (binary cross-entropy):

$$L = -\\tfrac{1}{n}\\sum_i \\left[y_i \\log \\hat{p}_i + (1-y_i)\\log(1-\\hat{p}_i)\\right]$$

The scratch implementation minimizes it by plain **gradient descent**, stepping
both weights and bias down the gradient:

$$w \\leftarrow w - \\eta \\nabla_w L$$

so `learning_rate` controls how fast (and how stably) it climbs. sklearn
instead minimizes the L2-regularized log-loss — strength `1/C` — with a
quasi-Newton solver (`lbfgs`) that has no learning rate at all: `C` has **no**
scratch counterpart. Regularization keeps $\\hat{p}$ away from 0/1 and shrinks
the boundary's slope; larger `C` ⇒ weaker regularization ⇒ the boundary fits
harder. Both engines are **binary** — this card's datasets are binary only.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return LogisticRegressionScratch(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
        ).fit(data.X, data.y)
    return LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        random_state=0,
    ).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundaries(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    scratch_fig = labeled_scatter(ctx.data.X, ctx.data.y, "scratch")
    decision_boundary(scratch_fig, ctx.scratch.predict, ctx.data.X, "scratch", card.grid_resolution)
    sklearn_fig = labeled_scatter(ctx.data.X, ctx.data.y, "sklearn")
    decision_boundary(sklearn_fig, ctx.sklearn.predict, ctx.data.X, "sklearn", card.grid_resolution)

    fig = go.Figure()
    fig.add_traces([t.update(xaxis="x", yaxis="y") for t in scratch_fig.data])
    fig.add_traces([t.update(xaxis="x2", yaxis="y2") for t in sklearn_fig.data])
    fig.update_layout(
        title="Decision boundaries — scratch (left) vs sklearn (right)",
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _accuracy_vs_learning_rate(ctx):
    # sklearn's accuracy does not depend on the learning rate — the divergence
    # made visible: scratch climbs a sweep, sklearn is a flat reference line.
    data = ctx.data
    sklearn_acc = accuracy_score(data.y, ctx.sklearn.predict(data.X))
    lrs = list(np.geomspace(0.001, 1.0, 20))
    scratch_acc = []
    for lr in lrs:
        fitted = fit(data, dict(ctx.params, learning_rate=lr), "scratch")
        scratch_acc.append(float(accuracy_score(data.y, fitted.predict(data.X))))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lrs,
            y=scratch_acc,
            mode="lines+markers",
            name="scratch — training accuracy",
            line=dict(color="#EF553B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min(lrs), max(lrs)],
            y=[sklearn_acc, sklearn_acc],
            mode="lines",
            name="sklearn (learning-rate independent)",
            line=dict(color="#636EFA", width=3),
        )
    )
    fig.update_xaxes(type="log", title="learning_rate")
    fig.update_yaxes(title="accuracy", range=[0, 1.02])
    fig.update_layout(
        title="Accuracy vs learning_rate — GD overshoots at high η; sklearn has no η",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=1.0),
    )
    return fig


card = AlgorithmCard(
    id="logistic-regression",
    title="Logistic Regression",
    family="classification",
    when_to_use="predicting class probabilities with a linear boundary — the classifier twin of linear regression",
    theory=THEORY,
    sources=(("src/models/linear_models.py", ("LogisticRegressionScratch",)),),
    hypers=(
        Slider("C", 0.01, 10.0, 0.01, 1.0, "inverse regularization — sklearn only, scratch has none"),
        Slider("learning_rate", 0.001, 1.0, 0.001, 0.1, "scratch gradient-descent step"),
        Slider("max_iter", 100, 5000, 100, 1000),
    ),
    datasets=("moons", "circles", "lin_separable"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_learning_rate),
    notes=(
        "drag C — only the sklearn boundary moves: the scratch implementation "
        "has no regularization",
        "learning_rate → 1.0 on `moons`: scratch gradient descent overshoots "
        "and wobbles — watch the sweep dip without converging",
        "max_iter = 100 on `moons`: scratch is still climbing when sklearn "
        "has converged — GD needs iterations, lbfgs needs none",
    ),
)
