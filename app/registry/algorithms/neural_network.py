"""Card: Neural Network (basics) — dual-engine.

From-scratch MLPClassifierScratch (numpy MLP trained by adam on the softmax
cross-entropy) vs sklearn's MLPClassifier. Same architecture knobs — hidden
width, activation, L2 alpha, max_iter — so per-engine boundaries show where
two independently written optimizers land; they will NOT match exactly
(different initialization draw under the same seed), and that divergence is
the lesson.

The two playgrounds separate two very different knobs: hidden-layer width is
**capacity** (can the boundary bend enough?), while alpha is **regularization**
(how hard should it resist bending?).
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from app.core.card import AlgorithmCard, Select, Slider
from src.models.neural_network_models import MLPClassifierScratch

THEORY = """## Theory

A multilayer perceptron chains linear layers through elementwise nonlinear
activations — every layer:

$$h^{(l)} = \\phi(W^{(l)} h^{(l-1)} + b^{(l)})$$

and the output layer, for classification, reads:

$$\\hat{p} = \\mathrm{softmax}(W^{(L)} h^{(L-1)} + b^{(L)})$$

**Capacity** = width × depth. One hidden layer of width 4 can barely bend
around the moons; width 32 can wrap them with slack. More capacity only
helps if you can still train it — weights come from backprop under the
**adam** optimizer, which adapts a per-parameter step size and is the default
solver here.

**Regularization** — the L2 penalty, added to the loss as

$$+ \\alpha \\|W\\|_2^2$$

— is capacity's counterpart: it makes every weight expensive to keep, so
large $\\alpha$ drags the boundary toward straight (a linear model is the
limit $\\alpha \\to \\infty$). A ConvergenceWarning at small `max_iter` is
*normal* here: the optimizer simply has not reached its tolerance yet — that
is undertraining, not an app error. Watch accuracy drop and the boundary
straighten as you push $\\alpha$ up.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return MLPClassifierScratch(
            hidden_layer_sizes=params["hidden"],
            activation=params["activation"],
            alpha=params["alpha"],
            max_iter=params["max_iter"],
            random_state=0,
        ).fit(data.X, data.y)
    return MLPClassifier(
        hidden_layer_sizes=params["hidden"],
        activation=params["activation"],
        alpha=params["alpha"],
        max_iter=params["max_iter"],
        random_state=0,
    ).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundaries(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    n = card.grid_resolution
    scratch_fig = labeled_scatter(ctx.data.X, ctx.data.y, "scratch")
    decision_boundary(scratch_fig, ctx.scratch.predict, ctx.data.X, "scratch", n)
    sklearn_fig = labeled_scatter(ctx.data.X, ctx.data.y, "sklearn")
    decision_boundary(sklearn_fig, ctx.sklearn.predict, ctx.data.X, "sklearn", n)

    fig = go.Figure()
    fig.add_traces([t.update(xaxis="x", yaxis="y") for t in scratch_fig.data])
    fig.add_traces([t.update(xaxis="x2", yaxis="y2") for t in sklearn_fig.data])
    fig.update_layout(
        title=(
            f"Neural net decision regions — hidden={ctx.params['hidden']}, "
            f"α={ctx.params['alpha']}, {ctx.params['activation']} — "
            "scratch (left) vs sklearn (right)"
        ),
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _accuracy_vs_alpha(ctx):
    # log-spaced alpha sweep: regularization straightens the boundary —
    # watch train fall from its memorizing peak and test rise toward it
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    alphas = list(np.geomspace(0.0001, 1.0, 20))
    fig = go.Figure()
    for name, color in [("scratch", "#EF553B"), ("sklearn", "#636EFA")]:
        train_acc = []
        test_acc = []
        for a in alphas:
            fitted = fit(sub, dict(ctx.params, alpha=a), name)
            train_acc.append(float(accuracy_score(ytr, fitted.predict(Xtr))))
            test_acc.append(float(accuracy_score(yte, fitted.predict(Xte))))
        fig.add_trace(
            go.Scatter(
                x=alphas, y=train_acc, mode="lines+markers",
                name=f"{name} train", line=dict(color=color, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=alphas, y=test_acc, mode="lines+markers",
                name=f"{name} test", line=dict(color=color, dash="dot", width=2),
            )
        )
    fig.update_xaxes(type="log", title="alpha (L2 strength)")
    fig.update_yaxes(title="accuracy", range=[0.4, 1.02])
    fig.update_layout(
        title="Accuracy vs alpha — strong L2 straightens the boundary toward linear",
        template="plotly_white",
    )
    return fig


card = AlgorithmCard(
    id="neural-network",
    title="Neural Network (basics)",
    family="classification",
    when_to_use="one hidden layer's capacity vs regularization — bend the boundary with width, straighten it with alpha",
    theory=THEORY,
    sources=(("src/models/neural_network_models.py", ("MLPClassifierScratch",)),),
    hypers=(
        Select("hidden", ((4,), (16,), (32,)), (16,), "one hidden layer width"),
        Select("activation", ("relu", "tanh", "logistic"), "relu"),
        Slider("alpha", 0.0001, 1.0, 0.0001, 0.01, "L2 strength"),
        Slider("max_iter", 50, 500, 50, 200),
    ),
    datasets=("moons", "circles", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_alpha),
    notes=(
        "max_iter = 50 → ConvergenceWarning in the console: the network is "
        "simply undertrained",
        "alpha → 1.0: the boundary straightens toward linear",
        "(4,) vs (32,) on `moons`: capacity to bend",
        "swap activation relu ↔ logistic: relu's hard corners bend the "
        "boundary; logistic's smooth squashing rounds it",
        "the scratch adam starts from its own draw under the same seed — "
        "boundaries will differ inside the same geometry family; treat "
        "agreement of SHAPE, not of pixels, as the expectation",
    ),
)
