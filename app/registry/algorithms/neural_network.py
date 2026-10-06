"""Card: Neural Network (basics) — MLPClassifier (sklearn-only; no from-scratch
implementation exists yet).

When MLPScratch lands in src/models/, this card upgrades to dual-engine by
adding a sources entry and a scratch branch in fit.

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
large $\alpha$ drags the boundary toward straight (a linear model is the
limit $\alpha \\to \\infty$). A ConvergenceWarning at small `max_iter` is
*normal* here: the optimizer simply has not reached its tolerance yet — that
is undertraining, not an app error. Watch accuracy drop and the boundary
straighten as you push $\alpha$ up.
"""


def fit(data, params, engine):
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


def _boundary(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    fig = labeled_scatter(ctx.data.X, ctx.data.y, "data")
    decision_boundary(fig, ctx.sklearn.predict, ctx.data.X, "mlp", card.grid_resolution)
    fig.update_layout(
        title=(
            f"Neural net decision regions — hidden={ctx.params['hidden']}, "
            f"α={ctx.params['alpha']}, {ctx.params['activation']}"
        ),
        template="plotly_white",
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
    train_acc = []
    test_acc = []
    for a in alphas:
        fitted = fit(sub, dict(ctx.params, alpha=a), "sklearn")
        train_acc.append(float(accuracy_score(ytr, fitted.predict(Xtr))))
        test_acc.append(float(accuracy_score(yte, fitted.predict(Xte))))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=alphas, y=train_acc, mode="lines+markers",
            name="train", line=dict(color="#EF553B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=alphas, y=test_acc, mode="lines+markers",
            name="test", line=dict(color="#636EFA", width=2),
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
    sources=(),
    sklearn_only=True,
    hypers=(
        Select("hidden", ((4,), (16,), (32,)), (16,), "one hidden layer width"),
        Select("activation", ("relu", "tanh", "logistic"), "relu"),
        Slider("alpha", 0.0001, 1.0, 0.0001, 0.01, "L2 strength"),
        Slider("max_iter", 50, 500, 50, 200),
    ),
    datasets=("moons", "circles", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundary, _accuracy_vs_alpha),
    notes=(
        "max_iter = 50 → ConvergenceWarning in the console: the network is "
        "simply undertrained",
        "alpha → 1.0: the boundary straightens toward linear",
        "(4,) vs (32,) on `moons`: capacity to bend",
        "swap activation relu ↔ logistic: relu's hard corners bend the "
        "boundary; logistic's smooth squashing rounds it",
    ),
)
