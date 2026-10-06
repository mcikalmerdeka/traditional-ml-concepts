"""Card: Gradient Boosting (sklearn-only — no from-scratch implementation exists yet).

When GradientBoostingScratch lands in src/models/ensemble_models.py, this
card upgrades to dual-engine by adding a sources entry and a scratch branch
in fit.

The lesson next to the Random Forest page: boosting reduces *bias* stage by
stage, so more trees keep helping — until the stages begin fitting the noise
and the test curve turns over.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.core.card import AlgorithmCard, Slider

THEORY = """## Theory

Gradient boosting is a **stagewise additive model**: start from a weak
initial guess and grow the prediction by small corrections,

$$F_m(x) = F_{m-1}(x) + \\nu\\, h_m(x)$$

where each new tree `h_m` fits the **negative gradient** of the loss — for
log-loss, those gradients are just the current residuals (probabilities minus
true labels). **Shrinkage** `ν` (sklearn's `learning_rate`) scales each step:
each stage buys a smaller, steadier improvement.

Boosting reduces **bias** stepwise — that is the exact opposite of bagging,
which reduces **variance** by averaging decorrelated full-size trees (the
Random Forest page's lesson). The price: because every stage re-focuses on
the rows that are still wrong, late stages fit increasingly noise-shaped
residuals. With enough trees *and* a high learning rate, train accuracy
climbs to 1.0 while the test curve peaks and then decays — the classic
boosting-overfit signature, and why `learning_rate` small + many trees lands
steadier than `learning_rate` large + few trees.
"""


def fit(data, params, engine):
    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=0,
    ).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundary(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    fig = labeled_scatter(ctx.data.X, ctx.data.y, "data")
    decision_boundary(fig, ctx.sklearn.predict, ctx.data.X, "boosting", card.grid_resolution)
    fig.update_layout(
        title=(
            f"Gradient boosting decision regions — B={ctx.params['n_estimators']}, "
            f"ν={ctx.params['learning_rate']}, max_depth={ctx.params['max_depth']}"
        ),
        template="plotly_white",
    )
    return fig


def _accuracy_vs_n_estimators(ctx):
    # the boosting-overfit signature: train climbs to 1.0 while test peaks
    # then decays — most visible at the current learning_rate
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    bs = list(range(5, 201, 5))
    train_acc = []
    test_acc = []
    for b in bs:
        fitted = fit(sub, dict(ctx.params, n_estimators=b), "sklearn")
        train_acc.append(float(accuracy_score(ytr, fitted.predict(Xtr))))
        test_acc.append(float(accuracy_score(yte, fitted.predict(Xte))))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bs, y=train_acc, mode="lines+markers",
            name="train", line=dict(color="#EF553B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bs, y=test_acc, mode="lines+markers",
            name="test", line=dict(color="#636EFA", width=2),
        )
    )
    fig.update_layout(
        title=(
            f"Accuracy vs n_estimators (ν={ctx.params['learning_rate']}) — "
            "train → 1.0; test peaks then decays when ν is large"
        ),
        template="plotly_white",
        xaxis_title="n_estimators",
        yaxis_title="accuracy",
    )
    return fig


card = AlgorithmCard(
    id="gradient-boosting",
    title="Gradient Boosting",
    family="ensembles",
    when_to_use="small trees fitted stage by stage on residuals — bias reduction, watch it overfit",
    theory=THEORY,
    sources=(),
    sklearn_only=True,
    hypers=(
        Slider("n_estimators", 5, 200, 5, 100),
        Slider("learning_rate", 0.01, 1.0, 0.01, 0.1, "shrinkage per stage"),
        Slider("max_depth", 1, 5, 1, 3),
    ),
    datasets=("moons", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundary, _accuracy_vs_n_estimators),
    notes=(
        "learning_rate 1.0 + 200 trees: test accuracy decays — the classic "
        "boosting overfit, the signature you came here to see",
        "low learning_rate needs more trees but lands steadier — watch the "
        "test curve flatten instead of dropping",
        "compare with the Random Forest page: boosting vs bagging — one "
        "peaks then decays, one plateaus",
    ),
)
