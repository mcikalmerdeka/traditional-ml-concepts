"""Card: Gradient Boosting — dual-engine.

From-scratch GradientBoostingClassifierScratch (stagewise log-loss gradient
descent over regression trees with Newton leaf values) vs sklearn's
GradientBoostingClassifier. The B sweep is log-spaced and coarse (12 points):
scratch cost grows linearly in B and the multiclass booster fits one tree
per class per stage — the log grid keeps the same peak-and-decay shape the
lesson needs without freezing the fragment.

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
from src.models.ensemble_models import GradientBoostingClassifierScratch

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
    if engine == "scratch":
        return GradientBoostingClassifierScratch(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=0,
        ).fit(data.X, data.y)
    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
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
            f"Gradient boosting decision regions — B={ctx.params['n_estimators']}, "
            f"ν={ctx.params['learning_rate']}, max_depth={ctx.params['max_depth']} — "
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


def _accuracy_vs_n_estimators(ctx):
    # the boosting-overfit signature: train climbs to 1.0 while test peaks
    # then decays — most visible at the current learning_rate
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    bs = [int(b) for b in np.geomspace(5, 200, 12)]
    fig = go.Figure()
    for name, color in [("scratch", "#EF553B"), ("sklearn", "#636EFA")]:
        train_acc = []
        test_acc = []
        for b in bs:
            fitted = fit(sub, dict(ctx.params, n_estimators=b), name)
            train_acc.append(float(accuracy_score(ytr, fitted.predict(Xtr))))
            test_acc.append(float(accuracy_score(yte, fitted.predict(Xte))))
        fig.add_trace(
            go.Scatter(
                x=bs, y=train_acc, mode="lines+markers",
                name=f"{name} train", line=dict(color=color, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bs, y=test_acc, mode="lines+markers",
                name=f"{name} test", line=dict(color=color, dash="dot", width=2),
            )
        )
    fig.update_layout(
        title=(
            f"Accuracy vs n_estimators (ν={ctx.params['learning_rate']}) — "
            "train → 1.0; test peaks then decays when ν is large"
        ),
        template="plotly_white",
        xaxis_title="n_estimators",
        xaxis_type="log",
        yaxis_title="accuracy",
    )
    return fig


card = AlgorithmCard(
    id="gradient-boosting",
    title="Gradient Boosting",
    family="ensembles",
    when_to_use="small trees fitted stage by stage on residuals — bias reduction, watch it overfit",
    theory=THEORY,
    sources=(("src/models/ensemble_models.py", ("GradientBoostingClassifierScratch",)),),
    hypers=(
        Slider("n_estimators", 5, 200, 5, 100),
        Slider("learning_rate", 0.01, 1.0, 0.01, 0.1, "shrinkage per stage"),
        Slider("max_depth", 1, 5, 1, 3),
    ),
    datasets=("moons", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_n_estimators),
    notes=(
        "learning_rate 1.0 + 200 trees: test accuracy decays — the classic "
        "boosting overfit, the signature you came here to see",
        "low learning_rate needs more trees but lands steadier — watch the "
        "test curve flatten instead of dropping",
        "compare with the Random Forest page: boosting vs bagging — one "
        "peaks then decays, one plateaus",
        "on `blobs noisy` each stage fits one tree per class — the scratch "
        "twin pays triple there; the log-spaced sweep keeps it usable",
    ),
)
