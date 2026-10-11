"""Card: Random Forest — classifier and regressor twins, dual-engine.

`algorithm` selects the variant; both twins wire scratch
(RandomForestClassifierScratch / RandomForestRegressorScratch — honest
bootstrap + sqrt-feature trees in numpy) against sklearn. The B sweep is
log-spaced and coarse (12 points): even after the split-search vectorization
scratch cost grows linearly in B — 40 linearly-spaced points would cost
minutes per fragment rerun, while the log grid keeps the same plateau shape
the lesson needs.

Bagging next to the Decision Tree page: averaging several overconfident
trees, and the ensemble's boundary obeys the *mean* of memorizations instead
of any one of them.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from app.core.card import AlgorithmCard, Select, Slider
from src.models.ensemble_models import (
    RandomForestClassifierScratch,
    RandomForestRegressorScratch,
)

THEORY = """## Theory

Bootstrap aggregation (**bagging**) trains B trees, each on a random
sample-with-replacement of the rows, and averages their predictions:

$$\\hat{f}(x) = \\tfrac{1}{B}\\sum_{b=1}^{B} T_b(x)$$

Each tree is fully grown — a loud, overfit memorizer. The average is much
calmer *because* their errors disagree. For correlated members the mean's
variance obeys:

$$\\mathrm{Var}(\\bar{f}) = \\rho\\sigma^2 + \\tfrac{1 - \\rho}{B}\\sigma^2$$

with ρ the pairwise tree correlation and σ² a single tree's variance. Two
knobs control the formula's two terms: feature subsampling and row bootstrap
keep ρ low, and B pushes the second term toward 0. That is why
`max_depth = 2` with `n_estimators = 200` regularly beats
`max_depth = 20` with `n_estimators = 5`: the forest needs diversity more
than it needs any single tree's full capacity.

Contrast with a **single** tree (see the Decision Tree page): alone it keeps
every memorization — red goes up where test goes down. Inside the forest the
same memorization survives, but averaged away. Watch the boundary straighten
as B grows.
"""


def fit(data, params, engine):
    variant = params["algorithm"]
    if variant == "classifier" and engine == "scratch":
        return RandomForestClassifierScratch(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=0,
        ).fit(data.X, data.y)
    if variant == "regressor" and engine == "scratch":
        return RandomForestRegressorScratch(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=0,
        ).fit(data.X, data.y)
    sk_cls = RandomForestClassifier if variant == "classifier" else RandomForestRegressor
    return sk_cls(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=0,
    ).fit(data.X, data.y)


def metrics(fitted, data):
    # variant metric: classifier models advertise classes_; regressors don't
    pred = fitted.predict(data.X)
    if hasattr(fitted.raw, "classes_"):
        return [("Accuracy", float(accuracy_score(data.y, pred)))]
    return [
        ("R²", float(r2_score(data.y, pred))),
    ]


def _boundaries(ctx):
    if ctx.params["algorithm"] == "regressor":
        return _fit_view(ctx)
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
            f"Random forest decision regions — B={ctx.params['n_estimators']}, "
            f"max_depth={ctx.params['max_depth']} — scratch (left) vs sklearn (right)"
        ),
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _fit_view(ctx):
    # regressor twin: forest averages on a shared predicted-vs-actual axis
    fig = go.Figure()
    for name, fit_obj, color in [
        ("scratch", ctx.scratch, "#EF553B"),
        ("sklearn", ctx.sklearn, "#636EFA"),
    ]:
        pred = fit_obj.predict(ctx.data.X)
        fig.add_trace(
            go.Scatter(
                x=pred,
                y=ctx.data.y,
                mode="markers",
                name=name,
                marker=dict(size=6, color=color, opacity=0.7),
            )
        )
    all_pred_min = min(float(t.x.min()) for t in fig.data)
    all_pred_max = max(float(t.x.max()) for t in fig.data)
    lo = float(min(ctx.data.y.min(), all_pred_min))
    hi = float(max(ctx.data.y.max(), all_pred_max))
    fig.add_shape(
        type="line", x0=lo, y0=lo, x1=hi, y1=hi,
        line=dict(color="gray", dash="dot", width=1),
    )
    fig.update_layout(
        title=(
            f"Predicted vs actual — B={ctx.params['n_estimators']}, "
            f"max_depth={ctx.params['max_depth']}: averaging smooths the "
            "single tree's staircase"
        ),
        template="plotly_white",
        xaxis_title="predicted",
        yaxis_title="actual",
    )
    return fig


def _accuracy_vs_n_estimators(ctx):
    # log-spaced B sweep: bagging's plateau arrives in the first few
    # multiplications of B, which the log axis reads naturally; scratch fits
    # cost linearly in B, so the grid stays coarse in points, not lessons
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    bs = [int(b) for b in np.geomspace(5, 200, 12)]
    is_classifier = ctx.params["algorithm"] == "classifier"

    def score(y_true, y_pred):
        if is_classifier:
            return float(accuracy_score(y_true, y_pred))
        return float(r2_score(y_true, y_pred))

    fig = go.Figure()
    for name, color in [("scratch", "#EF553B"), ("sklearn", "#636EFA")]:
        train_acc = []
        test_acc = []
        for b in bs:
            fitted = fit(sub, dict(ctx.params, n_estimators=b), name)
            train_acc.append(score(ytr, fitted.predict(Xtr)))
            test_acc.append(score(yte, fitted.predict(Xte)))
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
    metric_name = "accuracy" if is_classifier else "R²"
    fig.update_layout(
        title=(
            f"{metric_name} vs n_estimators — more trees calm the vote; "
            "no decay, no overfit"
        ),
        template="plotly_white",
        xaxis_title="n_estimators",
        xaxis_type="log",
        yaxis_title=metric_name,
    )
    return fig


card = AlgorithmCard(
    id="random-forest",
    title="Random Forest",
    family="ensembles",
    when_to_use="many decorrelated overfit trees averaged into one calm classifier — variance reduction by bagging",
    theory=THEORY,
    sources=(
        ("src/models/ensemble_models.py", ("RandomForestClassifierScratch", "RandomForestRegressorScratch")),
    ),
    hypers=(
        Select("algorithm", ("classifier", "regressor"), "classifier",
               "which twin; regression toys route to the regressor"),
        Slider("n_estimators", 5, 200, 5, 100),
        Slider("max_depth", 1, 20, 1, 5, "cap each tree; forest averages the rest"),
    ),
    datasets=("moons", "circles", "blobs_noisy"),
    variant_datasets=(
        ("classifier", ("moons", "circles", "blobs_noisy")),
        ("regressor", ("lin_clean_1f", "lin_noisy_1f", "lin_outliers_1f", "lin_2f")),
    ),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_n_estimators),
    notes=(
        "one deep tree memorizes (`moons`) — the forest averages the "
        "memorization away: depth 20 + B=2 vs depth 20 + B=100",
        "depth 2 + 200 trees often beats depth 20 + 5 trees — the "
        "(1−ρ)/B term rewrites accuracy as diversity × count",
        "watch the boundary straighten as B grows: single-tree islands "
        "smooth into a net-shaped region",
        "compare with the Decision Tree page at the same max_depth: train "
        "accuracy drops, test rises — that transfer is the bagging lesson",
        "scratch tolerates the whole slider but pays honestly for big B × "
        "deep trees — a heavy drag here is seconds, not an error",
    ),
)
