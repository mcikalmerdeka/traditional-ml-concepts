"""Card: K-Nearest Neighbors — classifier and regressor twins, dual-engine.

`algorithm` selects the variant: classifier (majority vote, side-by-side
decision regions, accuracy-vs-k) or regressor (neighbor-mean, predicted-vs-
actual fit view, R²-vs-k). Same distance/weight/metric sliders teach the
bias-variance trade on both variants; the regression toys route through
variant_datasets.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from app.core.card import AlgorithmCard, Select, Slider
from src.models.knn_models import KNNClassifierScratch, KNNRegressorScratch

THEORY = """## Theory

KNN keeps the entire training set and answers a query point from its
neighbors. Everything hinges on the distance:

$$d(x, x') = \\sqrt{\\sum_j (x_j - x'_j)^2} \\qquad \\text{(euclidean)}$$

$$d(x, x') = \\sum_j |x_j - x'_j| \\qquad \\text{(manhattan)}$$

$$d_p(x, x') = \\left(\\sum_j |x_j - x'_j|^p\\right)^{1/p} \\qquad \\text{(minkowski)}$$

With `uniform` weighting the k nearest rows vote with equal weight:

$$\\hat{y}(x) = \\operatorname{mode}\\{y_{(1)}, \\dots, y_{(k)}\\}$$

or, for the regressor variant, the neighbors' targets simply average:

$$\\hat{y}(x) = \\tfrac{1}{k} \\sum_{i \\in N_k(x)} y_i$$

With `distance` weighting, nearer neighbors dominate:

$$\\hat{y}(x) = \\arg\\max_c \\sum_{i: y_i = c,\\, i \\in N_k(x)} \\frac{1}{d(x, x_i)}$$

**k is the bias-variance dial.** Small k → flexible, jagged boundaries that
memorize noise; large k → smooth boundaries that blur class structure. There
is no training phase — prediction cost grows with dataset size. Distances on
unscaled features are dominated by the largest-range feature; these toy sets
are all on comparable scales.
"""


def fit(data, params, engine):
    variant = params["algorithm"]
    scraper_cls = KNNClassifierScratch if variant == "classifier" else KNNRegressorScratch
    if engine == "scratch":
        return scraper_cls(
            n_neighbors=params["n_neighbors"],
            metric=params["metric"],
            p=params["p"],
            weights=params["weights"],
        ).fit(data.X, data.y)
    sk_cls = KNeighborsClassifier if variant == "classifier" else KNeighborsRegressor
    return sk_cls(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        metric=params["metric"],
        p=params["p"],
    ).fit(data.X, data.y)


def metrics(fitted, data):
    # variant metric: classifier models advertise classes_; regressors don't
    pred = fitted.predict(data.X)
    if hasattr(fitted.raw, "classes_"):
        return [("Accuracy", float(accuracy_score(data.y, pred)))]
    return [
        ("R²", float(r2_score(data.y, pred))),
        ("MSE", float(mean_squared_error(data.y, pred))),
    ]


def _boundaries(ctx):
    if ctx.params["algorithm"] == "regressor":
        return _fit_view(ctx)
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
        title="Decision regions — scratch (left) vs sklearn (right)",
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _fit_view(ctx):
    # regressor twin: neighbor means over a shared predicted-vs-actual axis
    # (the diagonal is the perfect fit, any feature count)
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
            f"Predicted vs actual at k={ctx.params['n_neighbors']} — "
            "small k follows the noise, large k flattens toward the mean"
        ),
        template="plotly_white",
        xaxis_title="predicted",
        yaxis_title="actual",
    )
    return fig


def _accuracy_vs_k(ctx):
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    ks = list(range(1, 26))
    is_classifier = ctx.params["algorithm"] == "classifier"

    def score(y_true, y_pred):
        if is_classifier:
            return float(accuracy_score(y_true, y_pred))
        return float(r2_score(y_true, y_pred))

    # fit once per (engine, k); read both train and test scores from it
    fits = {
        name: [fit(sub, dict(ctx.params, n_neighbors=k), name) for k in ks]
        for name in ("scratch", "sklearn")
    }
    fig = go.Figure()
    for name, color in [("scratch", "#EF553B"), ("sklearn", "#636EFA")]:
        for split, target, dash in [
            ("train", (ytr, Xtr), "solid"),
            ("test", (yte, Xte), "dot"),
        ]:
            y_target, X_target = target
            accs = [
                score(y_target, fitted.predict(X_target))
                for fitted in fits[name]
            ]
            fig.add_trace(
                go.Scatter(
                    x=ks,
                    y=accs,
                    name=f"{name} {split}",
                    line=dict(color=color, dash=dash, width=2),
                )
            )
    metric_name = "accuracy" if is_classifier else "R²"
    fig.update_layout(
        title=f"{metric_name} vs k — small k memorizes, large k over-smooths",
        template="plotly_white",
        xaxis_title="k (n_neighbors)",
        yaxis_title=metric_name,
    )
    return fig


card = AlgorithmCard(
    id="knn",
    title="K-Nearest Neighbors",
    family="classification",
    when_to_use="small data, locally smooth decision regions, zero training time",
    theory=THEORY,
    sources=(
        ("src/models/knn_models.py", ("KNNClassifierScratch", "KNNRegressorScratch")),
    ),
    hypers=(
        Select("algorithm", ("classifier", "regressor"), "classifier",
               "which twin; regression toys route to the regressor"),
        Slider("n_neighbors", 1, 25, 1, 5, "number of neighbors consulted"),
        Select("weights", ("uniform", "distance"), "uniform", "how neighbors vote"),
        Select("metric", ("euclidean", "manhattan", "minkowski"), "euclidean",
               "distance measure"),
        Slider("p", 1, 5, 1, 2, "minkowski power; used only when metric='minkowski'"),
    ),
    datasets=("moons", "circles", "lin_separable"),
    variant_datasets=(
        ("classifier", ("moons", "circles", "lin_separable")),
        ("regressor", ("lin_clean_1f", "lin_noisy_1f", "lin_outliers_1f", "lin_2f")),
    ),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_k),
    notes=(
        "k=1 → jagged boundary, perfect train accuracy, worst test accuracy",
        "drag k up: the boundary smooths out — find the sweet spot on `moons`",
        "distance weighting lets closer neighbors dominate: smoother than "
        "uniform at the same k on noisy sets",
        "compare euclidean vs manhattan: the boundary geometry changes shape "
        "(diagonals become axis-aligned stairs)",
        "regressor on `lin noisy 1f`: the fit view shows neighbor means "
        "plateauing — k=1 chases every noise spike, k=25 draws a flat line",
    ),
)
