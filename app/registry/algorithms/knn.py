"""Card: K-Nearest Neighbors (Classifier).

Dual-engine: from-scratch KNNClassifierScratch vs sklearn's
KNeighborsClassifier. The playground shows per-engine decision boundaries and
the accuracy-vs-k curves that expose the bias-variance trade-off in k.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from app.core.card import AlgorithmCard, Select, Slider
from src.models.knn_models import KNNClassifierScratch

THEORY = """## Theory

KNN keeps the entire training set and classifies a query point by its
neighbors' votes. Everything hinges on the distance:

$$d(x, x') = \\sqrt{\\sum_j (x_j - x'_j)^2} \\qquad \\text{(euclidean)}$$

$$d(x, x') = \\sum_j |x_j - x'_j| \\qquad \\text{(manhattan)}$$

$$d_p(x, x') = \\left(\\sum_j |x_j - x'_j|^p\\right)^{1/p} \\qquad \\text{(minkowski)}$$

With `uniform` weighting the k nearest rows vote with equal weight:

$$\\hat{y}(x) = \\operatorname{mode}\\{y_{(1)}, \\dots, y_{(k)}\\}$$

With `distance` weighting, nearer neighbors dominate:

$$\\hat{y}(x) = \\arg\\max_c \\sum_{i: y_i = c,\\, i \\in N_k(x)} \\frac{1}{d(x, x_i)}$$

**k is the bias-variance dial.** Small k → flexible, jagged boundaries that
memorize noise; large k → smooth boundaries that blur class structure. There
is no training phase — prediction cost grows with dataset size. Distances on
unscaled features are dominated by the largest-range feature; these toy sets
are all on comparable scales.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return KNNClassifierScratch(
            n_neighbors=params["n_neighbors"],
            metric=params["metric"],
            p=params["p"],
            weights=params["weights"],
        ).fit(data.X, data.y)
    return KNeighborsClassifier(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        metric=params["metric"],
        p=params["p"],
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
        title="Decision regions — scratch (left) vs sklearn (right)",
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _accuracy_vs_k(ctx):
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    ks = list(range(1, 26))
    # fit once per (engine, k); read both train and test accuracy from it
    fits = {
        name: [fit(sub, dict(ctx.params, n_neighbors=k),
                   "scratch" if name == "scratch" else "sklearn")
               for k in ks]
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
                accuracy_score(y_target, fitted.predict(X_target))
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
    fig.update_layout(
        title="Accuracy vs k — small k memorizes, large k over-smooths",
        template="plotly_white",
        xaxis_title="k (n_neighbors)",
        yaxis_title="accuracy",
    )
    return fig


card = AlgorithmCard(
    id="knn",
    title="K-Nearest Neighbors",
    family="classification",
    when_to_use="small data, locally smooth decision regions, zero training time",
    theory=THEORY,
    sources=(("src/models/knn_models.py", ("KNNClassifierScratch",)),),
    hypers=(
        Slider("n_neighbors", 1, 25, 1, 5, "number of neighbors consulted"),
        Select("weights", ("uniform", "distance"), "uniform", "how neighbors vote"),
        Select("metric", ("euclidean", "manhattan", "minkowski"), "euclidean",
               "distance measure"),
        Slider("p", 1, 5, 1, 2, "minkowski power; used only when metric='minkowski'"),
    ),
    datasets=("moons", "circles", "lin_separable"),
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
    ),
)
