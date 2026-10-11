"""Card: Naive Bayes — GaussianNB.

Dual-engine: from-scratch GaussianNaiveBayesScratch (the same log-likelihood
product, closed form) vs sklearn's GaussianNB; both fit in microseconds, so
every playground renders both engines.

The lesson: per-feature Gaussians carve axis-aligned ellipse contours, so
correlated features — like the moons crescents — bend the boundary. The one
hyperparameter, the variance floor ε, is invisible until 1e-3.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

from app.core.card import AlgorithmCard, Select
from src.models.naive_bayes_models import GaussianNaiveBayesScratch

THEORY = """## Theory

Bayes' rule flips the question around: instead of learning where class 1
lives in feature space, model how each class *generates* features, then
compare generations:

$$P(y \\mid x) \\propto P(y) \\prod_j P(x_j \\mid y)$$

The "naive" part: features are assumed **independent given the class**, so
the joint likelihood is a plain product. With GaussianNB each feature gets
its own class-conditional Gaussian:

$$P(x_j \\mid y) = \\tfrac{1}{\\sqrt{2\\pi\\sigma_{jy}^2}}\\;
e^{-\\frac{(x_j - \\mu_{jy})^2}{2\\sigma_{jy}^2}}$$

Per-feature Gaussians ⇒ each class-likelihood contour is an **ellipse aligned
to the axes** — the independence assumption's geometry, visible in the
decision regions. When features actually correlate (the `moons` crescents),
the true contour is tilted; forcing axis alignment bends the boundary.

The variance floor protects against a real bug: a feature with near-zero
variance in some class makes one $P(x_j \\mid y)$ explode and dominate every
decision. Smoothing widens every class-conditional variance to at least
$$\\sigma^2 + \\epsilon$$
so no single feature silently rules the vote. For most data ε can stay at
its tiny default — watch the regions move only when you drag it to 1e-3.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return GaussianNaiveBayesScratch(var_smoothing=params["var_smoothing"]).fit(
            data.X, data.y
        )
    # GaussianNB has no random_state parameter — pass nothing extra
    return GaussianNB(var_smoothing=params["var_smoothing"]).fit(data.X, data.y)


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
            "GaussianNB decision regions — axis-aligned ellipse contours = "
            f"independence (var_smoothing={ctx.params['var_smoothing']})"
        ),
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _confusion_matrix(ctx):
    # two side-by-side heatmaps: the engines can absorb the crescents'
    # overlap onto different classes — the puzzle lives in the off-diagonal
    labels = sorted(set(ctx.data.y))
    fig = go.Figure()
    for slot, name, fitted_obj in [
        ("x", "scratch", ctx.scratch),
        ("x2", "sklearn", ctx.sklearn),
    ]:
        if fitted_obj is None:
            continue
        pred = np.asarray(fitted_obj.predict(ctx.data.X))
        cm = confusion_matrix(ctx.data.y, pred, labels=labels)
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=[str(i) for i in labels],
                y=[str(i) for i in labels],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
                xaxis=slot,
                yaxis="y",
            )
        )
    fig.update_layout(
        title="Confusion matrix — diagonal = right class, off-diagonal = wrong — "
              "scratch (left) vs sklearn (right)",
        template="plotly_white",
        xaxis=dict(domain=[0.03, 0.43], anchor="y"),
        xaxis2=dict(domain=[0.57, 0.97], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
    )
    return fig


card = AlgorithmCard(
    id="naive-bayes",
    title="Naive Bayes (Gaussian)",
    family="classification",
    when_to_use="fast probabilistic classifier whose independence assumption you can *see* bend",
    theory=THEORY,
    sources=(("src/models/naive_bayes_models.py", ("GaussianNaiveBayesScratch",)),),
    hypers=(
        Select("var_smoothing", (1e-9, 1e-7, 1e-5, 1e-3), 1e-9, "variance floor"),
    ),
    datasets=("moons", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _confusion_matrix),
    notes=(
        "axis-aligned ellipse decision regions = per-feature Gaussians — the "
        "independence assumption drawn",
        "on `moons` the crescents' features are correlated — the independence "
        "assumption bends the boundary where the tilted contour should be",
        "smoothing only matters at 1e-3: watch regions merge as the variance "
        "floor flattens every peak",
        "the confusion matrices show *which* class absorbs the crescents' "
        "overlap — accuracy alone hides the asymmetry",
    ),
)
