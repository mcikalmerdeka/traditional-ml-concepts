"""Card: Naive Bayes — GaussianNB (sklearn-only; no from-scratch implementation exists yet).

When NaiveBayesScratch lands in src/models/ (no module exists yet), this card
upgrades to dual-engine by adding a sources entry and a scratch branch in fit.

The lesson: per-feature Gaussians carve axis-aligned ellipse contours, so
correlated features — like the moons crescents — bend the boundary. The one
hyperparameter, the variance floor ε, is invisible until 1e-3.
"""

import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

from app.core.card import AlgorithmCard, Select

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
    # GaussianNB has no random_state parameter — pass nothing extra
    return GaussianNB(var_smoothing=params["var_smoothing"]).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundary(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    fig = labeled_scatter(ctx.data.X, ctx.data.y, "data")
    decision_boundary(fig, ctx.sklearn.predict, ctx.data.X, "gaussian-nb", card.grid_resolution)
    fig.update_layout(
        title=(
            "GaussianNB decision regions — axis-aligned ellipse contours = "
            f"independence (var_smoothing={ctx.params['var_smoothing']})"
        ),
        template="plotly_white",
    )
    return fig


def _confusion_matrix(ctx):
    pred = ctx.sklearn.predict(ctx.data.X)
    labels = sorted(set(ctx.data.y))
    cm = confusion_matrix(ctx.data.y, pred, labels=labels)

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[str(i) for i in labels],
            y=[str(i) for i in labels],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion matrix — diagonal = right class, off-diagonal = wrong",
        template="plotly_white",
        xaxis_title="predicted class",
        yaxis_title="true class",
    )
    return fig


card = AlgorithmCard(
    id="naive-bayes",
    title="Naive Bayes (Gaussian)",
    family="classification",
    when_to_use="fast probabilistic classifier whose independence assumption you can *see* bend",
    theory=THEORY,
    sources=(),
    sklearn_only=True,
    hypers=(
        Select("var_smoothing", (1e-9, 1e-7, 1e-5, 1e-3), 1e-9, "variance floor"),
    ),
    datasets=("moons", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundary, _confusion_matrix),
    notes=(
        "axis-aligned ellipse decision regions = per-feature Gaussians — the "
        "independence assumption drawn",
        "on `moons` the crescents' features are correlated — the independence "
        "assumption bends the boundary where the tilted contour should be",
        "smoothing only matters at 1e-3: watch regions merge as the variance "
        "floor flattens every peak",
        "the confusion matrix shows *which* class absorbs the crescents' "
        "overlap — accuracy alone hides the asymmetry",
    ),
)
