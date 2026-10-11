"""Card: Decision Tree — classifier and regressor twins, dual-engine.

`algorithm` selects the variant (pattern from the linear-regression card):
- classifier: from-scratch DecisionTreeClassifierScratch (gini/entropy) vs
  sklearn's DecisionTreeClassifier — side-by-side decision regions and
  train/test accuracy-vs-depth curves.
- regressor: from-scratch DecisionTreeRegressorScratch (MSE splits) vs
  sklearn's DecisionTreeRegressor — a predicted-vs-actual fit view and
  R²-vs-depth curves; the classifier's boundary language does not apply, and
  the regression toys route through variant_datasets.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.core.card import AlgorithmCard, Select, Slider
from src.models.tree_models import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
)

THEORY = """## Theory

A decision tree predicts by asking a sequence of yes/no feature questions.
Each split partitions the training rows to reduce **impurity**. Entropy-based
impurity (used by the scratch implementation's `entropy` option):

$$H(y) = -\\sum_c p_c \\log_2 p_c$$

Gini impurity (`gini` option):

$$G(y) = 1 - \\sum_c p_c^2$$

A candidate split on feature $j$ at threshold $t$ is scored by the impurity
drop — **information gain**:

$$IG = H(parent) - \\sum_{branches} \\tfrac{n_b}{n} \\, H(branch_b)$$

For regression the same machinery runs on squared error: a split scores by
the **variance reduction** between parent and weighted children, and the
independence assumption's geometry is absent — leaves hold the *mean* of
their rows, so a depth-1 tree is a piecewise-constant approximation.

The tree greedily picks the split with the highest gain, recursively, until a
stopping criterion fires: `max_depth` reached, a node has fewer rows than
`min_samples_split`, or a node is pure. With no depth limit a tree can keep
splitting until every leaf is a single training row — memorizing the training
set and generalizing badly. Regularize with `max_depth` and
`min_samples_split`; criterion choice (gini vs entropy) rarely matters much.
"""


def fit(data, params, engine):
    variant = params["algorithm"]
    if variant == "classifier":
        if engine == "scratch":
            return DecisionTreeClassifierScratch(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                criterion=params["criterion"],
            ).fit(data.X, data.y)
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            criterion=params["criterion"],
            random_state=0,
        ).fit(data.X, data.y)
    if engine == "scratch":
        return DecisionTreeRegressorScratch(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
        ).fit(data.X, data.y)
    return DecisionTreeRegressor(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=0,
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
    # regressor twin: predicted-vs-actual on one shared axis — the diagonal
    # is the perfect fit for any feature count (linear-regression card's
    # ≥2-feature pattern)
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
            f"Predicted vs actual at depth {ctx.params['max_depth']} — "
            "the diagonal is perfect; a tree's piecewise staircase shows here"
        ),
        template="plotly_white",
        xaxis_title="predicted",
        yaxis_title="actual",
    )
    return fig


def _accuracy_vs_depth(ctx):
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    depths = list(range(1, 21))
    is_classifier = ctx.params["algorithm"] == "classifier"

    def score(y_true, y_pred):
        if is_classifier:
            return float(accuracy_score(y_true, y_pred))
        return float(r2_score(y_true, y_pred))

    series = {"scratch": {}, "sklearn": {}}
    for depth in depths:
        params = dict(ctx.params, max_depth=depth)
        for name in series:
            fitted = fit(sub, params, name)
            series[name][depth] = (
                score(ytr, fitted.predict(Xtr)),
                score(yte, fitted.predict(Xte)),
            )
    fig = go.Figure()
    for name, color in [("scratch", "#EF553B"), ("sklearn", "#636EFA")]:
        fig.add_trace(
            go.Scatter(
                x=depths,
                y=[series[name][d][0] for d in depths],
                name=f"{name} train",
                line=dict(color=color, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=depths,
                y=[series[name][d][1] for d in depths],
                name=f"{name} test",
                line=dict(color=color, dash="dot", width=2),
            )
        )
    metric_name = "accuracy" if is_classifier else "R²"
    fig.update_layout(
        title=f"{metric_name} vs max_depth — train and test split apart as the tree memorizes",
        template="plotly_white",
        xaxis_title="max_depth",
        yaxis_title=metric_name,
    )
    return fig


card = AlgorithmCard(
    id="decision-tree",
    title="Decision Tree",
    family="classification",
    when_to_use="non-linear rules with human-readable logic — watch it overfit",
    theory=THEORY,
    sources=(
        ("src/models/tree_models.py", ("DecisionTreeClassifierScratch", "DecisionTreeRegressorScratch")),
    ),
    hypers=(
        Select("algorithm", ("classifier", "regressor"), "classifier",
               "which twin of the tree family; regression toys route to the regressor"),
        Slider("max_depth", 1, 20, 1, 3, "maximum tree depth"),
        Slider("min_samples_split", 2, 20, 1, 2, "minimum rows to consider a split"),
        Select("criterion", ("gini", "entropy"), "gini", "split quality measure (classifier only)"),
    ),
    datasets=("moons", "circles", "lin_separable"),
    variant_datasets=(
        ("classifier", ("moons", "circles", "lin_separable")),
        ("regressor", ("lin_clean_1f", "lin_noisy_1f", "lin_outliers_1f", "lin_2f")),
    ),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_depth),
    notes=(
        "on `moons`, drag max_depth 3 → 20: train accuracy → 1.0 while test "
        "collapses — memorization",
        "min_samples_split fights overfitting from the other side — raise it "
        "with a deep tree",
        "switch algorithm to regressor: the same depth slider now shapes a "
        "staircase over continuous y — depth 1 draws a first-segment plateau",
        "regressor with max_depth=1 on `lin 2f`: two rectangles track the "
        "plane where variance drops fastest",
        "compare scratch vs sklearn regions: small differences near the moons' "
        "overlap reveal tie-breaking differences",
    ),
)
