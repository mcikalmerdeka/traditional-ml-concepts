"""Card: Linear Regression (with Ridge and Lasso variants).

Dual-engine: from-scratch classes from src/models/linear_models.py vs sklearn.
The alpha slider only affects ridge/lasso; the variant select chooses which
model both engines fit.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

from app.core.card import AlgorithmCard, Select, Slider, Toggle
from src.models.linear_models import (
    LassoRegressionScratch,
    LinearRegressionScratch,
    RidgeRegressionScratch,
)

THEORY = """## Theory

Linear regression fits a linear function through the data:

$$\\hat{y} = Xw + b$$

Training minimizes the mean squared error:

$$\\min_w \\; \\tfrac{1}{n} \\|y - Xw - b\\|_2^2$$

The closed-form solution (the **normal equation**) sets the gradient to zero:

$$w = (X^\\top X)^{-1} X^\\top y$$

Two regularized variants trade bias for variance by penalizing large weights.
**Ridge** (L2) adds a squared penalty and has a closed form:

$$w = (X^\\top X + \\alpha I)^{-1} X^\\top y$$

**Lasso** (L1) adds an absolute-value penalty and is solved by coordinate
descent:

$$\\min_w \\; \\tfrac{1}{n} \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_1$$

The L1 geometry is what makes lasso push coefficients exactly to zero —
feature selection for free. Ridge shrinks coefficients smoothly but keeps
them nonzero. Plain linear regression is the special case $\\alpha = 0$:
fast, interpretable, but sensitive to outliers and multicollinearity.
"""


def fit(data, params, engine):
    variant = params["algorithm"]
    if engine == "scratch":
        cls = {
            "linear": LinearRegressionScratch,
            "ridge": RidgeRegressionScratch,
            "lasso": LassoRegressionScratch,
        }[variant]
        kwargs = {"fit_intercept": params["fit_intercept"]}
        if variant != "linear":
            kwargs["alpha"] = params["alpha"]
        return cls(**kwargs).fit(data.X, data.y)
    # sklearn side — closed-form estimators are deterministic; only Lasso takes
    # random_state (used by its coordinate-selection path)
    sk_cls = {"linear": LinearRegression, "ridge": Ridge, "lasso": Lasso}[variant]
    kwargs = {"fit_intercept": params["fit_intercept"]}
    if variant != "linear":
        kwargs["alpha"] = params["alpha"]
    if variant == "lasso":
        kwargs["random_state"] = 0
    return sk_cls(**kwargs).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [
        ("R²", float(r2_score(data.y, pred))),
        ("MSE", float(mean_squared_error(data.y, pred))),
    ]


def _overlay(ctx):
    from app.components.scatter import labeled_scatter

    if ctx.data.X.shape[1] >= 2:
        # ≥2 features: a line over an x₀-sorted scatter zigzags (predictions
        # depend on both features) — show predicted-vs-actual instead, the
        # fit view that works for any feature count
        fig = go.Figure()
        for name, fit_obj, color in [
            ("scratch", ctx.scratch, "#EF553B"),
            ("sklearn", ctx.sklearn, "#636EFA"),
        ]:
            if fit_obj is None:
                continue
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
        lo = float(min(ctx.data.y.min(), fig.data[0].x.min()))
        hi = float(max(ctx.data.y.max(), fig.data[0].x.max()))
        fig.add_shape(
            type="line", x0=lo, y0=lo, x1=hi, y1=hi,
            line=dict(color="gray", dash="dot", width=1),
        )
        fig.update_layout(
            title="Predicted vs actual — the diagonal is the perfect fit; off-diagonal is error",
            template="plotly_white",
            xaxis_title="predicted",
            yaxis_title="actual",
        )
        return fig

    fig = labeled_scatter(ctx.data.X, ctx.data.y, "Fit overlay — scratch vs sklearn")
    order = np.argsort(ctx.data.X[:, 0])
    X_sorted = ctx.data.X[order]
    for name, fit_obj, style in [
        ("scratch", ctx.scratch, dict(color="#EF553B", dash="dash", width=3)),
        ("sklearn", ctx.sklearn, dict(color="#636EFA", width=3)),
    ]:
        if fit_obj is None:
            continue
        pred = fit_obj.predict(X_sorted)
        fig.add_trace(
            go.Scatter(
                x=X_sorted[:, 0],
                y=pred,
                mode="lines",
                name=name,
                line=style,
            )
        )
    return fig


def _residuals(ctx):
    fig = go.Figure()
    for name, fit_obj, color in [
        ("scratch", ctx.scratch, "#EF553B"),
        ("sklearn", ctx.sklearn, "#636EFA"),
    ]:
        if fit_obj is None:
            continue
        pred = fit_obj.predict(ctx.data.X)
        fig.add_trace(
            go.Scatter(
                x=pred,
                y=ctx.data.y - pred,
                mode="markers",
                name=name,
                marker=dict(size=6, color=color),
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Residuals vs predictions — structure here means the model is missing something",
        template="plotly_white",
        xaxis_title="predicted",
        yaxis_title="residual",
    )
    return fig


card = AlgorithmCard(
    id="linear-regression",
    title="Linear Regression",
    family="regression",
    when_to_use="a roughly linear relationship, or a fast interpretable baseline",
    theory=THEORY,
    sources=(("src/models/linear_models.py",
              ("LinearRegressionScratch", "RidgeRegressionScratch", "LassoRegressionScratch")),),
    hypers=(
        Select("algorithm", ("linear", "ridge", "lasso"), "linear",
               "variant; alpha applies only to ridge/lasso"),
        Slider("alpha", 0.0, 5.0, 0.2, 1.0, "regularization strength (ridge/lasso only)"),
        Toggle("fit_intercept", True, "include bias term b"),
    ),
    datasets=("lin_clean_1f", "lin_noisy_1f", "lin_outliers_1f", "lin_2f"),
    fit=fit,
    metrics=metrics,
    visualizations=(_overlay, _residuals),
    notes=(
        "switch `algorithm` to ridge or lasso and drag α on `lin outliers` — "
        "watch coefficients shrink as α grows",
        "lasso zeroes coefficients entirely (sparsity); ridge only shrinks them",
        "on clean data the scratch and sklearn lines should overlap almost exactly",
        "fit_intercept off forces the line through the origin — see the residuals tilt",
    ),
)
