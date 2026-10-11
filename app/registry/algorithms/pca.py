"""Card: PCA — dual-engine (PCAScratch vs sklearn PCA).

Family "dimensionality-reduction": engines.run() wraps the fit with
kind="transform", so the projection viz flows through Fitted.transform. The
scratch eigendecomposition resolves the same principal directions up to
sign, so projections agree up to flips — the explained-variance bars agree
exactly.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from app.core.card import AlgorithmCard, Slider
from src.models.dimensionality_reduction import PCAScratch

THEORY = """## Theory

Principal component analysis compresses features by projecting onto the
directions along which the data actually spreads. Those directions come from
the covariance eigendecomposition:

$$\\Sigma v = \\lambda v$$

with each **PC's variance** its eigenvalue:

$$\\lambda_i$$

and the **explained-variance ratio** its share of the total:

$$\\lambda_i / \\sum_j \\lambda_j$$

The components are *ordered by variance*: $\\lambda_1 \\ge \\lambda_2 \\ge \\dots$,
so PCA's model is fitted once and in full — the slider only chooses how many
to **keep**. That is what compression means here: keep few components while
their ratio still covers most of the total, and each row travels as 2 numbers
instead of 4. On `pca correlated 4f` the first two components should capture
visibly more than half the variance — a real compression, not a truncation.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return PCAScratch(n_components=params["n_components"]).fit(data.X)
    return PCA(n_components=params["n_components"], random_state=0).fit(data.X)


def metrics(fitted, data):
    # the sum FOR THE KEPT components is the lesson: k=2 on this dataset is
    # visibly less than 4.0 but more than half
    return [
        (
            "Explained variance",
            float(fitted.raw.explained_variance_ratio_.sum()),
        )
    ]


def _projection(ctx):
    fig = go.Figure()
    for slot, name, fitted_obj, color in [
        ("x", "scratch", ctx.scratch, "#EF553B"),
        ("x2", "sklearn", ctx.sklearn, "#636EFA"),
    ]:
        proj = fitted_obj.transform(ctx.data.X)  # kind="transform" — never predict
        k = proj.shape[1]
        y = proj[:, 1] if k > 1 else np.zeros(len(proj))
        fig.add_trace(
            go.Scatter(
                x=proj[:, 0],
                y=y,
                xaxis=slot,
                yaxis="y",
                mode="markers",
                marker=dict(size=6, color=color),
                name=name,
            )
        )
    fig.update_layout(
        title="Projection (PC1 × PC2) — scratch (left) vs sklearn (right); "
              "sign flips of an eigenvector are free",
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
    )
    return fig


def _explained_variance_ratio(ctx):
    # bar heights never move: ratios are dataset properties; the slider
    # (marker) only chooses the cut point of the cumulative line
    fig = go.Figure()
    for name, color, fitted_cls in [
        ("scratch", "#EF553B", PCAScratch),
        ("sklearn", "#636EFA", PCA),
    ]:
        raw = fitted_cls(n_components=ctx.data.X.shape[1]).fit(ctx.data.X)
        ratios = raw.explained_variance_ratio_
        ks = list(range(1, len(ratios) + 1))
        cumulative = list(np.cumsum(ratios))
        fig.add_trace(
            go.Bar(
                x=ks,
                y=ratios,
                name=f"{name} per-component ratio",
                marker_color=color,
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=cumulative,
                mode="lines+markers",
                name=f"{name} cumulative",
                line=dict(color=color, width=2),
            )
        )
    current = ctx.params["n_components"]
    if 1 <= current <= ctx.data.X.shape[1]:
        fig.add_trace(
            go.Scatter(
                x=[current],
                y=[cumulative[current - 1]],
                mode="markers",
                marker=dict(size=14, color="red"),
                name="current k",
            )
        )
    fig.update_layout(
        title="Explained variance ratio — keep the few components that cover most of it",
        template="plotly_white",
        xaxis_title="component",
        yaxis_title="variance ratio",
    )
    return fig


card = AlgorithmCard(
    id="pca",
    title="PCA",
    family="dimensionality-reduction",
    when_to_use="compressing correlated features onto their spread directions — explained variance is the lesson",
    theory=THEORY,
    sources=(("src/models/dimensionality_reduction.py", ("PCAScratch",)),),
    hypers=(
        Slider("n_components", 1, 4, 1, 2),
    ),
    datasets=("pca_correlated_4f",),
    fit=fit,
    metrics=metrics,
    visualizations=(_projection, _explained_variance_ratio),
    notes=(
        "k=2 keeps most of four features' information — that is compression",
        "drag k to 1: the projection collapses onto one axis",
        "the two engines' projections can mirror each other in sign — an "
        "eigenvector's sign is arbitrary; explained variance never disagrees",
        "the bar heights never move — ratios are dataset properties, the "
        "slider only moves the marker",
    ),
)
