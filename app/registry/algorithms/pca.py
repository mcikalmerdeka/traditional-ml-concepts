"""Card: PCA (sklearn-only — no from-scratch implementation exists yet).

When PCAScratch lands in src/models/dimensionality_reduction.py, this card
upgrades to dual-engine by adding a sources entry and a scratch branch in fit.

Family "dimensionality-reduction": engines.run() wraps the fit with
kind="transform", so the projection viz flows through Fitted.transform.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from app.core.card import AlgorithmCard, Slider

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
    proj = ctx.sklearn.transform(ctx.data.X)  # kind="transform" — never predict
    k = proj.shape[1]
    y = proj[:, 1] if k > 1 else np.zeros(len(proj))
    fig = go.Figure(
        go.Scatter(
            x=proj[:, 0],
            y=y,
            mode="markers",
            marker=dict(size=6, color="#636EFA"),
            name="rows",
        )
    )
    title = "Projection (PC1 × PC2)" if k > 1 else "Projection (PC1 — k=1 collapses onto one axis)"
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="PC1",
        yaxis_title="PC2" if k > 1 else "—",
    )
    return fig


def _explained_variance_ratio(ctx):
    # bar heights never move: ratios are dataset properties; the slider
    # (marker) only chooses the cut point of the cumulative line
    raw = fit(ctx.data, dict(ctx.params, n_components=len(ctx.data.X[0])), "sklearn")
    ratios = raw.explained_variance_ratio_
    ks = list(range(1, len(ratios) + 1))
    cumulative = list(np.cumsum(ratios))
    current = ctx.params["n_components"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ks,
            y=ratios,
            name="per-component ratio",
            marker_color="#636EFA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=cumulative,
            mode="lines+markers",
            name="cumulative",
            line=dict(color="#EF553B", width=2),
        )
    )
    if 1 <= current <= len(ks):
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
    sources=(),
    sklearn_only=True,
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
        "the bar heights never move — ratios are dataset properties, the "
        "slider only moves the marker",
        "compression = few components with most of the cumulative line: read "
        "the red marker, not the bar",
    ),
)
