"""Card: K-Means.

Dual-engine: from-scratch KMeansScratch (k-means++ seeding + Lloyd restarts)
vs sklearn's KMeans — both engines expose labels_/cluster_centers_/inertia_,
so the playground shows per-engine cluster panels and a per-engine elbow.

On tiny blobs the two engines converge to near-identical partitions; the
visible divergence happens at n_init=1, where initialization dominates.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.core.card import AlgorithmCard, Select, Slider
from src.models.clustering_models import KMeansScratch

THEORY = """## Theory

K-Means partitions the rows into k clusters by minimizing **inertia** — the
total squared distance of each point to its assigned cluster's centroid:

$$\\min_{C} \\; \\sum_{i=1}^{n} \\|x_i - \\mu_{c_i}\\|^2$$

Lloyd's algorithm alternates two steps until convergence: **assign** every
point to its nearest centroid, then **update** each centroid to its cluster's
mean:

$$\\mu_j \\leftarrow \\tfrac{1}{|C_j|} \\sum_{x_i \\in C_j} x_i$$

Inertia is not convex, so initialization matters: sklearn restarts the whole
algorithm `n_init` times (random_state pinned) and keeps the best. K-Means
only finds **convex, blob-like** clusters of roughly similar size — on
concentric rings it slices by sector instead of by radius. Choosing k: the
**elbow** — plot inertia against k and look for the bend where extra clusters
stop buying much reduction.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return KMeansScratch(
            n_clusters=params["n_clusters"],
            n_init=params["n_init"],
            max_iter=params["max_iter"],
            random_state=0,
        ).fit(data.X)
    return KMeans(
        n_clusters=params["n_clusters"],
        n_init=params["n_init"],
        max_iter=params["max_iter"],
        random_state=0,
    ).fit(data.X)


def metrics(fitted, data):
    values = [("Inertia", float(fitted.raw.inertia_))]
    n_labels = len(set(fitted.raw.labels_))
    if n_labels > 1:
        values.append(("Silhouette", float(silhouette_score(data.X, fitted.raw.labels_))))
    return values


def _clusters(ctx):
    fig = go.Figure()
    for slot, name, fitted_obj, template in [
        ("x", "scratch", ctx.scratch, "scratch (left)"),
        ("x2", "sklearn", ctx.sklearn, "sklearn (right)"),
    ]:
        labels = fitted_obj.raw.labels_
        centers = fitted_obj.raw.cluster_centers_
        for k in np.unique(labels):
            mask = labels == k
            fig.add_trace(
                go.Scatter(
                    x=ctx.data.X[mask, 0],
                    y=ctx.data.X[mask, 1],
                    xaxis=slot,
                    yaxis="y",
                    mode="markers",
                    marker=dict(size=6),
                    name=f"{name} cluster {k}",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                xaxis=slot,
                yaxis="y",
                mode="markers",
                marker=dict(symbol="x", size=12, color="red", line_width=2),
                name=f"{name} centroids",
            )
        )
    fig.update_layout(
        title=f"Clusters & centroids (k={params_k(ctx)}) — {template}",
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
    )
    return fig


def params_k(ctx):
    return ctx.params["n_clusters"]


def _elbow(ctx):
    ks = list(range(1, 9))
    fig = go.Figure()
    for name, color, fitted_cls in [
        ("scratch", "#EF553B", KMeansScratch),
        ("sklearn", "#636EFA", KMeans),
    ]:
        inertias = [
            fitted_cls(
                n_clusters=k, n_init=10, max_iter=300, random_state=0
            )
            .fit(ctx.data.X)
            .inertia_
            for k in ks
        ]
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=inertias,
                mode="lines+markers",
                name=f"{name} inertia",
                line=dict(color=color, width=2),
            )
        )
    current = params_k(ctx)
    if 1 <= current <= 8:
        fig.add_trace(
            go.Scatter(
                x=[current],
                y=[inertias[current - 1]],
                mode="markers",
                marker=dict(size=14, color="red"),
                name="current k",
            )
        )
    fig.update_layout(
        title="Elbow — inertia vs k; the bend marks the natural cluster count",
        template="plotly_white",
        xaxis_title="k",
        yaxis_title="inertia",
    )
    return fig


card = AlgorithmCard(
    id="kmeans",
    title="K-Means",
    family="clustering",
    when_to_use="finding blob-like groups without labels — needs convex clusters",
    theory=THEORY,
    sources=(("src/models/clustering_models.py", ("KMeansScratch",)),),
    row_cap=None,
    hypers=(
        Slider("n_clusters", 1, 10, 1, 4, "number of clusters to find"),
        Select("n_init", (1, 10), 10, "restarts; more = stabler centroids"),
        Slider("max_iter", 10, 500, 10, 300, "Lloyd iterations per restart"),
    ),
    datasets=("kmeans_4blobs", "kmeans_rings"),
    fit=fit,
    metrics=metrics,
    visualizations=(_clusters, _elbow),
    notes=(
        "on `kmeans rings` with k=2, clusters split by sector — K-Means "
        "assumes convex blobs, not radial structure",
        "drop `n_init` to 1 to see initialization sensitivity: centroids land "
        "differently across reruns — and scratch vs sklearn can disagree",
        "watch the red dot slide along the elbow curve as you change k",
    ),
)
