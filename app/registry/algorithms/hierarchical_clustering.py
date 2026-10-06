"""Card: Hierarchical Clustering (sklearn-only — no from-scratch implementation exists yet).

When AgglomerativeClusteringScratch lands in src/models/clustering_models.py,
this card upgrades to dual-engine by adding a sources entry and a scratch
branch in fit.

Agglomerative merging: start with every point as its own cluster and merge
the two closest clusters repeatedly. The linkage rule defines what "closest
cluster pair" means, and cutting the merge hierarchy at height k is the
cluster count. No `predict` — labels exist only for fitted rows.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from app.core.card import AlgorithmCard, Select, Slider

THEORY = """## Theory

Agglomerative clustering starts from n singletons and repeatedly merges the
closest pair of clusters until one (or `n_clusters`) remain. Every decision
is governed by the **linkage rule** — the distance between two clusters:

$$d(A, B)$$ per linkage — the inter-cluster distance the merger minimizes.

Four linkages, four philosophies:

- **ward** — merge the pair that grows total within-cluster variance least;
  convex, blob-chasing, the closest cousin of K-Means
- **average** — distance between the *average* pair across the two clusters
- **complete** — distance between the two **farthest** members; resists
  long chains, keeps clusters compact
- **single** — distance between the two **closest** members; the classic
  chaining rule: a sparse bridge of points links two dense regions into one
  snake

Two things make this model unusual. It has **no `predict`**: agglomerative
clustering is *transductive* — labels exist only for the rows that were
fitted, never for new rows. And it has **no k to optimize during the fit**:
the merge hierarchy is built once; *choosing k = cutting that hierarchy* at
height k. The silhouette-vs-k plot is your cut-point advisor: its peak marks
the k the hierarchy itself prefers.
"""


def fit(data, params, engine):
    # AgglomerativeClustering has no random_state — pass nothing extra
    return AgglomerativeClustering(
        n_clusters=params["n_clusters"],
        linkage=params["linkage"],
    ).fit(data.X)


def metrics(fitted, data):
    labels = fitted.raw.labels_
    values = [("Clusters", float(len(set(labels))))]
    if len(set(labels)) >= 2:
        values.append(("Silhouette", float(silhouette_score(data.X, labels))))
    return values


def _scatter_by_label(ctx):
    fig = go.Figure()
    labels = ctx.sklearn.raw.labels_
    for k in np.unique(labels):
        mask = labels == k
        fig.add_trace(
            go.Scatter(
                x=ctx.data.X[mask, 0],
                y=ctx.data.X[mask, 1],
                mode="markers",
                marker=dict(size=6),
                name=f"cluster {k}",
            )
        )
    fig.update_layout(
        title=(
            f"Clusters at k={ctx.params['n_clusters']}, "
            f"linkage={ctx.params['linkage']} — no centroids, no predict"
        ),
        template="plotly_white",
        xaxis_title="x₀",
        yaxis_title="x₁",
    )
    return fig


def _silhouette_vs_k(ctx):
    # sweep covers the n_clusters slider's full range (2..10) so the
    # "current k" marker never vanishes inside the slider's bounds
    # (cleanup minor #6)
    ks = list(range(2, 11))
    scores = []
    for k in ks:
        # module-level fit returns the RAW model (run() adds the Fitted wrap)
        raw = fit(ctx.data, dict(ctx.params, n_clusters=k), "sklearn")
        labels = raw.labels_
        if len(set(labels)) >= 2:
            scores.append(float(silhouette_score(ctx.data.X, labels)))
        else:
            scores.append(float("nan"))
    current = ctx.params["n_clusters"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=scores,
            mode="lines+markers",
            name="silhouette",
            line=dict(color="#636EFA", width=2),
        )
    )
    if 2 <= current <= 10:
        fig.add_trace(
            go.Scatter(
                x=[current],
                y=[scores[current - 2]],
                mode="markers",
                marker=dict(size=14, color="red"),
                name="current k",
            )
        )
    fig.update_layout(
        title="Silhouette vs k — the peak marks the k the merge hierarchy wants",
        template="plotly_white",
        xaxis_title="k (cut of the merge hierarchy)",
        yaxis_title="silhouette",
    )
    return fig


card = AlgorithmCard(
    id="hierarchical-clustering",
    title="Hierarchical Clustering",
    family="clustering",
    when_to_use="a full merge hierarchy you can cut at any k — linkage choice is the lesson",
    theory=THEORY,
    sources=(),
    sklearn_only=True,
    hypers=(
        Select("linkage", ("ward", "average", "complete", "single"), "ward"),
        Slider("n_clusters", 2, 10, 1, 4),
    ),
    datasets=("kmeans_4blobs", "var_blobs"),
    fit=fit,
    metrics=metrics,
    visualizations=(_scatter_by_label, _silhouette_vs_k),
    notes=(
        "single linkage chains across `var_blobs`' sparse regions — ward "
        "refuses to; switch linkages and watch k=2 swallow the diffuse blob",
        "ward ≈ K-Means on convex blobs: compare pages — near-identical "
        "partitions, different machinery",
        "the silhouette peak marks the k the merge hierarchy wants — on "
        "`var_blobs` try ward: k=3 is the truth",
        "drag k and watch the marker ride the silhouette line: no centroid "
        "moves here, only the hierarchy cut does",
    ),
)
