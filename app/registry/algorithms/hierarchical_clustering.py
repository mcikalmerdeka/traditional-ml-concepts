"""Card: Hierarchical Clustering.

Dual-engine: from-scratch AgglomerativeClusteringScratch (bottom-up merging
with ward/complete/average/single Lance-Williams updates) vs sklearn's
AgglomerativeClustering — both expose labels_ only, so the playground shows
per-engine scatter panels and a per-engine silhouette-vs-k sweep.

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
from src.models.clustering_models import AgglomerativeClusteringScratch

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
    if engine == "scratch":
        return AgglomerativeClusteringScratch(
            n_clusters=params["n_clusters"],
            linkage=params["linkage"],
        ).fit(data.X)
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
    for slot, name, fitted_obj in [
        ("x", "scratch", ctx.scratch),
        ("x2", "sklearn", ctx.sklearn),
    ]:
        labels = fitted_obj.raw.labels_
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
    fig.update_layout(
        title=(
            f"Clusters at k={ctx.params['n_clusters']}, "
            f"linkage={ctx.params['linkage']} — no centroids, no predict — "
            "scratch (left) vs sklearn (right)"
        ),
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
    )
    return fig


def _silhouette_vs_k(ctx):
    # sweep covers the n_clusters slider's full range (2..10) so the
    # "current k" marker never vanishes inside the slider's bounds
    # (cleanup minor #6)
    ks = list(range(2, 11))
    fig = go.Figure()
    for name, color, fitted_cls in [
        ("scratch", "#EF553B", AgglomerativeClusteringScratch),
        ("sklearn", "#636EFA", AgglomerativeClustering),
    ]:
        scores = []
        for k in ks:
            raw = fitted_cls(
                n_clusters=k, linkage=ctx.params["linkage"]
            ).fit(ctx.data.X)
            labels = raw.labels_
            if len(set(labels)) >= 2:
                scores.append(float(silhouette_score(ctx.data.X, labels)))
            else:
                scores.append(float("nan"))
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=scores,
                mode="lines+markers",
                name=f"{name} silhouette",
                line=dict(color=color, width=2),
            )
        )
    current = ctx.params["n_clusters"]

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
    sources=(("src/models/clustering_models.py", ("AgglomerativeClusteringScratch",)),),
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
        "scratch's ward uses the Lance-Williams variance update while "
        "sklearn's invokes its own — partitions agree on blobs, watch them "
        "diverge on `var_blobs`' tail points",
    ),
)
