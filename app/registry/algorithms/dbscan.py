"""Card: DBSCAN.

Dual-engine: from-scratch DBSCANScratch (grid-free density reachability) vs
sklearn's DBSCAN — both label rows, mark -1 noise and expose labels_, so the
playground shows per-engine scatter panels and a per-engine clusters-vs-eps
sweep.

DBSCAN needs no k: eps (neighborhood radius) and min_samples (core-point
threshold) ARE the model. Its slider bounds deliberately include the
degenerate end — eps = 0.1 on the rings turns every point into noise, zero
clusters — and this page survives it (Review Focus #1).
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from app.core.card import AlgorithmCard, Slider
from src.models.clustering_models import DBSCANScratch

THEORY = """## Theory

DBSCAN defines clusters by **density**, not distance-to-centroid. A point is
a **core point** if its neighborhood contains enough friends:

$$|N_\\epsilon(x)| \\ge \\text{min\\_samples}$$

Start from a core point, take the core points inside its radius, then theirs:
**density-reachability** — clusters are the connected components of the core
graph. Rows that belong to no cluster's core chain are labeled **noise**:

$$y_i = -1$$

That noise label is DBSCAN's honest answer — K-Means cannot say "this point
is nothing", it must invent a centroid for it. The two sliders ARE the whole
model: there is no k to declare. eps too small → everything is noise and the
metric row shows zero clusters (try it — the page survives, that is the
pinned degenerate case); eps too large → one blob eats every ring (on
`kmeans rings`, eps ≈ 0.5 separates the two radii — exactly the failure
K-Means cannot fix).
"""


def fit(data, params, engine):
    if engine == "scratch":
        return DBSCANScratch(eps=params["eps"], min_samples=params["min_samples"]).fit(
            data.X
        )
    # DBSCAN has no random_state parameter — pass nothing extra
    return DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit(data.X)


def metrics(fitted, data):
    # every metric is guarded: the slider bounds reach legitimate failure
    # states (all-noise, single cluster) and the page must degrade gracefully
    labels = fitted.raw.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    values = [
        ("Clusters", float(n_clusters)),
        ("Noise", float((labels == -1).mean())),
    ]
    if n_clusters >= 2:
        # silhouette on non-noise rows only: scattered noise must not pose
        # as a pseudo "-1 cluster" (basis matches the sibling clustering
        # pages). mask.sum() > n_clusters keeps sklearn's 2 <= n_labels < n_samples
        # precondition — the all-singleton-clusters edge would otherwise raise.
        mask = labels != -1
        if mask.sum() > n_clusters:
            values.append(("Silhouette", float(silhouette_score(data.X[mask], labels[mask]))))
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
            if k == -1:
                fig.add_trace(
                    go.Scatter(
                        x=ctx.data.X[mask, 0],
                        y=ctx.data.X[mask, 1],
                        xaxis=slot,
                        yaxis="y",
                        mode="markers",
                        marker=dict(size=6, color="gray", symbol="x"),
                        name=f"{name} noise",
                    )
                )
            else:
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
            f"Density clusters — eps={ctx.params['eps']}, "
            f"min_samples={ctx.params['min_samples']} (gray × = noise) — "
            "scratch (left) vs sklearn (right)"
        ),
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
    )
    return fig


def _clusters_vs_eps(ctx):
    # sweep eps at the current min_samples; the marker rides the count line
    eps_values = list(np.arange(0.1, 3.05, 0.05))
    fig = go.Figure()
    counts_by_name = {}
    for name, color, fitted_cls in [
        ("scratch", "#EF553B", DBSCANScratch),
        ("sklearn", "#636EFA", DBSCAN),
    ]:
        counts = []
        for eps in eps_values:
            raw = fitted_cls(min_samples=ctx.params["min_samples"], eps=eps).fit(ctx.data.X)
            labels = raw.labels_
            counts.append(float(len(set(labels)) - (1 if -1 in labels else 0)))
        counts_by_name[name] = counts
        fig.add_trace(
            go.Scatter(
                x=eps_values,
                y=counts,
                mode="lines+markers",
                name=f"{name} clusters",
                line=dict(color=color, width=2),
            )
        )
    current = ctx.params["eps"]

    fig.add_trace(
        go.Scatter(
            x=[current],
            y=[counts_by_name["sklearn"][int(round((current - 0.1) / 0.05))]],
            mode="markers",
            marker=dict(size=14, color="red"),
            name="current eps",
        )
    )
    fig.update_layout(
        title="Clusters vs eps — 0 at the noise end, 1 as blobs merge; rings separate near 0.5",
        template="plotly_white",
        xaxis_title="eps (neighborhood radius)",
        yaxis_title="cluster count",
    )
    return fig


card = AlgorithmCard(
    id="dbscan",
    title="DBSCAN",
    family="clustering",
    when_to_use="clusters of any shape with real noise handling — no k, only density",
    theory=THEORY,
    sources=(("src/models/clustering_models.py", ("DBSCANScratch",)),),
    hypers=(
        Slider("eps", 0.1, 3.0, 0.05, 0.5, "neighborhood radius"),
        Slider("min_samples", 2, 20, 1, 5, "core-point threshold"),
    ),
    datasets=("kmeans_rings", "var_blobs"),
    fit=fit,
    metrics=metrics,
    visualizations=(_scatter_by_label, _clusters_vs_eps),
    notes=(
        "on `kmeans_rings`, eps ≈ 0.5 finds both rings — the failure K-Means "
        "cannot fix (compare the K-Means page)",
        "eps = 0.1: everything is noise, zero clusters — the page must "
        "survive it (it's the pinned degenerate case)",
        "eps = 3.0: one blob eats the rings — watch the count line fall to 1",
        "raise min_samples on `var_blobs` and the diffuse blob dissolves "
        "into noise first — density is a threshold, not a count",
    ),
)
