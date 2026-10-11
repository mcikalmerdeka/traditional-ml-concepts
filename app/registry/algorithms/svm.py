"""Card: Support Vector Machine.

Dual-engine: from-scratch SVMClassifierScratch (simplified-SMO kernel dual)
vs sklearn's SVC. The playground shows per-engine decision boundaries and
the training/test accuracy-vs-C curves that expose the margin tradeoff.

The playground makes two tradeoffs visible: C (margin width vs violations)
and gamma (how island-like the RBF regions become).
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from app.core.card import AlgorithmCard, Select, Slider
from src.models.svm_models import SVMClassifierScratch

THEORY = r"""## Theory

An SVM finds the separating boundary with the **maximum margin** — the
half-width `2/‖w‖` between the two classes' supporting planes:

$$\max_{w,b} \;\; \frac{2}{\|w\|} \;\; \text{s.t.}\;\; y_i(w^\top x_i + b) \ge 1$$

Rows that end up on the wrong side (or inside the margin) pay a penalty —
the **hinge loss**:

$$L = \max(0,\; 1 - y_i (w^\top x_i + b))$$

That relaxation is controlled by **C**. Small C → a wide margin, more train
errors tolerated (violations are cheap); large C → a narrow margin, every
training point is nearly sacred.

Non-linearity comes from kernels: the **RBF kernel** lets the boundary close
around pockets of data:

$$K(x, x') = e^{-\gamma \|x - x'\|^2}$$

**gamma** is the kernel width. Small γ → smooth, island-free regions; large γ
→ each training point spawns its own island of class 1 surrounded by the
other class — memorization in feature-space clothes. The `linear` kernel has
no bend at all: on `circles` it cannot separate a ring from its center, no
matter how large C is — geometry beats capacity.
"""


def fit(data, params, engine):
    if engine == "scratch":
        return SVMClassifierScratch(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            random_state=0,
        ).fit(data.X, data.y)
    return SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        random_state=0,
    ).fit(data.X, data.y)


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
            f"SVM decision regions — kernel={ctx.params['kernel']}, "
            f"C={ctx.params['C']}, γ={ctx.params['gamma']} — "
            "scratch (left) vs sklearn (right)"
        ),
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _accuracy_vs_c(ctx):
    # log-spaced C sweep 0.01 → 10: wide-margin underfit on the left, tight
    # margin overfit on the right — train and test drift apart
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    cs = list(np.geomspace(0.01, 10.0, 20))
    curves = {name: {"train": [], "test": []} for name in ("scratch", "sklearn")}
    for c in cs:
        for name in curves:
            fitted = fit(sub, dict(ctx.params, C=c), name)
            curves[name]["train"].append(float(accuracy_score(ytr, fitted.predict(Xtr))))
            curves[name]["test"].append(float(accuracy_score(yte, fitted.predict(Xte))))

    fig = go.Figure()
    for name, color in [("scratch", "#EF553B"), ("sklearn", "#636EFA")]:
        fig.add_trace(
            go.Scatter(
                x=cs, y=curves[name]["train"], mode="lines+markers",
                name=f"{name} train", line=dict(color=color, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cs, y=curves[name]["test"], mode="lines+markers",
                name=f"{name} test", line=dict(color=color, dash="dot", width=2),
            )
        )
    fig.update_xaxes(type="log", title="C")
    floor = min(min(curves[name]["test"]) for name in curves)
    fig.update_yaxes(title="accuracy", range=[min(0.5, floor - 0.05), 1.02])
    fig.update_layout(
        title=(
            "Accuracy vs C — small C: wide margin underfits; large C: "
            "train climbs toward every point"
        ),
        template="plotly_white",
    )
    return fig


card = AlgorithmCard(
    id="svm",
    title="Support Vector Machine",
    family="classification",
    when_to_use="maximum-margin boundaries up to kernel-shaped geometry — C vs gamma is the lesson",
    theory=THEORY,
    sources=(("src/models/svm_models.py", ("SVMClassifierScratch",)),),
    hypers=(
        Select("kernel", ("linear", "rbf", "poly"), "rbf", "map to feature space"),
        Slider("C", 0.01, 10.0, 0.01, 1.0),
        Select("gamma", ("scale", 0.1, 1.0, 5.0), "scale", "kernel width (rbf/poly)"),
    ),
    datasets=("moons", "circles", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _accuracy_vs_c),
    notes=(
        "C small → wide margin, more train errors tolerated on the "
        "overlapping `blobs noisy` rows",
        "γ = 5 on `moons` → islands around single points; γ = 'scale' keeps "
        "the regions smooth and connected",
        "linear kernel cannot bend around `circles` — geometry beats capacity",
        "swap kernel rbf ↔ linear while watching the C sweep: linear barely "
        "moves, rbf's test curve climbs then flattens",
        "the scratch simplified-SMO pays for large C on the un-separated "
        "`blobs noisy` rows — a slow fit is the honest cost of the exact "
        "margin solver it approximates",
    ),
)
