"""Card: Support Vector Machine (sklearn-only — no from-scratch implementation exists yet).

When SVMScratch lands in src/models/svm_models.py, this card upgrades to
dual-engine by adding a sources entry and a scratch branch in fit.

The playground makes two tradeoffs visible: C (margin width vs violations)
and gamma (how island-like the RBF regions become).
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from app.core.card import AlgorithmCard, Select, Slider

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
    return SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        random_state=0,
    ).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundary(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    fig = labeled_scatter(ctx.data.X, ctx.data.y, "data")
    decision_boundary(fig, ctx.sklearn.predict, ctx.data.X, "svm", card.grid_resolution)
    fig.update_layout(
        title=(
            f"SVM decision regions — kernel={ctx.params['kernel']}, "
            f"C={ctx.params['C']}, γ={ctx.params['gamma']}"
        ),
        template="plotly_white",
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
    train_acc = []
    test_acc = []
    for c in cs:
        fitted = fit(sub, dict(ctx.params, C=c), "sklearn")
        train_acc.append(float(accuracy_score(ytr, fitted.predict(Xtr))))
        test_acc.append(float(accuracy_score(yte, fitted.predict(Xte))))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cs, y=train_acc, mode="lines+markers",
            name="train", line=dict(color="#EF553B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cs, y=test_acc, mode="lines+markers",
            name="test", line=dict(color="#636EFA", width=2),
        )
    )
    fig.update_xaxes(type="log", title="C")
    fig.update_yaxes(title="accuracy", range=[min(0.5, min(test_acc) - 0.05), 1.02])
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
    sources=(),
    sklearn_only=True,
    hypers=(
        Select("kernel", ("linear", "rbf", "poly"), "rbf", "map to feature space"),
        Slider("C", 0.01, 10.0, 0.01, 1.0),
        Select("gamma", ("scale", 0.1, 1.0, 5.0), "scale", "kernel width (rbf/poly)"),
    ),
    datasets=("moons", "circles", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundary, _accuracy_vs_c),
    notes=(
        "C small → wide margin, more train errors tolerated on the "
        "overlapping `blobs noisy` rows",
        "γ = 5 on `moons` → islands around single points; γ = 'scale' keeps "
        "the regions smooth and connected",
        "linear kernel cannot bend around `circles` — geometry beats capacity",
        "swap kernel rbf ↔ linear while watching the C sweep: linear barely "
        "moves, rbf's test curve climbs then flattens",
    ),
)
