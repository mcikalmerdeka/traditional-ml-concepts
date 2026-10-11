"""Card: Voting Ensemble — dual-engine.

Both engines vote over the SAME member types (one per bias family — linear,
tree-of-depth-3, 5-NN): sklearn members for sklearn, the scratch twins for
scratch. Soft voting needs predict_proba, which every scratch twin
implements.

Members are fixed (plan Decisions §4): one member per bias family rather
than three tweaks of one model. Diversity of model *classes* is the lesson;
member tuning is not.
"""

import plotly.graph_objects as go
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from app.core.card import AlgorithmCard, Select
from src.models.ensemble_models import VotingEnsembleClassifierScratch
from src.models.knn_models import KNNClassifierScratch
from src.models.linear_models import LogisticRegressionScratch
from src.models.tree_models import DecisionTreeClassifierScratch

THEORY = """## Theory

A voting ensemble trains a panel of members and merges their verdicts.
**Hard vote** — majority of predicted labels:

$$\\hat{y} = \\mathrm{mode}_m\\, \\hat{y}_m(x)$$

**Soft vote** — argmax of the averaged class *probabilities*:

$$\\hat{p} = \\tfrac{1}{M} \\sum_m p_m(y \\mid x)$$

Soft voting lets a confident 0.9 outweigh two lukewarm 0.55s — probabilities,
not just labels, carry conviction. The ensembles pay off when member *errors
decorrelate*: the members must be wrong in different places, so the merged
verdict cancels what each one gets wrong alone. That is why this panel is one
member of each **bias family** — linear (logistic regression), recursive-rule
(depth-3 tree), instance-based (5-nearest neighbors) — rather than three
tweaks of one model.

Two honest observations the bar chart will show: the depth-3 tree member
*underfits*, and the ensemble recovers part of that — but an ensemble can
also **lose** to its own best member: averaging is a bet on decorrelated
errors, and diversity is not free.
"""


def fit(data, params, engine):
    if engine == "scratch":
        members = [
            ("lr", LogisticRegressionScratch()),
            ("tree", DecisionTreeClassifierScratch(max_depth=3)),
            ("knn", KNNClassifierScratch(n_neighbors=5)),
        ]
        return VotingEnsembleClassifierScratch(
            estimators=members, voting=params["voting"]
        ).fit(data.X, data.y)
    members = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
    return VotingClassifier(estimators=members, voting=params["voting"]).fit(data.X, data.y)


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
            "Voting ensemble decision regions — lr + depth-3 tree + 5-NN — "
            f"({ctx.params['voting']} vote) — scratch (left) vs sklearn (right)"
        ),
        template="plotly_white",
        xaxis=dict(domain=[0.0, 0.45], anchor="y"),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.0, 1.0]),
        showlegend=False,
    )
    return fig


def _member_vs_ensemble(ctx):
    # refit the 3 members per engine (they have no context); each ensemble
    # row REUSES its ctx fit — the page already fit exactly that model,
    # refitting it would waste work and double-count randomness (cleanup
    # minor #7, extended to both engines)
    scratch_members = [
        ("lr (linear)", LogisticRegressionScratch()),
        ("tree (depth 3)", DecisionTreeClassifierScratch(max_depth=3)),
        ("knn (5)", KNNClassifierScratch(n_neighbors=5)),
    ]
    sklearn_members = [
        ("lr (linear)", LogisticRegression(max_iter=1000)),
        ("tree (depth 3)", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("knn (5)", KNeighborsClassifier(n_neighbors=5)),
    ]
    names, accs, colors = [], [], []
    for name, m in sklearn_members:
        names.append(name)
        accs.append(
            float(accuracy_score(ctx.data.y, m.fit(ctx.data.X, ctx.data.y).predict(ctx.data.X)))
        )
        colors.append("#EF553B")
    if ctx.scratch is not None:
        for name, m in scratch_members:
            names.append(f"{name} (scratch)")
            accs.append(
                float(accuracy_score(ctx.data.y, m.fit(ctx.data.X, ctx.data.y).predict(ctx.data.X)))
            )
            colors.append("#ff9e6b")
        names.append("ensemble (scratch)")
        accs.append(float(accuracy_score(ctx.data.y, ctx.scratch.predict(ctx.data.X))))
        colors.append("#ffd9c2")

    names.append("ensemble")
    accs.append(float(accuracy_score(ctx.data.y, ctx.sklearn.predict(ctx.data.X))))
    colors.append("#636EFA")

    fig = go.Figure(
        go.Bar(
            x=names,
            y=accs,
            text=[f"{a:.3f}" for a in accs],
            textposition="auto",
            marker_color=colors,
        )
    )
    fig.update_yaxes(title="accuracy", range=[min(0.5, min(accs) - 0.05), 1.02])
    fig.update_layout(
        title="Member vs ensemble accuracy — does the vote beat its own bench?",
        template="plotly_white",
    )
    return fig


card = AlgorithmCard(
    id="voting-ensemble",
    title="Voting Ensemble",
    family="ensembles",
    when_to_use="merging one linear, one tree, one instance-based member — diversity of errors is the lesson",
    theory=THEORY,
    sources=(("src/models/ensemble_models.py", ("VotingEnsembleClassifierScratch",)),),
    hypers=(
        Select("voting", ("hard", "soft"), "soft"),
    ),
    datasets=("moons", "circles"),  # binary only: the scratch lr member is
    # binary (see the logistic-regression page), so 3-class rows would break
    # the panel's apples-to-apples vote
    fit=fit,
    metrics=metrics,
    visualizations=(_boundaries, _member_vs_ensemble),
    notes=(
        "hard vs soft on `moons`: averaging probabilities changes close calls",
        "the depth-3 tree member underfits — the ensemble recovers some of it",
        "the ensemble can lose to its best member: diversity is not free",
        "the datasets are binary-only by design: the panel's scratch lr "
        "member is a binary classifier — 3-class voting needs a multiclass "
        "linear twin first",
        "scratch members are plain gradient-descent / exact trees — read "
        "member-vs-member differences as standalone-implementation fidelity, "
        "not engine error",
    ),
)
