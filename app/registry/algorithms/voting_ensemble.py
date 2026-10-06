"""Card: Voting Ensemble (sklearn-only — no from-scratch implementation exists yet).

When a scratch ensemble lands in src/models/ensemble_models.py, this card
upgrades to dual-engine by adding a sources entry and a scratch branch in fit.

Members are fixed (plan Decisions §4): one member per bias family — linear
(LogisticRegression), tree (depth-3 DecisionTree), instance-based (5-NN).
Diversity of model *classes* is the lesson; member tuning is not.
"""

import plotly.graph_objects as go
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from app.core.card import AlgorithmCard, Select

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
    members = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
    return VotingClassifier(estimators=members, voting=params["voting"]).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundary(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    fig = labeled_scatter(ctx.data.X, ctx.data.y, "data")
    decision_boundary(fig, ctx.sklearn.predict, ctx.data.X, "voting-ensemble", card.grid_resolution)
    fig.update_layout(
        title=(
            "Voting ensemble decision regions — lr + depth-3 tree + 5-NN "
            f"({ctx.params['voting']} vote)"
        ),
        template="plotly_white",
    )
    return fig


def _member_vs_ensemble(ctx):
    # refit the 3 members (they have no context); the ensemble row REUSES
    # ctx.sklearn — the page already fit exactly this model, refitting it
    # wastes work and would double-count randomness (cleanup minor #7)
    members = [
        ("lr (linear)", LogisticRegression(max_iter=1000)),
        ("tree (depth 3)", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("knn (5)", KNeighborsClassifier(n_neighbors=5)),
    ]
    names = [name for name, _ in members] + ["ensemble"]
    accs = [float(accuracy_score(ctx.data.y, m.fit(ctx.data.X, ctx.data.y).predict(ctx.data.X)))
            for _, m in members]
    accs.append(float(accuracy_score(ctx.data.y, ctx.sklearn.predict(ctx.data.X))))

    fig = go.Figure(
        go.Bar(
            x=names,
            y=accs,
            marker_color=["#EF553B", "#EF553B", "#EF553B", "#636EFA"],
            text=[f"{a:.3f}" for a in accs],
            textposition="auto",
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
    sources=(),
    sklearn_only=True,
    hypers=(
        Select("voting", ("hard", "soft"), "soft"),
    ),
    datasets=("moons", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundary, _member_vs_ensemble),
    notes=(
        "hard vs soft on `moons`: averaging probabilities changes close calls",
        "the depth-3 tree member underfits — the ensemble recovers some of it",
        "the ensemble can lose to its best member: diversity is not free",
        "switch datasets `moons` ↔ `blobs noisy`: decorrelated errors help "
        "most where no member is cleanly right",
    ),
)
