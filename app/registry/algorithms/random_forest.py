"""Card: Random Forest (sklearn-only — no from-scratch implementation exists yet).

When RandomForestScratch lands in src/models/ensemble_models.py, this card
upgrades to dual-engine by adding a sources entry and a scratch branch in fit.

The lesson next to the Decision Tree page: bagging averages several
overconfident trees, and the ensemble's boundary obeys the *mean* of
memorizations instead of any one of them.
"""

import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.core.card import AlgorithmCard, Slider

THEORY = """## Theory

Bootstrap aggregation (**bagging**) trains B trees, each on a random
sample-with-replacement of the rows, and averages their predictions:

$$\\hat{f}(x) = \\tfrac{1}{B}\\sum_{b=1}^{B} T_b(x)$$

Each tree is fully grown — a loud, overfit memorizer. The average is much
calmer *because* their errors disagree. For correlated members the mean's
variance obeys:

$$\\mathrm{Var}(\\bar{f}) = \\rho\\sigma^2 + \\tfrac{1 - \\rho}{B}\\sigma^2$$

with ρ the pairwise tree correlation and σ² a single tree's variance. Two
knobs control the formula's two terms: feature subsampling and row bootstrap
keep ρ low, and B pushes the second term toward 0. That is why
`max_depth = 2` with `n_estimators = 200` regularly beats
`max_depth = 20` with `n_estimators = 5`: the forest needs diversity more
than it needs any single tree's full capacity.

Contrast with a **single** tree (see the Decision Tree page): alone it keeps
every memorization — red goes up where test goes down. Inside the forest the
same memorization survives, but averaged away. Watch the boundary straighten
as B grows.
"""


def fit(data, params, engine):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=0,
    ).fit(data.X, data.y)


def metrics(fitted, data):
    pred = fitted.predict(data.X)
    return [("Accuracy", float(accuracy_score(data.y, pred)))]


def _boundary(ctx):
    from app.components.boundary import decision_boundary
    from app.components.scatter import labeled_scatter

    fig = labeled_scatter(ctx.data.X, ctx.data.y, "data")
    decision_boundary(fig, ctx.sklearn.predict, ctx.data.X, "forest", card.grid_resolution)
    fig.update_layout(
        title=(
            f"Random forest decision regions — B={ctx.params['n_estimators']}, "
            f"max_depth={ctx.params['max_depth']}"
        ),
        template="plotly_white",
    )
    return fig


def _accuracy_vs_n_estimators(ctx):
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.data.X, ctx.data.y, test_size=0.3, random_state=0
    )
    sub = type(ctx.data)(X=Xtr, y=ytr, note="", family=ctx.data.family)
    bs = list(range(5, 201, 5))
    train_acc = []
    test_acc = []
    for b in bs:
        fitted = fit(sub, dict(ctx.params, n_estimators=b), "sklearn")
        train_acc.append(float(accuracy_score(ytr, fitted.predict(Xtr))))
        test_acc.append(float(accuracy_score(yte, fitted.predict(Xte))))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bs, y=train_acc, mode="lines+markers",
            name="train", line=dict(color="#EF553B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bs, y=test_acc, mode="lines+markers",
            name="test", line=dict(color="#636EFA", width=2),
        )
    )
    fig.update_layout(
        title="Accuracy vs n_estimators — more trees calm the vote; no decay, no overfit",
        template="plotly_white",
        xaxis_title="n_estimators",
        yaxis_title="accuracy",
    )
    return fig


card = AlgorithmCard(
    id="random-forest",
    title="Random Forest",
    family="ensembles",
    when_to_use="many decorrelated overfit trees averaged into one calm classifier — variance reduction by bagging",
    theory=THEORY,
    sources=(),
    sklearn_only=True,
    hypers=(
        Slider("n_estimators", 5, 200, 5, 100),
        Slider("max_depth", 1, 20, 1, 5, "cap each tree; forest averages the rest"),
    ),
    datasets=("moons", "circles", "blobs_noisy"),
    fit=fit,
    metrics=metrics,
    visualizations=(_boundary, _accuracy_vs_n_estimators),
    notes=(
        "one deep tree memorizes (`moons`) — the forest averages the "
        "memorization away: depth 20 + B=2 vs depth 20 + B=100",
        "depth 2 + 200 trees often beats depth 20 + 5 trees — the "
        "(1−ρ)/B term rewrites accuracy as diversity × count",
        "watch the boundary straighten as B grows: single-tree islands "
        "smooth into a net-shaped region",
        "compare with the Decision Tree page at the same max_depth: train "
        "accuracy drops, test rises — that transfer is the bagging lesson",
    ),
)
