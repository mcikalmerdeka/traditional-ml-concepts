"""Shared scatter painter for toy-dataset figures."""

import numpy as np
import plotly.graph_objects as go


def labeled_scatter(X: np.ndarray, y: np.ndarray | None, title: str) -> go.Figure:
    """Scatter of the dataset in its teaching shape:

    - 1 feature + labels (regression): (x₀, y)
    - ≥ 2 features (classification / clustering): (x₀, x₁) colored by y if present
    """
    fig = go.Figure()
    if X.shape[1] == 1 and y is not None:
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=y,
                mode="markers",
                marker=dict(size=6, color="#636EFA"),
                name="data",
            )
        )
        fig.update_xaxes(title="x₀")
        fig.update_yaxes(title="y")
    else:
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=None if y is None else y,
                    colorscale="Viridis",
                    showscale=False,
                ),
                name="data",
            )
        )
        fig.update_xaxes(title="x₀")
        fig.update_yaxes(title="x₁")
    fig.update_layout(title=title, template="plotly_white")
    return fig
