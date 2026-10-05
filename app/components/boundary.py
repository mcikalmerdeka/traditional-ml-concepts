"""Decision-region painter for classifier playgrounds."""

from typing import Callable

import numpy as np
import plotly.graph_objects as go


def decision_boundary(
    fig: go.Figure,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    name: str,
) -> go.Figure:
    """Paint a filled contour of predict_fn over an 80×80 meshgrid spanning
    X's range ±0.5, then the data points on top. Mutates and returns fig."""
    x0 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 80)
    x1 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 80)
    grid = np.array([[a, b] for a in x0 for b in x1])
    Z = np.array([predict_fn(row[None, :])[0] for row in grid]).reshape(len(x0), len(x1))

    fig.add_trace(
        go.Contour(
            x=x0,
            y=x1,
            z=Z.T,
            showscale=False,
            opacity=0.5,
            colorscale="Viridis",
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
            name=f"{name} region",
            hoverinfo="skip",
        )
    )
    return fig
