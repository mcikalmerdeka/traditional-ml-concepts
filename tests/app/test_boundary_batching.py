"""decision_boundary must predict the whole mesh grid in ONE batched call.

The per-row loop (1600 single-row predicts) made scratch Random Forest
boundaries cost ~6.3s; one batched call costs ~0.36s and paints the identical
contour (predict(grid) row r equals predict(grid[r])[0] for every model the
app feeds it — sklearn and scratch alike, all predict methods accept (n, d)).

RED: the current implementation calls predict_fn once per row, so
`calls == [(grid_len, 2)]` fails and the grid argument differs.
"""

import numpy as np
import pytest
from plotly.graph_objects import Figure

from app.components.boundary import decision_boundary


class RecordingPredictor:
    """Records every call's input shape; answers a (mostly) constant label."""

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, X):
        X = np.asarray(X)
        self.calls.append(X.shape)
        return np.full(len(X), (X[:, 0].mean() > 0).astype(float))


def _mesh_X():
    return np.array([[-2.0, -2.0], [2.0, 2.0], [0.0, 0.0], [-1.0, 1.0]])


@pytest.mark.parametrize("resolution", [40, 12])
def test_decision_boundary_predicts_grid_in_one_call(resolution):
    calls = []
    fig = Figure()
    decision_boundary(fig, RecordingPredictor(calls), _mesh_X(), "unit", resolution)
    assert calls == [(resolution ** 2, 2)]


def test_decision_boundary_paints_same_contour_as_per_row_reference():
    # value-level equivalence: batched Z equals stacking per-row predicts
    def mock_predict(X):
        X = np.asarray(X)
        return (X[:, 0] > X[:, 1]).astype(float)

    fig = Figure()
    decision_boundary(fig, mock_predict, _mesh_X(), "unit", 10)

    x0 = np.linspace(_mesh_X()[:, 0].min() - 0.5, _mesh_X()[:, 0].max() + 0.5, 10)
    x1 = np.linspace(_mesh_X()[:, 1].min() - 0.5, _mesh_X()[:, 1].max() + 0.5, 10)
    grid = np.array([[a, b] for a in x0 for b in x1])
    reference_z = np.array(
        [mock_predict(grid[i][None, :])[0] for i in range(len(grid))]
    ).reshape(len(x0), len(x1))

    contour = fig.data[0]
    assert np.allclose(contour.z.T, reference_z)
