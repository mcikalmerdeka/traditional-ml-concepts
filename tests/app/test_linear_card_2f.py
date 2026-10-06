"""Linear card pin for the 2-feature dataset (final-review Important #1).

`lin_2f` was wired into the linear card in Task 1, but `_overlay`'s
line-over-scatter only makes sense for 1-feature data: with 2 features the
line trace zigzags over an x₀-sorted scatter. On ≥2-feature datasets the
overlay must be the predicted-vs-actual view (markers, both engines) instead.
"""

import plotly.graph_objects as go

from app.core.card import PlayContext
from app.core.datasets import get_dataset
from app.core.engines import run
from app.registry.discovery import get_card


def test_overlay_is_pred_vs_actual_on_2f_dataset():
    card = get_card("linear-regression")
    data = get_dataset("lin_2f")
    params = {h.name: h.default for h in card.hypers}
    fits = {e: run(card, data, params, e) for e in ["scratch", "sklearn"]}
    ctx = PlayContext(data, params, fits["scratch"], fits["sklearn"])

    fig = card.visualizations[0](ctx)
    assert isinstance(fig, go.Figure)
    line_traces = [t for t in fig.data if t.mode == "lines"]
    assert not line_traces  # the x₀-sorted zigzag line must not appear
    # both engines show a predicted-vs-actual marker cloud
    pv_traces = [t for t in fig.data if t.mode == "markers" and t.name in ("scratch", "sklearn")]
    assert len(pv_traces) == 2


def test_overlay_keeps_line_on_1f_datasets():
    # 1-feature regression keeps the slice-1 look: sorted fit lines on the scatter
    card = get_card("linear-regression")
    data = get_dataset("lin_clean_1f")
    params = {h.name: h.default for h in card.hypers}
    fits = {e: run(card, data, params, e) for e in ["scratch", "sklearn"]}
    ctx = PlayContext(data, params, fits["scratch"], fits["sklearn"])

    fig = card.visualizations[0](ctx)
    line_traces = [t for t in fig.data if t.mode == "lines"]
    assert len(line_traces) == 2  # scratch + sklearn fit lines
