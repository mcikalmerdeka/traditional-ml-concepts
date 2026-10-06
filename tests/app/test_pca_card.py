"""PCA transform pin — Review Focus #4.

Dimensionality-reduction cards must flow through `Fitted.transform`, not
`predict`: `engines.run()` sets kind="transform" iff the family is
"dimensionality-reduction", and the projection viz reads that path.
"""

from app.core.datasets import get_dataset
from app.core.engines import run
from app.registry.discovery import get_card


def test_pca_flows_through_transform():
    card = get_card("pca")
    data = get_dataset("pca_correlated_4f")
    fitted = run(card, data, {h.name: h.default for h in card.hypers}, "sklearn")
    assert fitted.kind == "transform"                      # Review Focus #4
    proj = fitted.transform(data.X)
    assert proj.shape == (240, 2)
    assert fitted.raw.explained_variance_ratio_[0] > fitted.raw.explained_variance_ratio_[1]
