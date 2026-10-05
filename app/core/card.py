"""The AlgorithmCard contract: the single authoring unit for an algorithm page.

One card = everything the generic renderer (core/page.py) needs to teach one
algorithm: distilled theory with LaTeX, hyperparameter widget specs, which toy
datasets apply, and the fit/metrics/viz glue. Cards are frozen declarations;
nothing here mutates at runtime.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

Engine = Literal["scratch", "sklearn"]

# Task families; drives default metrics and adapter kind (see core/engines.py).
FAMILIES = (
    "regression",
    "classification",
    "clustering",
    "dimensionality-reduction",
    "ensembles",
)


@dataclass(frozen=True)
class Data:
    """A toy dataset: features, optional labels, and the teaching note."""

    X: np.ndarray
    y: np.ndarray | None  # None for unsupervised families
    note: str
    family: str


@dataclass(frozen=True)
class Fitted:
    """Engine-agnostic wrapper around a fitted model (scratch or sklearn)."""

    raw: Any
    kind: str  # "predict" or "transform"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.raw.predict(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.raw.transform(X)


@dataclass(frozen=True)
class PlayContext:
    """Everything a pure visualization function may touch: data, params, both fits."""

    data: Data
    params: dict
    scratch: Fitted | None
    sklearn: Fitted


@dataclass(frozen=True)
class Slider:
    """Numeric hyperparameter → st.slider."""

    name: str
    min: float
    max: float
    step: float
    default: float
    help: str = ""


@dataclass(frozen=True)
class Select:
    """Categorical hyperparameter → st.selectbox."""

    name: str
    options: tuple
    default: Any
    help: str = ""


@dataclass(frozen=True)
class Toggle:
    """Boolean hyperparameter → st.toggle."""

    name: str
    default: bool
    help: str = ""


@dataclass(frozen=True)
class AlgorithmCard:
    """One algorithm's teaching page, declared. See spec §7."""

    id: str
    title: str
    family: str
    when_to_use: str
    theory: str
    sources: tuple[tuple[str, tuple[str, ...]], ...]
    hypers: tuple[Slider | Select | Toggle, ...]
    datasets: tuple[str, ...]
    fit: Callable[[Data, dict, Engine], Any]  # returns the RAW fitted model
    metrics: Callable[[Fitted, Data], list[tuple[str, float]]]
    visualizations: tuple[Callable[[PlayContext], Any], ...]
    row_cap: int | None = None
    sklearn_only: bool = False

    def validate(self) -> None:
        """Fail fast on malformed cards — contract tests run this on every card."""
        if not self.id or not self.title or not self.theory:
            raise ValueError("card id, title and theory must be non-empty")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family}")
        names = [h.name for h in self.hypers]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate hyperparameter name in {self.id}")
        for h in self.hypers:
            if isinstance(h, Slider) and not (h.min <= h.default <= h.max):
                raise ValueError(
                    f"hyper '{h.name}' default {h.default} outside bounds "
                    f"[{h.min}, {h.max}]"
                )
            if isinstance(h, Select) and h.default not in h.options:
                raise ValueError(f"hyper '{h.name}' default not in options")
        # sources empty ⇔ sklearn_only (sklearn-only cards show their fit glue)
        if not self.sources and not self.sklearn_only:
            raise ValueError(
                f"{self.id}: empty sources requires sklearn_only=True"
            )
        if self.sources and self.sklearn_only:
            raise ValueError(
                f"{self.id}: sklearn_only cards must not declare sources"
            )
