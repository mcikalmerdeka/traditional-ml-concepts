"""Discover algorithm cards by scanning the algorithms directory.

Card modules are loaded by file path (not as packages) so the registry works
regardless of .gitignore rules or cwd. Each module must expose a module-level
`card: AlgorithmCard`. `all_cards()` caches after the first call; the app's
navigation is built from it, so adding a card file is the whole job.
"""

import importlib.util
from pathlib import Path

from app.core.card import AlgorithmCard
from app.paths import ROOT

ALGORITHMS_DIR = ROOT / "app" / "registry" / "algorithms"

_cards_cache: list[AlgorithmCard] | None = None


def discover_cards(directory: Path | None = None) -> list[AlgorithmCard]:
    """Scan `app/registry/algorithms/*.py`, load, validate; duplicate ids fail."""
    directory = directory if directory is not None else ALGORITHMS_DIR
    cards: list[AlgorithmCard] = []
    seen_ids: set[str] = set()
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"app_cards_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load card module from {path}")
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(path)  # so inspect.getsource works on card glue
        import sys

        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # noqa: S102 — repo-local, trusted

        card = getattr(module, "card", None)
        if card is None:
            raise ValueError(f"card module has no 'card' attribute: {path}")
        card.validate()
        if card.id in seen_ids:
            raise ValueError(f"duplicate card id: {card.id}")
        seen_ids.add(card.id)
        cards.append(card)
    return cards


def get_card(card_id: str) -> AlgorithmCard:
    for card in all_cards():
        if card.id == card_id:
            return card
    raise KeyError(card_id)


def all_cards() -> list[AlgorithmCard]:
    global _cards_cache
    if _cards_cache is None:
        _cards_cache = discover_cards()
    return _cards_cache
