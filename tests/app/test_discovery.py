import pytest

from app.registry.discovery import all_cards, discover_cards, get_card


def test_finds_real_cards_once_authored():
    ids = {c.id for c in all_cards()}
    assert "linear-regression" in ids


def test_unknown_card_raises():
    with pytest.raises(KeyError):
        get_card("nope")


def test_duplicate_ids_rejected(tmp_path):
    body = (
        "from app.core.card import AlgorithmCard, Slider\n"
        "card = AlgorithmCard(id='dup', title='T', family='classification', "
        "when_to_use='', theory='t', sources=(('src/models/tree_models.py', "
        "('DecisionTreeClassifierScratch',)),), hypers=(), datasets=(), "
        "fit=lambda d,p,e: None, metrics=lambda f,d: [('m',1.0)], visualizations=())\n"
    )
    (tmp_path / "a_one.py").write_text(body)
    (tmp_path / "a_two.py").write_text(body)
    with pytest.raises(ValueError, match="duplicate card id"):
        discover_cards(tmp_path)


def test_no_notebook_references_anywhere():
    # G5: the app and its tests must never reference the notebook world.
    # needle is assembled to keep this test's own source clean of the literal.
    from pathlib import Path

    import app
    import tests

    needle = "notebook" + "s"
    roots = [Path(app.__path__[0]), Path(tests.__path__[0])]
    for root in roots:
        for f in root.rglob("*.py"):
            assert needle not in f.read_text(encoding="utf-8"), f
