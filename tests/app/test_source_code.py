import pytest

from app.core.source_code import get_class_source

from app.registry.discovery import all_cards


def test_extracts_real_class_from_src_models():
    src = get_class_source("src/models/linear_models.py", "LinearRegressionScratch")
    assert "class LinearRegressionScratch" in src
    assert "def fit" in src


def test_every_card_source_extracts_without_executing(tmp_path):
    # regression: ensemble_models.py uses RELATIVE imports — the old loader
    # exec_module()'d the file standalone and crashed. Source display must be
    # a pure parse: every (file, class) pair any card declares reads cleanly.
    sources = {
        (rel_path, class_name)
        for card in all_cards()
        for rel_path, class_names in card.sources
        for class_name in class_names
    }
    assert sources, "card fixtures must declare sources"
    for rel_path, class_name in sources:
        src = get_class_source(rel_path, class_name)
        assert f"class {class_name}" in src


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        get_class_source("src/models/does_not_exist.py", "X")


def test_missing_class_raises():
    with pytest.raises(ValueError, match="class not found"):
        get_class_source("src/models/linear_models.py", "NoSuchClass")


def test_cache_respects_file_edits(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("class A:\n    x = 1\n")
    rel = str(f)  # absolute path also allowed
    assert "x = 1" in get_class_source(rel, "A")
    f.write_text("class A:\n    x = 22\n")  # different size → new cache key
    assert "x = 22" in get_class_source(rel, "A")
    assert "x = 1" not in get_class_source(rel, "A")
