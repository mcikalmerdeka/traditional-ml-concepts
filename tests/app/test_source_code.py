import pytest

from app.core.source_code import get_class_source


def test_extracts_real_class_from_src_models():
    src = get_class_source("src/models/linear_models.py", "LinearRegressionScratch")
    assert "class LinearRegressionScratch" in src
    assert "def fit" in src


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
