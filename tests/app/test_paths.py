import sys

def test_bootstrap_adds_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # simulate launch from elsewhere
    import importlib

    import app.paths as paths

    importlib.reload(paths)
    paths.ensure_root_on_path()
    assert str(paths.ROOT) in sys.path
    assert (paths.ROOT / "pyproject.toml").exists()
