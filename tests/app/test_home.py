from streamlit.testing.v1 import AppTest

from app.paths import ROOT

HOME = str(ROOT / "app" / "Home.py")


def test_home_boots_without_exception():
    at = AppTest.from_file(HOME, default_timeout=30)
    at.run()
    assert not at.exception


def test_home_launches_under_streamlit_sys_path_contract(tmp_path):
    """Review Focus #3, pinned for real: `streamlit run` puts ONLY the script's
    directory on sys.path — no cwd, no PYTHONPATH — so Home.py must bootstrap
    the repo root itself before any `app.*` import. AppTest cannot catch a
    violation (pytest's conftest.py fixes sys.path for the whole process)."""
    import json
    import os
    import subprocess
    import sys

    # Streamlit's contract: script dir only — no cwd, no PYTHONPATH, and in
    # particular NO repo root (pytest's conftest adds it; real launches don't).
    # Exact-path matching: the venv's site-packages lives INSIDE the repo and
    # must survive the filter.
    drop = {str(ROOT), str(ROOT / "tests"), str(ROOT / "tests" / "app"), str(tmp_path), os.getcwd()}
    drop_norm = {os.path.normcase(os.path.abspath(m)) for m in drop}
    keep = [str(ROOT / "app")]  # script dir, first, like streamlit inserts it
    keep += [
        p
        for p in sys.path
        if p and os.path.normcase(os.path.abspath(p)) not in drop_norm
    ]

    driver = tmp_path / "launch_driver.py"
    driver.write_text(
        "import json, sys, runpy\n"
        "cfg = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "sys.path[:] = cfg['paths']\n"
        "try:\n"
        "    runpy.run_path(cfg['home'], run_name='__main__')\n"
        "except ModuleNotFoundError as exc:\n"
        "    print(f'BOOT-FAIL: {exc}')\n"
        "    raise SystemExit(1)\n"
        "print('BOOT-OK')\n",
        encoding="utf-8",
    )
    config = tmp_path / "launch_config.json"
    config.write_text(
        json.dumps({"paths": keep, "home": HOME}), encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, str(driver), str(config)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,  # launch from outside the repo, like a real terminal
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BOOT-OK" in proc.stdout


def test_all_cards_present():
    # slice 2 complete: nav completeness restored to strict equality over all
    # 14 spec cards (task-2 ruling relaxed this to inclusion mid-slice)
    from app.registry.discovery import all_cards

    assert {c.id for c in all_cards()} == {
        "linear-regression",
        "decision-tree",
        "knn",
        "kmeans",
        "logistic-regression",
        "svm",
        "random-forest",
        "gradient-boosting",
        "naive-bayes",
        "hierarchical-clustering",
        "dbscan",
        "pca",
        "neural-network",
        "voting-ensemble",
    }
