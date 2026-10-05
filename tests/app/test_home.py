from streamlit.testing.v1 import AppTest

from app.paths import ROOT

HOME = str(ROOT / "app" / "Home.py")


def test_home_boots_without_exception():
    at = AppTest.from_file(HOME, default_timeout=30)
    at.run()
    assert not at.exception


def test_slice1_cards_all_present():
    # nav completeness is guaranteed by construction in Home.py (nav list is
    # built from discover_cards()); assert the source of truth directly
    from app.registry.discovery import all_cards

    assert {c.id for c in all_cards()} == {
        "linear-regression",
        "decision-tree",
        "knn",
        "kmeans",
    }
