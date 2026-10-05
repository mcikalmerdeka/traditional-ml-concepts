from streamlit.testing.v1 import AppTest

from app.paths import ROOT

HOME = str(ROOT / "app" / "Home.py")


def test_home_boots_without_exception():
    at = AppTest.from_file(HOME, default_timeout=30)
    at.run()
    assert not at.exception
