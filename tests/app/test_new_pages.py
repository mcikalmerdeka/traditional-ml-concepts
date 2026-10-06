"""Smoke + resilience pins for the two slice-3 pages (spec §7).
TDD: written BEFORE app/navigation.py / app/compare_page.py / app/reference_page.py.
"""

import sys

import pytest
from streamlit.testing.v1 import AppTest

from app.paths import ROOT

COMPARE_PAGE = str(ROOT / "app" / "compare_page.py")
REFERENCE_PAGE = str(ROOT / "app" / "reference_page.py")


def test_compare_page_smoke_boots_without_exception():
    at = AppTest.from_file(COMPARE_PAGE, default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def test_page_modules_do_not_render_on_import():
    # Review Focus #5: Home imports both page modules; a module-level render
    # would paint the page before navigation ever chooses it.
    sys.modules.pop("app.compare_page", None)
    import app.compare_page          # noqa: F401
    import app.Home                  # noqa: F401


def test_compare_row_fit_failure_leaves_page_usable(caplog):
    # Review Focus #4: patch the DISCOVERED card's fit (kmeans-degenerate
    # precedent) — one card's row errors, the page and other rows survive.
    # Ledger ruling: the plan's sys.modules["app_cards_<stem>"].fit patch is
    # inert — frozen AlgorithmCard.frozen dataclass instances hold their own
    # reference; `card.fit is module.fit` is False (verified live). Patch the
    # card the run() call path actually reads: object.__setattr__.
    import logging

    from app.registry.discovery import all_cards

    lin = [c for c in all_cards() if c.id == "linear-regression"][0]
    original = lin.fit

    def boom(data, params, engine):
        raise ValueError("forced failure")

    object.__setattr__(lin, "fit", boom)
    try:
        at = AppTest.from_file(COMPARE_PAGE, default_timeout=120)
        at.run()
        assert not at.exception, at.exception
        assert len(at.error) >= 1            # the aggregate row-failure st.error
        assert any("Compare" in t.value for t in at.title)  # page body still renders
        assert any(r.exc_info for r in caplog.records), caplog.text
    finally:
        object.__setattr__(lin, "fit", original)


def test_reference_page_smoke_boots_and_lists_every_card():
    at = AppTest.from_file(REFERENCE_PAGE, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
