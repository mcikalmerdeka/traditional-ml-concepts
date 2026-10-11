"""Landing copy honesty: after the full-engine integration, every card runs
structurally from scratch + scikit-learn — the intro must not claim any page
runs scikit-learn alone.

RED: app/navigation.py still ships the slice-3 copy ("The rest run
scikit-learn alone until their scratch twin lands in src/models/").
"""

from app.navigation import LANDING_INTRO


def test_landing_intro_no_longer_promises_sklearn_only_pages():
    assert "scikit-learn alone" not in LANDING_INTRO
    assert "until their scratch twin lands" not in LANDING_INTRO


def test_landing_intro_names_the_compare_and_reference_pages():
    assert "Compare" in LANDING_INTRO
    assert "Quick Reference" in LANDING_INTRO
