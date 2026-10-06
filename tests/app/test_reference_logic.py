"""Quick-reference pure logic (slice 3 Task 2): one declaration row per
card. TDD: written BEFORE app/reference_logic.py (spec: slice-3 design §5).
"""

from app.reference_logic import engines_label, hypers_label, reference_rows
from app.registry.discovery import all_cards, get_card

ALL = all_cards()


def test_rows_cover_exactly_the_discovered_cards():
    rows = reference_rows(ALL)
    assert [r["page"] for r in rows] == [c.id for c in ALL]
    assert all(set(r) == {"card", "page", "family", "engines", "datasets", "hypers", "when_to_use"} for r in rows)


def test_engines_and_hypers_cells_match_card_declarations():
    logistic = get_card("logistic-regression")
    assert engines_label(logistic) == "scratch + sklearn"
    svm = get_card("svm")
    assert engines_label(svm) == "sklearn only"
    rf = get_card("random-forest")
    assert hypers_label(rf) == "n_estimators=100, max_depth=5"
    row = {r["page"]: r for r in reference_rows(ALL)}["random-forest"]
    assert row["card"] == "Random Forest"
    assert row["hypers"] == "n_estimators=100, max_depth=5"
    assert row["datasets"] == "moons, circles, blobs_noisy"
