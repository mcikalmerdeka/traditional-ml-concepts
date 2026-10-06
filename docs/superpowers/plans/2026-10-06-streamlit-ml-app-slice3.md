# Streamlit ML Study Companion — Slice 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Compare page (per-family metric tables, engine columns side by side) and the Quick Reference page (one dense table over every card), plus the 6 remaining cleanup minors — the app's final slice.

**Architecture:** Two pure logic modules (`app/compare_logic.py`, `app/reference_logic.py`) + two thin AppTest-bootable page scripts (`app/compare_page.py`, `app/reference_page.py`) + one nav module (`app/navigation.py`) that becomes the single builder of `st.Page` objects (required: `st.page_link` to callable pages needs the Page object itself, v1.65 docs). Every page renders purely from `all_cards()` declarations — G2 preserved; zero card edits except the cleanup task's six one-liners.

**Tech Stack:** Python ≥ 3.14 (uv), Streamlit ≥ 1.65, pytest + `streamlit.testing.v1.AppTest`. No new dependencies (`st.dataframe`/`st.page_link` are core).

**Spec:** `docs/superpowers/specs/2026-10-06-streamlit-ml-app-slice3-design.md` (§2 locked decisions, §4 compare, §5 quick-reference, §6 cleanup)

## Global Constraints

- Never modify or create files under `notebooks/`, `src/`, `examples/`, `assets/` (spec §1 scope; G5: app/tests never reference `notebooks/` at all).
- No `__init__.py` anywhere under `app/` or `tests/` — PEP 420 namespace packages.
- Every random element stays seeded (`np.random.default_rng(0)` / `random_state=0`); this slice adds no randomness.
- Imports absolute `app....` / `src.models...`; commands run `uv run ...` from the repo root; pathlib everywhere.
- Page scripts (like `Home.py`) must bootstrap the repo root on `sys.path` inline BEFORE any `app.*` import (streamlit launch contract — see `tests/app/test_home.py::test_home_launches_under_streamlit_sys_path_contract`).
- New pages never render at import time: page-module bodies only define; execution happens via `if __name__ == "__main__":` (AppTest's `runpy` sets `__main__`) or Home's nav. `Home.py` itself stays unguarded (slice-1 precedent — only ever executed, never imported).
- `metrics` values are `float`; compare-page cells format `f"{value:.3f}"` like the metric row in `app/core/page.py:116`.

## Review Focus

Five failure modes the spec implies but no single task's tests fully exercise — each pinned by the task named:

1. **`st.page_link` against callable pages** — v1.65 requires the *actual* `st.Page` object; a rebuilt lookalike or a url_path string raises at render. Expected: compare/reference pages render their jump-links without exception. Pinned by Task 3 Step 1 (smoke asserts `not at.exception` — streamlit raises invalid-page_link errors into AppTest's exception channel) and Task 4's smoke.
2. **Headline-label fallback** — a clustering card in an all-noise state returns `[(Clusters, …), (Noise, …)]` with no `Silhouette`; empty metric lists are conceivable for future cards. Expected: `headline()` falls back to the first metric, returns `None` on empty. Pinned by Task 1's fallback/empty tests.
3. **Most-shared tie** — clustering's three datasets are each declared by exactly 2 of 3 cards. Expected: tie breaks to the earliest position in the first-seen union → `kmeans_4blobs`. Pinned by Task 1's exact-value test.
4. **Row fit failure at defaults leaves the page usable** — a card whose fit raises must render an error cell, log + toast, and the table/other rows survive. Pinned by Task 3's forced-failure test (patch the discovered module's `fit`, kmeans-degenerate precedent).
5. **Import-time rendering** — `Home.py` importing a page module must not execute its render (double-render before navigation). Pinned by Task 3's import-safety test (import `app.compare_page` then `app.Home` in one interpreter — both must succeed cleanly).

---

## File Structure

```
app/navigation.py                  # Task 3 — single builder of st.Page objects (Home, Compare, Quick Reference, 14 cards); hosts render_home + landing copy
app/Home.py                        # Task 3 — rewritten as a thin entry script (bootstrap + st.navigation(...).run()); Task 5 — nothing (copy/bootstrap move here in Task 3)
app/compare_logic.py               # Task 1 — pure: FAMILY_ORDER, HEADLINE_LABEL, NOT_DECLARED, family_sections, dataset_union, default_dataset, headline
app/reference_logic.py             # Task 2 — pure: engines_label, hypers_label, reference_rows
app/compare_page.py                # Task 3 — AppTest-bootable script: bootstrap, render_compare(), __main__ guard
app/reference_page.py              # Task 4 — AppTest-bootable script: bootstrap, render_reference(), __main__ guard
tests/app/test_compare_logic.py    # Task 1
tests/app/test_reference_logic.py  # Task 2
tests/app/test_new_pages.py        # Tasks 3–4 — smokes, forced row-failure, import safety, nav content
tests/app/test_cards_contract.py   # Task 5 — SVM LaTeX pin + voting-reuse pin + hierarchical marker pin
tests/app/test_datasets.py         # Task 5 — var_blobs y-is-None pin
tests/app/test_dbscan_degenerate.py # Task 5 — page-level degenerate AppTest variant
```

## Interfaces (produced/consumed)

- From slice 1–2 (unchanged): `all_cards() -> list[AlgorithmCard]` (discovery, sorted by filename); `AlgorithmCard(id, title, family, when_to_use, theory, sources, hypers, datasets, row_cap, sklearn_only, fit, metrics, visualizations, notes, grid_resolution)`; `Slider/Select/Toggle` specs with `.name` and `.default`; `run(card, data, params, engine) -> Fitted`; `get_dataset(id) -> Data`; `render_card(card)` in `app/core/page.py`; `ROOT`/`ensure_root_on_path()` in `app/paths.py`.
- `FAMILY_ORDER` values must match `app/core/card.py:17-23` exactly: `("regression", "classification", "clustering", "dimensionality-reduction", "ensembles")`.

---

### Task 1: Compare logic (pure)

**Files:**
- Create: `app/compare_logic.py`
- Test: `tests/app/test_compare_logic.py`

**Interfaces:**
- Consumes: `all_cards()`, `AlgorithmCard` (fields above), family strings as in `app/core/card.py:FAMILIES`.
- Produces (Task 3 consumes):
  - `FAMILY_ORDER: tuple[str, ...]` — the 5 families in spec §4 order.
  - `HEADLINE_LABEL: dict[str, str]` — `{"regression": "R²", "classification": "Accuracy", "clustering": "Silhouette", "dimensionality-reduction": "Explained variance", "ensembles": "Accuracy"}`.
  - `NOT_DECLARED = "n/a (not declared)"`.
  - `family_sections(cards) -> list[tuple[str, list]]` — families in `FAMILY_ORDER` order, non-empty only; cards keep discovery order within a family.
  - `dataset_union(cards) -> tuple[str, ...]` — dataset ids across the section's cards, first-seen order.
  - `default_dataset(cards) -> str` — the union id declared by the most cards; tie → earliest position in the union.
  - `headline(values: list[tuple[str, float]], family: str) -> float | None` — first tuple whose label equals `HEADLINE_LABEL[family]`; else the first tuple's value; `None` on empty list.

- [ ] **Step 1: Write the failing tests** — create `tests/app/test_compare_logic.py`:

```python
import pytest

from app.compare_logic import (
    FAMILY_ORDER, HEADLINE_LABEL, NOT_DECLARED,
    dataset_union, default_dataset, family_sections, headline,
)
from app.registry.discovery import all_cards, get_card

ALL = all_cards()


def test_family_sections_cover_all_cards_in_spec_order():
    sections = family_sections(ALL)
    assert [f for f, _ in sections] == [f for f in FAMILY_ORDER]
    assert sum(len(cards) for _, cards in sections) == len(ALL)
    sizes = dict(sections)
    assert sizes["regression"] == 1            # linear-regression
    assert sizes["classification"] == 6        # dt, knn, logistic, svm, nb, nn
    assert sizes["clustering"] == 3            # hierarchical, dbscan, kmeans
    assert sizes["dimensionality-reduction"] == 1  # pca
    assert sizes["ensembles"] == 3             # gb, rf, voting


def test_classification_union_and_default():
    cards = [c for c in ALL if c.family == "classification"]
    assert dataset_union(cards) == ("moons", "circles", "lin_separable", "blobs_noisy")
    assert default_dataset(cards) == "moons"   # declared by all 6 — unique max


def test_clustering_tie_breaks_to_earliest_union_position():
    cards = [c for c in ALL if c.family == "clustering"]
    # each of the 3 ids is declared by exactly 2 of 3 cards — a genuine tie
    assert dataset_union(cards) == ("kmeans_4blobs", "var_blobs", "kmeans_rings")
    assert default_dataset(cards) == "kmeans_4blobs"  # earliest in the union


def test_headline_picks_family_label_and_falls_back():
    accuracy = [("Accuracy", 0.91)]
    assert headline(accuracy, "classification") == 0.91
    clustered = [("Clusters", 3.0), ("Noise", 0.02), ("Silhouette", 0.55)]
    assert headline(clustered, "clustering") == 0.55
    all_noise = [("Clusters", 0.0), ("Noise", 1.0)]          # no Silhouette
    assert headline(all_noise, "clustering") == 0.0           # fallback: first
    assert headline([], "clustering") is None
    assert NOT_DECLARED == "n/a (not declared)"
    assert HEADLINE_LABEL["regression"] == "R²"
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/app/test_compare_logic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.compare_logic'`.

- [ ] **Step 3: Implement `app/compare_logic.py`** — pure functions with the exact signatures/types above; no streamlit imports; docstring states the spec source (§4). `default_dataset` counts with `sum(ds in c.datasets for c in cards)`; ties resolve by `dataset_union` order.

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/app/test_compare_logic.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add app/compare_logic.py tests/app/test_compare_logic.py
git commit -m "feat: compare-page logic — family sections, dataset union/default, headline extraction"
```

---

### Task 2: Quick-reference logic (pure)

**Files:**
- Create: `app/reference_logic.py`
- Test: `tests/app/test_reference_logic.py`

**Interfaces:**
- Produces (Task 4 consumes):
  - `engines_label(card) -> str` — `"scratch + sklearn"` when `card.sklearn_only` is False, else `"sklearn only"`.
  - `hypers_label(card) -> str` — `", ".join(f"{h.name}={h.default}" for h in card.hypers)`.
  - `reference_rows(cards) -> list[dict]` — one row per card, discovery order, keys in column order: `{"card": title, "page": card.id, "family": family, "engines": engines_label(card), "datasets": ", ".join(card.datasets), "hypers": hypers_label(card), "when_to_use": when_to_use}`.

- [ ] **Step 1: Write the failing tests** — create `tests/app/test_reference_logic.py`:

```python
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
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/app/test_reference_logic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.reference_logic'`.

- [ ] **Step 3: Implement `app/reference_logic.py`** — pure; docstring cites spec §5.

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/app/test_reference_logic.py -v`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add app/reference_logic.py tests/app/test_reference_logic.py
git commit -m "feat: quick-reference logic — declaration rows for every card"
```

---

### Task 3: navigation module + thin Home + Compare page

**Files:**
- Create: `app/navigation.py`, `app/compare_page.py`
- Modify: `app/Home.py` (rewrite as thin entry script), `tests/app/test_new_pages.py` (create)

**Interfaces:**
- Consumes: Task 1's `family_sections/dataset_union/default_dataset/headline/NOT_DECLARED`; `all_cards`, `render_card`, `get_dataset`, `run`, `ROOT`, `ensure_root_on_path`.
- Produces (Tasks 3–4 consume):
  - `app/navigation.py::render_home()` — the current Home page body (title, landing copy, family-grouped listing), **with the honest-engine copy** (see Step 3) and the **idempotent bootstrap guard** is in Home.py, not here.
  - `app/navigation.py::card_pages() -> list[st.Page]` — one `st.Page` per discovered card, `title=c.title`, `icon="📘"`, `url_path=c.id`, callable `lambda c=c: render_card(c)`.
  - `app/navigation.py::build_nav() -> list[st.Page]` — `[Home (default), Compare (📊, url_path="compare"), Quick Reference (🗂, url_path="quick-reference")] + card_pages()`.
  - `app/compare_page.py::render_compare()` — the page body; Task 4's reference page uses the same script skeleton.

- [ ] **Step 1: Write the failing tests** — create `tests/app/test_new_pages.py`:

```python
"""Smoke + resilience pins for the two slice-3 pages (spec §7)."""

import sys

import pytest
from streamlit.testing.v1 import AppTest

from app.paths import ROOT

COMPARE_PAGE = str(ROOT / "app" / "compare_page.py")


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
    # Review Focus #4: patch the DISCOVERED module's fit (kmeans-degenerate
    # precedent) — one card's row errors, the page and other rows survive.
    import logging

    module = sys.modules["app_cards_linear_regression"]
    original = module.fit

    def boom(data, params, engine):
        raise ValueError("forced failure")

    module.fit = boom
    try:
        at = AppTest.from_file(COMPARE_PAGE, default_timeout=120)
        at.run()
        assert not at.exception, at.exception
        assert len(at.error) >= 1            # the aggregate row-failure st.error
        assert any("Theory" in md.value or "Compare" in md.value for md in at.markdown)
        assert any(r.exc_info for r in caplog.records), caplog.text
    finally:
        module.fit = original
```

(Note on the patch: discovery loads card modules under the name `app_cards_<stem>` and registers them in `sys.modules`; `module.fit` is the module-global the cards' own code calls. The `object.__setattr__`/`setattr` dance is because module objects accept plain assignment — keep it a plain `module.fit = boom` and restore in `finally`.)

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/app/test_new_pages.py -v`
Expected: FAIL/ERROR on all 3 — `FileNotFoundError` on the missing `compare_page.py`; `ModuleNotFoundError` on the missing `app.navigation`/`app.compare_page` imports.

- [ ] **Step 3: Implement `app/navigation.py`, rewrite `app/Home.py`, create `app/compare_page.py`**

`app/navigation.py` — module docstring: "single builder of the app's st.Page objects; Home.py and the page scripts all navigate through these same objects (v1.65: st.page_link to a callable page needs the Page object itself)." Contains `render_home` (moved verbatim from the current `Home.py:30-40`) and `LANDING_INTRO` **updated** (cleanup minor #8) to describe both page kinds honestly — required content: (a) some algorithms run "your implementation and scikit-learn side by side", (b) the rest "run scikit-learn alone until their scratch twin lands in `src/models/`", (c) the divergence lesson sentence stays. `card_pages()` and `build_nav()` per the Interfaces block; `build_nav()`'s first entry is `st.Page(render_home, title="Home", icon="🏠", default=True)`.

`app/Home.py` — rewritten thin entry: inline bootstrap **guarded** (cleanup minor #3). Home does NOT import `app.paths` (chicken-and-egg: the module itself needs the root on sys.path first) — the guarded inline insert is the whole launch contract here:

```python
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import streamlit as st

from app.navigation import build_nav

st.set_page_config(page_title="Traditional ML Concepts", layout="wide")
st.navigation(build_nav()).run()
```

(The page scripts keep importing `ensure_root_on_path` as a safety net after their own guarded insert — the pattern the current `Home.py:16` established.)

`app/compare_page.py` — script skeleton (same bootstrap+guard pattern as Home.py, then):

```python
def render_compare():
    st.title("Compare")
    st.markdown(intro)  # one short paragraph: every card fit at defaults, engines side by side
    for family, cards in family_sections(all_cards()):
        st.markdown(f"### {family}")
        options = list(dataset_union(cards))
        ds = st.selectbox(
            "dataset", options,
            index=options.index(default_dataset(cards)),
            key=f"compare::{family}",
            format_func=lambda s: s.replace("_", " "),
        )
        data = get_dataset(ds)
        st.caption(data.note)
        rows = []
        for card in cards:
            row = {"card": card.title, "page": card.id}
            engines = ["sklearn"] if card.sklearn_only else ["scratch", "sklearn"]
            if ds not in card.datasets:
                row["scratch"] = row["sklearn"] = NOT_DECLARED
                rows.append(row)
                continue
            try:
                fitteds = {eng: run(card, data, {h.name: h.default for h in card.hypers}, eng) for eng in engines}
                for eng in ("scratch", "sklearn"):
                    if eng in fitteds:
                        value = headline(card.metrics(fitteds[eng], data), card.family)
                        row[eng] = f"{value:.3f}"
                    else:
                        row[eng] = "—"          # sklearn-only card's empty engine cell
            except Exception as exc:
                logger.exception("compare fit failed for %s on %s", card.id, ds)
                st.toast(f"Fit failed for {card.title} — details in the console.")
                for eng in ("scratch", "sklearn"):
                    row[eng] = f"error: {exc}"
            rows.append(row)
        st.dataframe(rows, hide_index=True)
        failed = [r["card"] for r in rows if str(r.get("scratch", "")).startswith("error") or str(r.get("sklearn", "")).startswith("error")]
        if failed:
            st.error(f"Fit failed for: {', '.join(failed)} — see the cells above; every other row still renders.")
        for page in [p for p in card_pages() if p.url_path in {c.id for c in cards}]:
            st.page_link(page=page, label=f"open {page.title}")


if __name__ == "__main__":
    render_compare()
```

(Implementer's judgment within this shape: the per-card `page`-column content and link labels; `logger = logging.getLogger(__name__)` at top; `all_cards` import from discovery; `card_pages` imported from `app.navigation` — the SAME objects `st.navigation` received, satisfying Review Focus #1.)

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/app/test_new_pages.py tests/app/test_home.py -v`
Expected: PASS — 3 new tests + home boot/sys-path/14-id gates (Home's rewrite must not regress the launch-contract test).

- [ ] **Step 5: Commit**

```bash
git add app/navigation.py app/Home.py app/compare_page.py tests/app/test_new_pages.py
git commit -m "feat: compare page with navigation module and thin Home entry"
```

---

### Task 4: Quick-reference page + nav completes

**Files:**
- Create: `app/reference_page.py`
- Modify: `app/navigation.py` (build_nav already includes the reference entry — only if Task 3 shipped it without it; otherwise no change), `tests/app/test_new_pages.py`

**Interfaces:**
- Consumes: Task 2's `reference_rows`; `app/navigation.py::build_nav`.
- Produces: `app/reference_page.py::render_reference()`.

- [ ] **Step 1: Write the failing test** — append to `tests/app/test_new_pages.py`:

```python
REFERENCE_PAGE = str(ROOT / "app" / "reference_page.py")


def test_reference_page_smoke_boots_and_lists_every_card():
    at = AppTest.from_file(REFERENCE_PAGE, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/app/test_new_pages.py::test_reference_page_smoke_boots_and_lists_every_card -v`
Expected: FAIL — FileNotFoundError on `reference_page.py`.

- [ ] **Step 3: Implement `app/reference_page.py`** — same script skeleton as compare: bootstrap, guard, `render_reference()` rendering `st.title("Quick Reference")`, intro line, then `st.dataframe(reference_rows(all_cards()), hide_index=True)` (columns in the dict key order), then one row of `st.page_link(page=page, label=page.title)` for every `card_pages()` entry. `__main__` guard at bottom. Confirm `build_nav()` contains the Quick Reference entry (add it here if Task 3 deferred it — then Task 4's commit message covers it).

  (Spec §5 says the table's *title* is the page link; `st.dataframe` cells cannot hold `st.page_link` targets (LinkColumn needs absolute URLs, and callable pages have no file path). The jump is delivered as the `page_link` button row under the table — same objects as the sidebar nav, same destinations.)

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/app/test_new_pages.py tests/app/test_home.py -v`
Expected: PASS — 4 page tests + home gates.

- [ ] **Step 5: Commit**

```bash
git add app/reference_page.py tests/app/test_new_pages.py
git commit -m "feat: quick-reference page over all card declarations"
```

---

### Task 5: Cleanup minors (6 remaining)

**Files:**
- Modify: `app/registry/algorithms/random_forest.py`, `gradient_boosting.py`, `voting_ensemble.py`, `svm.py`, `hierarchical_clustering.py`; `tests/app/test_datasets.py`, `tests/app/test_dbscan_degenerate.py`, `tests/app/test_cards_contract.py`

- [ ] **Step 1: Write the failing tests** (three pins + one AppTest variant):

In `tests/app/test_datasets.py` — add `"var_blobs"` to the loop in `test_clustering_sets_have_no_labels` (minor #4).

In `tests/app/test_dbscan_degenerate.py` (minor #5) — the page-level variant:

```python
def test_all_noise_page_stays_usable():
    import os

    from streamlit.testing.v1 import AppTest

    from app.paths import ROOT

    os.environ["SMOKE_CARD_ID"] = "dbscan"
    at = AppTest.from_file(str(ROOT / "tests" / "app" / "smoke_runner.py"), default_timeout=120)
    at.sidebar.slider(key="dbscan::eps").set_value(0.1)
    at.run()
    assert not at.exception, at.exception
    assert len(at.error) == 0  # eps=0.1 is IN-bounds and legal — not an error, a lesson
```

In `tests/app/test_cards_contract.py` — append (minors #2, #6, #7):

```python
@pytest.mark.parametrize("card", [c for c in ALL if c.id == "svm"])
def test_svm_theory_has_no_latex_row_break(card):
    # cleanup minor #2: the hard-margin formula once rendered with an
    # unintended LaTeX row break (\\;) — the theory string must contain none
    assert "\\\\" not in card.theory


@pytest.mark.parametrize("card", [c for c in ALL if c.id == "hierarchical-clustering"])
def test_hierarchical_marker_reaches_slider_top(card):
    # cleanup minor #6: the silhouette sweep must cover the slider's full
    # range (2..10) so the "current k" marker never vanishes
    data = get_dataset("kmeans_4blobs")
    params = {h.name: h.default for h in card.hypers} | {"n_clusters": 10}
    fitteds = {e: run(card, data, params, e) for e in ["sklearn"]}
    from app.core.card import PlayContext
    ctx = PlayContext(data, params, None, fitteds["sklearn"])
    fig = card.visualizations[1](ctx)
    marker = fig.data[-1]
    assert list(marker.x) == [10]


def test_voting_ensemble_reuses_context_fit():
    # cleanup minor #7: the member-vs-ensemble viz must refit members but
    # reuse ctx.sklearn for the ensemble row — patch the module-level fit to
    # raise; if the viz calls it, this fails
    import sys as _sys

    card = get_card("voting-ensemble")
    data = get_dataset("moons")
    params = {h.name: h.default for h in card.hypers}
    sklearn_fit = run(card, data, params, "sklearn")
    module = _sys.modules["app_cards_voting_ensemble"]

    def boom(data, params, engine):
        raise AssertionError("viz must reuse ctx.sklearn, not refit")

    module.fit = boom
    from app.core.card import PlayContext
    try:
        ctx = PlayContext(data, params, None, sklearn_fit)
        card.visualizations[1](ctx)  # must not call module.fit
    finally:
        module.fit = card.fit
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/app/test_datasets.py::test_clustering_sets_have_no_labels tests/app/test_dbscan_degenerate.py::test_all_noise_page_stays_usable "tests/app/test_cards_contract.py::test_svm_theory_has_no_latex_row_break" "tests/app/test_cards_contract.py::test_hierarchical_marker_reaches_slider_top" "tests/app/test_cards_contract.py::test_voting_ensemble_reuses_context_fit" -v`
Expected: FAIL — `test_clustering_sets_have_no_labels` KeyError var_blobs? No — it passes already and must STILL pass after the loop edit (the loop edit is not a new-behavior test; it pins an invariant — keep it green-throughout). RED items: the AppTest slider-set (if streamlit's AppTest rejects the `at.sidebar.slider(key=...)` accessor form, use `at.slider(key=...)` — adjust the test, not the app); svm row-break pin (`"\\\\" in theory` currently true); marker test (sweep stops at 8 → last marker x = current only if ≤ 8, and sweep range 2..8 — with k=10 the marker trace is absent → `fig.data[-1].x` is the silhouette line → fail); voting pin (viz currently refits the ensemble via `fit(...)` → boom raises → fail).

- [ ] **Step 3: Implement the six fixes**

1. Delete `import numpy as np` from `random_forest.py`, `gradient_boosting.py`, `voting_ensemble.py` (verify: no `np.` references remain in each).
2. `svm.py` hard-margin: `\\\\;\\;` → `\\;\\;` in the THEORY source (rendered `\\;` → `\;\;`).
3. (Home bootstrap guard — already done in Task 3; nothing here.)
4. `tests/app/test_datasets.py` loop edit (as above).
5. `hierarchical_clustering.py`: `ks = list(range(2, 9))` → `list(range(2, 11))` (sweep to k=10; the `2 <= current <= 8` marker guard becomes `2 <= current <= 10`).
6. `voting_ensemble.py` `_member_vs_ensemble`: drop the `fit(ctx.data, ...)` ensemble refit — use `ctx.sklearn` (raw) for the ensemble row: `accs.append(float(accuracy_score(ctx.data.y, ctx.sklearn.predict(ctx.data.X))))`; keep the three member refits (they have no context).

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/app -v`
Expected: PASS — full suite; 6 sklearn-only scratch skips + 0 other skips; no new warnings.

- [ ] **Step 5: Commit**

```bash
git add app/registry/algorithms/random_forest.py app/registry/algorithms/gradient_boosting.py app/registry/algorithms/voting_ensemble.py app/registry/algorithms/svm.py app/registry/algorithms/hierarchical_clustering.py tests/app/test_datasets.py tests/app/test_dbscan_degenerate.py tests/app/test_cards_contract.py
git commit -m "chore: slice-2 review minors — dead imports, latex break, sweep range, voting reuse, pins"
```

---

### Task 6: Full gates + manual acceptance

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest tests/app -v`
Expected: PASS — 18-card-free count ≈ 150 passed, 10 skipped (the sklearn-only scratch-engine skips), zero new warnings; wall time similar to slice 2 (~4–6 min).

- [ ] **Step 2: Manual acceptance run** (human step — the human partner runs the app and reports back; no browser automation)

```bash
uv run streamlit run app/Home.py
```

Checklist:
1. Sidebar nav: Home, **Compare**, **Quick Reference**, then all 14 cards.
2. Home: landing copy mentions both page kinds honestly; bootstrap survives reruns (no path growth).
3. Compare: five family sections in spec order; classification defaults to `moons`, clustering to `kmeans 4blobs`; DBSCAN row shows `n/a (not declared)` on `kmeans 4blobs`; logistic row has BOTH scratch and sklearn values; svm row shows `—` under scratch.
4. Compare: switch clustering's dataset to `kmeans rings` — DBSCAN row now shows numbers; jump-links open the right card pages.
5. Quick Reference: one row per card, engines cell says `scratch + sklearn` for exactly the 4 dual cards (linear-regression, decision-tree, knn, logistic-regression).
6. SVM page: hard-margin formula renders as one line (no stray row break).
7. `git status` — nothing under `notebooks/`, `src/`, `examples/`, `assets/`.

- [ ] **Step 3: Final commit (if the acceptance surfaced fixes)** — fix, suite green, commit; otherwise nothing to commit.

---

## After this slice

The spec's phasing table is exhausted: the app covers its full scope. Future work (not sliced): each card upgrades to dual-engine by editing its own file when its scratch twin lands in `src/models/`.
