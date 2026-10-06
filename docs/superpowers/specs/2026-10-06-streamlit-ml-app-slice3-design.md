# Streamlit ML Study Companion — Slice 3 Design (final slice)

- **Date:** 2026-10-06
- **Status:** Approved in-chat; awaiting written-spec review
- **Scope:** New pages inside the existing app layer (`app/`, `tests/app/`). No changes to `notebooks/`, `src/`, `examples/`, `assets/`. No core-framework edits, no new dependencies.
- **Supersedes:** nothing; extends the design in `2026-10-05-streamlit-ml-app-design.md` (§15 slice table).

---

## 1. Purpose

Close out the app with the two §15 optional features and bank the polish debt
slice 2's review deferred:

1. **Compare** — one metric table across all cards, per family: the app's
   governing idea (engines side by side, divergence visible) applied across
   algorithms, not just within one page.
2. **Quick Reference** — a single dense cheat-sheet table over every card:
   scan the whole app in one screen, jump anywhere.
3. **Cleanup** — the 8 deferred minors from slice 2's final review, all
   app-layer.

**Phasing:** this is the **final slice** (spec §15). After it the app covers
its entire spec scope. Foreseeable future work is not sliced: when a from-
scratch implementation lands in `src/models/`, its card upgrades to dual-
engine by editing that one card file (G2).

## 2. Decisions locked (from the in-chat design session)

1. **Scope** = both features + the deferred minors as one cleanup task.
2. **Comparison basis** = per-family dataset selectbox (union of that
   family's declared datasets, defaulting to the most-shared), rows are all
   the family's cards, undeclared combos render `n/a (not declared)` — a gap
   is information, never skipped silently.
3. **Row metric** = family headline metric, read from each card's own
   `metrics()` output by label match (fallback: first metric):
   regression → `R²`; classification → `Accuracy`; ensembles → `Accuracy`;
   clustering → `Silhouette` (fallback `Clusters`); dimensionality-reduction →
   `Explained variance`.
4. **Quick-reference presentation** = one dense table (title link · family ·
   engines · datasets · hypers · when-to-use).

## 3. New pages and navigation

`Home.py`'s nav list grows by two entries between Home and the cards:

```
[Home] + [Compare, Quick Reference] + [14 card pages]
```

Both pages render purely from `all_cards()` declarations — discovery-driven,
auto-correct as cards come and go (G2). Zero card edits anywhere in this
slice.

## 4. Compare page (`app/compare_page.py`)

Per family (fixed order: regression, classification, clustering,
dimensionality-reduction, ensembles — families with no cards render nothing):

- **Dataset selectbox** — union of the family's declared dataset ids,
  defaulting to the most-shared id in that family (classification →
  `moons`; clustering → `kmeans_4blobs`); ties/near-ties break on first
  declaration order. Selection reruns that family's rows only.
- **One row per card**: title (jump-link to `url_path=card.id`) · scratch
  value · sklearn value · used dataset id.
- **Engine cells** (G3 in table form): dual-engine cards show both fits'
  headline values side by side; sklearn-only cards show one value and "—"
  for scratch; a card not declaring the selected dataset shows
  `n/a (not declared)` in both cells.
- **Fits at declared defaults only** — this page has no hyperparameter
  widgets; `run()` on the selected dataset with every hyper at its default.
  A fit failure renders `st.error` in that row with the exception text;
  the table and the rest of the page stay usable (spec §11).
- **Performance:** 14 default-param fits per full-page selection change;
  the existing `st.cache_data` dataset cache applies; no new caching
  infrastructure (spec §12's fragment/caching story already covers this
  scale).

## 5. Quick-reference page (`app/reference_page.py`)

Single table over all discovered cards, one row per card:

**Title (page link) · family · engines (`scratch + sklearn` or `sklearn only`) ·
datasets · hypers (`name=default`, comma-joined) · when_to_use.**

No widgets, no fits. Pure declaration rendering.

## 6. Cleanup task (slice 2's deferred minors)

One task, all app-layer:

1. Drop unused `import numpy as np` from `random_forest.py`,
   `gradient_boosting.py`, `voting_ensemble.py`.
2. `svm.py` hard-margin formula: `\\;\\;` → `\;\;` (removes the unintended
   LaTeX row break).
3. `Home.py` bootstrap: guard the `sys.path.insert` with
   `if root not in sys.path` (idempotent across Streamlit reruns).
4. Pin `var_blobs`'s `y is None` in `tests/app/test_datasets.py`'s clustering
   loop.
5. DBSCAN degenerate case gains a page-level AppTest variant (eps = 0.1 in
   bounds → page renders, no exception — pins spec §11's "page stays usable"
   half end-to-end).
6. `hierarchical_clustering.py` silhouette sweep extended to k = 10 so the
   "current k" marker never vanishes inside the slider's range.
7. `voting_ensemble.py`'s member-vs-ensemble viz reuses `ctx.sklearn` instead
   of refitting the ensemble it already holds.
8. `Home.py` landing copy describes dual-engine vs sklearn-only pages
   honestly (10 of 14 cards are sklearn-only).

## 7. Testing (`tests/app/`)

- **Compare contract:** for every card × its declared datasets × engines,
  the table's rendered value equals a direct `run()` + headline-metric
  extraction (pure function, tested without AppTest); n/a cells verified for
  undeclared combos; every family section boots.
- **Quick-reference contract:** rows exactly cover `all_cards()` ids; every
  hypers cell parses to the card's declared hypers; engines cell matches
  `sklearn_only`.
- **Smoke:** `AppTest` boots both new pages without exception (added to the
  existing smoke pattern).
- **Cleanup pins** where behavior changed: LaTeX string test on the SVM
  theory; bootstrap-idempotency unit test on `Home.py`'s guard; hierarchical
  sweep length; voting viz figure equivalence after the reuse change.
- Whole suite stays green; existing gates (14-card home set, determinism,
  degenerate pins) untouched.

## 8. Error handling

Same contract as everywhere (spec §11): try/except around each family's fit
and table row; failures log to console with traceback + on-page `st.error`;
no empty catches; the page never dies on one bad row.

## 9. Out of scope

Anything beyond the two pages + cleanup: per-card hyperparameter controls on
the compare page, cross-family normalization, CSV export, deployment,
notebook integration (G5 stands).
