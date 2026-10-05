# Streamlit ML Study Companion — Design Spec

- **Date:** 2026-10-05
- **Status:** Draft for review
- **Scope:** New application layer (`app/`, `tests/app/`) on top of the existing repo. No changes to `notebooks/`, `src/models/`, `docs/`, or `examples/`.

---

## 1. Purpose

A single-user, personal study companion for traditional ML, built on this repo's from-scratch
implementations. The user picks an algorithm in the sidebar and lands on one page containing:

1. the core mathematics (inline LaTeX),
2. the actual from-scratch source code (`src/models/...`, shown live),
3. an interactive playground: curated toy datasets + hyperparameter sliders that re-run
   **both** the from-scratch NumPy implementation and scikit-learn side by side.

The governing idea: **watching the from-scratch implementation and sklearn behave the same or
differently, as hyperparameters move, is the lesson.**

## 2. Goals

- **G1 — One rhythm per algorithm.** Every algorithm page has the identical structure
  (header → theory → code → playground → metrics → notes), so focus stays on the algorithm.
- **G2 — Adding an algorithm = adding one file.** A new "card" module is the only work needed;
  navigation, page rendering, and contract tests pick it up automatically.
- **G3 — Engine divergence is deliberately visible.** Scratch vs sklearn outputs are overlaid
  or toggled in the same chart, never hidden.
- **G4 — Fast slider feedback.** Small seeded datasets + `st.fragment` + caching target
  under ~1 s per slider move on slice-1 cards.
- **G5 — Notebooks are a separate world.** The app never imports, renders, or links to
  `notebooks/`, and no app code or test may read anything from `notebooks/`. Notebooks remain
  the user's independent experimentation environment and must never be modified by app work.
  Neither references the other.

## 3. Non-goals

Deployment (private local app), authentication, multi-user, CSV upload, real example datasets
(`examples/`), rendering or embedding notebooks in-app, code annotation UI, mobile layout.

## 4. Context

- Repo: 17 notebooks with a fixed pedagogy (intro → theory/math → assumptions → when to use →
  NumPy scratch → sklearn → tuning). `src/models/` holds clean from-scratch classes in 7
  family modules. uv-managed; `pyproject.toml` requires Python ≥ 3.14.
- Decision log from brainstorming:
  - Audience: the repo owner alone → velocity over polish.
  - Execution: Python-native → **Streamlit** (chosen over Next.js+FastAPI because
    hyperparameter interactions need live Python execution; chosen over precomputed static
    because sweeps would require a rebuild pipeline; chosen over WASM/JS ports to avoid a
    second algorithm implementation).
  - Engines: both from-scratch and sklearn, side by side.
  - Data: curated deterministic toy datasets only.
  - Math: inline LaTeX in the app (Streamlit renders `st.markdown` LaTeX natively).
  - Code: real `src/models` source shown in-page via `st.code`.
  - Sequencing: vertical slice first (framework + 3 complete cards), then batch the rest.
- Known environment quirks (see §14): `.gitignore` excludes `docs/` and `__init__.py`.

## 5. Architecture

**Pattern: card registry + generic page factory** (chosen over per-page standalone scripts —
17 pages of duplicated wiring would drift; and over a YAML config engine — per-algorithm
visuals are too heterogeneous for config, the escape hatch would swallow the config).

```
app/Home.py ─▶ registry discovers cards ─▶ st.navigation
                     │
                     ▼
        page.py renders any AlgorithmCard:
        header → theory → source → playground fragment → metrics → notes
                          │
                          ├── engines.py: fit(X, y, params) × {scratch, sklearn}
                          ├── datasets.py: seeded toy data by id
                          └── source_code.py: pull class source from src/models
```

- `app/core/` = framework, written once. `app/registry/algorithms/` = per-algorithm cards,
  the only ongoing authorship. `app/components/` = shared viz painters (2-D scatter,
  decision-boundary painter, line overlays).
- `src/models/` is imported at runtime and also read via `inspect.getsource` for display —
  the app always shows the real, current implementation. Single source of truth.
- Notebook separation is structural: nothing in `app/` or `tests/app/` references `notebooks/`.

## 6. Repository layout

```
E:\Traditional ML Concepts\
├── notebooks/                  # UNTOUCHED (separate experimentation environment)
├── src/models/...              # UNTOUCHED — imported live by the app
├── docs/, examples/, assets/   # UNTOUCHED
├── docs/superpowers/specs/
│   └── 2026-10-05-streamlit-ml-app-design.md   # this file (force-committed, docs/ is ignored)
├── app/
│   ├── Home.py                 # landing page; builds sidebar nav from the registry
│   ├── paths.py                # sys.path bootstrap so imports work from any cwd
│   ├── registry/
│   │   ├── discovery.py        # scans algorithms/*.py by path, collects AlgorithmCards
│   │   └── algorithms/
│   │       ├── linear_regression.py
│   │       ├── decision_tree.py
│   │       └── kmeans.py
│   ├── core/
│   │   ├── card.py             # AlgorithmCard dataclass + widget spec types
│   │   ├── engines.py          # scratch/sklearn fit + thin unified adapter
│   │   ├── datasets.py         # toy dataset registry (id → (X, y, note, family))
│   │   ├── source_code.py      # class source extraction from src/models
│   │   └── page.py             # generic page renderer used by every card
│   └── components/
│       ├── scatter.py          # labeled 2-D scatter painter
│       ├── boundary.py         # decision-region painter (classifier grids)
│       └── regression_lines.py # prediction-line overlay helpers
└── tests/app/
    ├── test_cards_contract.py  # every card × dataset × engine works; viz returns Figure
    └── test_pages_smoke.py     # AppTest boots each card's page without exception
```

Run: `uv run streamlit run app/Home.py` (from repo root — `app/paths.py` makes it work from
any launch directory). `st.navigation`-based multipage; no page files beyond `Home.py`.

## 7. The card contract

One card = one authoring unit per algorithm. Fields:

```python
@dataclass(frozen=True)
class AlgorithmCard:
    id: str                          # "linear-regression" (slug, unique)
    title: str                       # "Linear Regression"
    family: str                      # regression | classification | clustering |
                                     # dimensionality-reduction | ensembles
    when_to_use: str                 # one-liner shown in the header
    theory: str                      # markdown + LaTeX, self-contained, no notebook refs
    sources: tuple[tuple[str, tuple[str, ...]], ...]
                                     # (path relative to repo root, class names) → shown via st.code
    hypers: tuple[HyperParam, ...]   # sidebar widgets, declared once, rendered generically
    datasets: tuple[str, ...]        # ids into datasets.py registry
    row_cap: int | None              # optional per-card sample cap for slow learners
    sklearn_only: bool               # True when no from-scratch implementation exists yet;
                                     # invariant: sources is empty ⇔ sklearn_only is True
    fit: Callable[[Data, Params, Engine], Fitted]
    metrics: Callable[[Fitted, Data], list[tuple[str, float]]]
                                     # engine-agnostic: called once per fitted engine
                                     # (y optional inside; e.g. silhouette for clustering)
    visualizations: tuple[Viz, ...]  # 1–3 functions: (PlayContext) → plotly Figure
```

Shared types (defined in `core/card.py`):

- `Data` — frozen bundle `(X: np.ndarray, y: np.ndarray | None, note: str)`;
  `y is None` for unsupervised families (clustering, dimensionality-reduction).
- `Params` — `dict[str, Any]` of resolved hyperparameter values, keyed by widget `name`.
- `Engine` — `Literal["scratch", "sklearn"]`.
- `PlayContext` — frozen bundle `(data: Data, params: Params,
  scratch: Fitted, sklearn: Fitted)`, so every visualization (a) has both engine results
  available for overlaying, (b) is pure and unit-testable.

Widget spec types (`core/card.py`), rendered by `page.py` via a small dispatch:

```python
Slider(name, min, max, step, default, help)   # → st.slider
Select(name, options, default, help)          # → st.selectbox
Toggle(name, default, help)                   # → st.toggle
```

- `fit` is the only per-card glue (~10–20 lines) — needed because the scratch API and sklearn
  API differ (e.g. `n_init`, `.score()` availability). It receives the resolved `Params`
  dict and returns the fitted model object; `core/engines.py` wraps both sides so `viz`
  code never branches on engine.
- Deliberately **not** config files (YAML): visuals like decision-boundary plots, elbow
  charts, and explained-variance bars are code-shaped; a config format would grow
  function-callback escape hatches until it was Python with worse syntax.

## 8. The rendered page (identical for every card)

1. **Header** — `title`, family badge, `when_to_use`.
2. **Theory** — `theory` markdown with inline LaTeX; concise distilled math
   (model form, loss/objective, closed form or update rule). No notebook links.
3. **The code** — every `sources` entry shown in `st.code` (expandable), read from the actual
   module at runtime.
4. **Playground** — a `@st.fragment` block:
   - sidebar section: dataset picker (only the card's datasets, each with its note),
     then the card's hyperparameter widgets with declared defaults;
   - main area: the card's 1–3 viz figures, recomputed per change; fit errors shown via
     `st.error` with the offending params (a bad param combo is a learning moment).
5. **Metrics row** — one row of `st.metric` columns per engine comparison
   (e.g. scratch R² vs sklearn R²; silhouette for clustering).
6. **Notes** — 3–5 bullets on what to try with which slider.

Default metric families (card may override via `metrics`): regression → R², MSE;
classification → accuracy, F1; clustering → silhouette, inertia;
dimensionality-reduction → explained-variance ratio; ensembles → per sub-type.

## 9. Engines (`core/engines.py`)

- `Engine = Literal["scratch", "sklearn"]`.
- `run(card, data, params, engine) -> Fitted` — calls `card.fit`, catching exceptions at the
  page-render layer (never silent).
- The `Fitted` side is a thin adapter: `.predict()` for predictive families,
  `.transform()` for dimensionality-reduction; both expose `.raw` (the underlying model
  object) for algorithm-specific access (`coef_`, `centers_`, `feature_importances_`) by
  the card's own viz code, which knows both implementations. Keeping the adapter minimal
  avoids a leaky second API.
- viz functions receive both fits and *choose* how to present: overlay (two regression
  lines on one scatter), tabs (two decision boundaries), or mean-line comparison.
- **sklearn-only cards** (spec addendum 2026-10-05): algorithms whose scratch implementation
  does not exist yet in `src/models/` (K-Means, SVM, PCA, ensembles — those modules are
  placeholders) declare `sklearn_only=True` and empty `sources`; the code section then shows
  `inspect.getsource(card.fit)` (proper sklearn usage). When the scratch implementation
  lands in `src/models/` later, the card upgrades to dual-engine by editing the card file
  only. The app never writes into `src/models/`.

## 10. Toy datasets (`core/datasets.py`)

Deterministic (fixed seeds), tiny (≤ 300 rows, ≤ 4 features), `factory → (X, y, family, note)`.

- **regression:** `lin-clean-1f`, `lin-noisy-1f`, `lin-outliers-1f` (Lasso/ridge visible), `lin-2f`
- **classification:** `moons`, `circles`, `lin-separable`, `blobs-noisy`
- **clustering:** `kmeans-4blobs`, `kmeans-rings` (K-Means failure mode visible), `var-blobs`
- **dimensionality-reduction:** `pca-correlated-4f`

`note` is rendered under the picker ("noisy data → watch the tree overfit as max_depth grows").

## 11. Error handling

- Page layer: try/except around fit and each viz; error → `st.error` + params snapshot +
  suggestion text; the rest of the page stays usable.
- Contract tests precondition: every card must succeed on every registered dataset × engine
  at default params, so mid-session failures are rare, legitimately interesting combos.
- No empty catches; log unexpected fit failures to `st.toast` + console.
- Dataset factories are pure and cached; nothing user-provided enters the pipeline.

## 12. Performance

- `st.cache_data` on dataset construction and source extraction.
- `@st.fragment` around the playground so slider moves re-run fits/viz only.
- Card-declared row caps for heavy learners (boosting scratch, NN card later).
- Explicit non-goal: async fit, multiprocessing, joblib parallelism.

## 13. Testing (`tests/app/`, pytest)

- **Contract test (parametrized over every discovered card):** id/title/theory non-empty;
  every `sources` path exists and contains the named classes; every declared dataset id
  exists; `fit` succeeds for `scratch` and `sklearn` on every dataset at default params;
  `metrics` returns ≥ 1 labeled value; each viz returns a `plotly` `Figure`; hypers defaults
  in range. Adding a broken card fails tests in seconds, not mid-session.
- **Smoke test:** `streamlit.testing.v1.AppTest` boots each card's page without exception.
- Run: `uv run pytest tests/app` — no network, seeds fixed, seconds to run.

## 14. Repo hygiene (git collisions)

- The last commit's `.gitignore` ignores `docs/` and `__init__.py`; the working copy already
  comments out both rules (owner's uncommitted change). Once that lands, the spec and any
  `__init__.py` track normally. This spec is staged with `git add -f` so it commits
  regardless of which `.gitignore` version lands first.
- Independent of the ignore rules, `app/` and `tests/app/` use **PEP 420 implicit namespace
  packages** (no `__init__.py` files) and the registry loads card modules by file path
  (`importlib.util.spec_from_file_location`) — immune to ignore-rule drift and cwd/package
  issues. `app/paths.py` inserts the repo root on `sys.path`.
- `pyproject.toml`: add `streamlit` (and `pytest`) to dependencies; no other tooling changes.

## 15. Phasing

| Slice | Contents | Exit proof |
|---|---|---|
| **1** | `core/` + `Home.py` + 4 cards: **Linear Regression, Decision Tree, KNN** (dual-engine) + **K-Means** (sklearn-only — no scratch impl exists yet) | pattern proven across regression / classification / clustering, and both engine modes (dual + sklearn-only); tests green; sliders feel fast |
| **2** | Remaining ~14 cards, mechanical authoring | every notebook-family algorithm has a page; all contract tests green |
| **3** (optional) | algorithm-comparison mode, quick-reference page | — |

Slice 1 is the design's real deliverable: if the card contract survives three task families
without core edits, slice 2 is production-line work.

## 16. Risks

- **Streamlit wheels on Python 3.14** — the toolchain pins `requires-python = ">=3.14"` while
  Streamlit's 3.14 support may lag. Mitigation: `uv add streamlit`; if wheels are missing,
  run the app on an installed 3.12/3.11 interpreter without touching the notebooks baseline.
- **From-scratch speed on ensemble/NN cards** — row caps per card (declared in card spec).
- **plotly API drift** — plotly is already pinned `>=6.8`; viz helpers confined to
  `app/components/` so adjustments are localized.
- **Windows paths** — pathlib everywhere; no hardcoded separators (repo is on Windows).

## 17. Out of scope

Deployment/hosting, auth, multi-user, CSV upload, real example datasets, in-app notebook
rendering or linking (notebooks are a separate environment, see G5), test coverage of
`notebooks/` content itself.
