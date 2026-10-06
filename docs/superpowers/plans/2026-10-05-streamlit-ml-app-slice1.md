# Streamlit ML Study Companion — Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Streamlit app framework (card registry + generic page renderer) and 4 complete algorithm cards (Linear Regression, Decision Tree, KNN dual-engine; K-Means sklearn-only) proving the pattern across regression, classification, and clustering.

**Architecture:** Algorithm "cards" (one Python module each) declare theory/LaTeX, hyperparameter widget specs, toy datasets, and a `fit` glue function. A generic page renderer turns any card into a full Streamlit page; `st.navigation` builds the sidebar from the registry automatically. `src/models/` is imported live — never modified.

**Tech Stack:** Python ≥ 3.14 (uv), Streamlit ≥ 1.37 (`st.navigation`, `st.fragment`), plotly ≥ 6.8, scikit-learn, numpy, pytest + `streamlit.testing.v1.AppTest`.

**Spec:** `docs/superpowers/specs/2026-10-05-streamlit-ml-app-design.md` (read together with this plan; §-references below point at it)

## Global Constraints

- Never modify or create files under `notebooks/`, `src/`, `docs/`, `examples/`, `assets/` (spec §1 Scope; G5: app/tests never reference `notebooks/` at all).
- No `__init__.py` files anywhere under `app/` or `tests/` — PEP 420 namespace packages (spec §14).
- Every random element seeded: dataset factories via `np.random.default_rng(0)` / sklearn `random_state=0`; every sklearn estimator constructed with `random_state=0`.
- Toy datasets: ≤ 300 rows, ≤ 4 features (spec §10).
- All charts are plotly figures (`go.Figure`); `st.plotly_chart` for rendering (spec §12).
- All file handling via `pathlib.Path` (Windows repo).
- Imports inside `app/` use absolute `app.core...` / `app.registry...` style; `app/paths.py` guarantees the repo root is on `sys.path`.
- Commands run with `uv run ...` from the repo root.

## Review Focus

Five failure modes the spec implies but no single task's tests fully exercise:

1. **Extreme hyperparameter combos crash the page** (e.g. `n_clusters` > number of rows, `max_depth` enormous on tiny data). Expected: `st.error` with the offending params, page stays usable (spec §11). Pinned by Task 12 Step 4.
2. **Notebook contamination** — any `import`, file read, or string reference to `notebooks/` inside `app/` or `tests/`. Expected: zero occurrences. Pinned by Task 7 Step 3 (grep-style test).
3. **Launch from a different working directory** — `streamlit run` from a subdirectory must still import `app.core.*`. Expected: `app/paths.py` bootstrap fixes it. Pinned by Task 1 Step 3 (chdir test).
4. **sklearn param drift / nondeterminism** (e.g. `KMeans` `n_init` default changes across versions). Expected: fit glue pins every non-slid sklearn param explicitly (`random_state=0`, `n_init`). Pinned by contract tests in Task 9 Step 2 for all four cards.
5. **Stale source display after editing `src/models`** — shown code must reflect the current file. Expected: cache keyed on file mtime. Pinned by Task 5 Step 5.

---

## File Structure

```
app/
├── Home.py                      # Task 8 — entry, st.navigation from registry
├── paths.py                     # Task 1 — sys.path bootstrap
├── registry/
│   ├── discovery.py             # Task 7 — scan + load cards by path
│   └── algorithms/
│       ├── linear_regression.py # Task 9  (dual)
│       ├── decision_tree.py     # Task 10 (dual)
│       ├── knn.py               # Task 11 (dual)
│       └── kmeans.py            # Task 12 (sklearn-only)
├── core/
│   ├── card.py                  # Task 2 — AlgorithmCard, Data, Fitted, widget specs
│   ├── datasets.py              # Task 3 — 8 seeded toy datasets
│   ├── engines.py               # Task 4 — run() + Fitted wrapping
│   ├── source_code.py           # Task 5 — class source extraction (mtime-cached)
│   └── page.py                  # Task 6 — generic renderer + widget dispatch
└── components/
    ├── scatter.py               # Task 9 — labeled 2-D scatter painter
    └── boundary.py              # Task 10 — decision-region painter
tests/app/
├── test_paths.py                # Task 1
├── test_card_types.py           # Task 2
├── test_datasets.py             # Task 3
├── test_engines.py              # Task 4
├── test_source_code.py          # Task 5
├── test_page_renderer.py        # Task 6
├── test_discovery.py            # Task 7
├── test_home.py                 # Task 8
├── test_cards_contract.py       # Task 9 — parametrized, auto-covers later cards
├── test_kmeans_degenerate.py    # Task 12
└── smoke_runner.py              # Task 8 — AppTest harness for card pages
```

---

### Task 1: Project scaffold + path bootstrap

**Files:**
- Modify: `pyproject.toml` (add `streamlit`, `pytest`)
- Create: `app/paths.py`
- Create: `tests/app/test_paths.py`

**Interfaces:**
- Produces: `app/paths.py` → `ROOT: Path`, `ensure_root_on_path() -> None` (idempotent; inserts repo root at `sys.path[0]` if missing). Every later task imports `from app.paths import ROOT, ensure_root_on_path` at module top.

- [ ] **Step 1: Add dependencies**

Run: `uv add streamlit pytest`
Expected: resolves and installs on Python 3.14.
Fallback (only if uv reports no compatible wheels for 3.14): change `requires-python = ">=3.14"` → `">=3.12"` in `pyproject.toml`, run `uv python pin 3.12`, then `uv sync`, then retry `uv add streamlit pytest`.

- [ ] **Step 2: Write the failing test** — `tests/app/test_paths.py`

```python
import os
import sys
from pathlib import Path

def test_bootstrap_adds_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)          # simulate launch from elsewhere
    import importlib
    import app.paths as paths            # requires root already importable in test env
    importlib.reload(paths)
    paths.ensure_root_on_path()
    assert str(paths.ROOT) in sys.path
    assert (paths.ROOT / "pyproject.toml").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/app/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'` (pytest rootdir config missing). Then create `tests/conftest.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
```

and re-run: Expected: FAIL with `AttributeError`/`ImportError` — `app.paths` has no `ensure_root_on_path`.

- [ ] **Step 4: Implement `app/paths.py`**

```python
"""Bootstrap so `app.*` imports work regardless of the launch directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def ensure_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/app/test_paths.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/paths.py tests/app/test_paths.py tests/conftest.py
git commit -m "feat: scaffold app with path bootstrap and streamlit/pytest deps"
```

---

### Task 2: Card contract types (`core/card.py`)

**Files:**
- Create: `app/core/card.py`
- Create: `tests/app/test_card_types.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces (exact — every later task uses these):
  - `@dataclass(frozen=True) class Data: X: np.ndarray; y: np.ndarray | None; note: str; family: str`
  - `@dataclass(frozen=True) class Fitted: raw: Any; kind: str` with methods `predict(X) -> np.ndarray` (delegates to `raw.predict`) and `transform(X) -> np.ndarray` (delegates to `raw.transform`)
  - `@dataclass(frozen=True) class PlayContext: data: Data; params: dict; scratch: Fitted; sklearn: Fitted`
  - `@dataclass(frozen=True) class Slider: name: str; min: float; max: float; step: float; default: float; help: str = ""`
  - `@dataclass(frozen=True) class Select: name: str; options: tuple; default: Any; help: str = ""` (options may hold ints, e.g. `(1, 10)`)
  - `@dataclass(frozen=True) class Toggle: name: str; default: bool; help: str = ""`
  - `@dataclass(frozen=True) class AlgorithmCard:` fields exactly as spec §7 (id, title, family, when_to_use, theory, sources, hypers, datasets, row_cap, sklearn_only, fit, metrics, visualizations) with `row_cap: int | None = None`, `sklearn_only: bool = False`
  - `Engine = Literal["scratch", "sklearn"]`
  - Validation method `AlgorithmCard.validate(self) -> None` raising `ValueError` on: empty id/title/theory; duplicate hypers names; default outside `Slider` bounds; `default` not in `Select.options`; `sources` empty XOR `sklearn_only` False (i.e. empty sources requires sklearn_only=True, and non-empty sources requires sklearn_only=False).

- [ ] **Step 1: Write the failing test** — `tests/app/test_card_types.py`

```python
import numpy as np
import pytest

from app.core.card import (
    AlgorithmCard, Data, Fitted, PlayContext, Select, Slider, Toggle,
)

def make_card(**overrides):
    fields = dict(
        id="dummy", title="Dummy", family="classification",
        when_to_use="testing", theory="## Theory\n$$y = Xw$$",
        sources=(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),),
        hypers=(Slider("max_depth", 1, 20, 1, 3),),
        datasets=("moons",), fit=lambda data, params, engine: None,
        metrics=lambda fitted, data: [("acc", 1.0)],
        visualizations=(),
    )
    fields.update(overrides)
    return AlgorithmCard(**fields)

def test_fitted_delegates_predict():
    class M:
        def predict(self, X): return np.sum(X, axis=1)
    f = Fitted(raw=M(), kind="classification")
    assert list(f.predict(np.array([[1.0, 2.0]]))) == [3.0]

def test_validate_rejects_slider_default_out_of_bounds():
    with pytest.raises(ValueError, match="bounds"):
        make_card(hypers=(Slider("max_depth", 1, 20, 1, 99),)).validate()

def test_validate_rejects_select_default_not_in_options():
    with pytest.raises(ValueError, match="options"):
        make_card(hypers=(Select("crit", ("gini", "entropy"), "chi2"),)).validate()

def test_validate_enforces_sklearn_only_sources_invariant():
    with pytest.raises(ValueError, match="sklearn_only"):
        make_card(sources=(), sklearn_only=False).validate()
    with pytest.raises(ValueError, match="sklearn_only"):
        make_card(sklearn_only=True).validate()

def test_validate_rejects_duplicate_hyper_names():
    with pytest.raises(ValueError, match="duplicate"):
        make_card(hypers=(Slider("a", 0, 1, 1, 0), Slider("a", 0, 2, 1, 1))).validate()

def test_select_accepts_non_string_options():
    card = make_card(hypers=(Select("n_init", (1, 10), 10),))
    card.validate()  # must not raise

def test_playcontext_is_frozen():
    ctx = PlayContext(
        data=Data(X=np.zeros((2, 2)), y=None, note="n", family="clustering"),
        params={}, scratch=None, sklearn=None,
    )
    with pytest.raises(Exception):
        ctx.params = {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_card_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core'` (namespace package import works without `__init__.py`; if it still fails on `app.core`, verify `tests/conftest.py` from Task 1 is present).

- [ ] **Step 3: Implement `app/core/card.py`**

Frozen dataclasses exactly per the Interfaces block. `Fitted.predict` delegates `self.raw.predict(X)`; `Fitted.transform` delegates `self.raw.transform(X)`. `validate()` implements the five checks listed above (match strings: "bounds", "options", "sklearn_only", "duplicate"). Use `from typing import Any, Callable, Literal`; no imports from other `app` modules.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/app/test_card_types.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add app/core/card.py tests/app/test_card_types.py
git commit -m "feat: add AlgorithmCard contract and widget spec types"
```

---

### Task 3: Toy dataset registry (`core/datasets.py`)

**Files:**
- Create: `app/core/datasets.py`
- Create: `tests/app/test_datasets.py`

**Interfaces:**
- Consumes: `Data` from `app.core.card`.
- Produces: `get_dataset(dataset_id: str) -> Data` (raises `KeyError` for unknown id); `dataset_ids() -> tuple[str, ...]`. Exactly these 8 ids: `lin_clean_1f`, `lin_noisy_1f`, `lin_outliers_1f`, `moons`, `circles`, `lin_separable`, `kmeans_4blobs`, `kmeans_rings`.

Factories (deterministic, exact values):
- `lin_clean_1f`: `rng = np.random.default_rng(0)`; `X = rng.uniform(-3, 3, size=(120, 1))`; `y = 3 * X[:, 0] + 2 + rng.normal(0, 0.5, 120)`; family `regression`; note `"clean linear data — both engines should agree"`
- `lin_noisy_1f`: same but noise σ = 3.0, note mentions noise
- `lin_outliers_1f`: lin_clean recipe, then inject 10 outliers: `idx = rng.choice(120, 10, replace=False); y[idx] += rng.uniform(20, 40, 10)`; note mentions robustness
- `moons`: `sklearn.datasets.make_moons(n_samples=240, noise=0.15, random_state=0)`; family `classification`
- `circles`: `make_circles(n_samples=240, factor=0.5, noise=0.08, random_state=0)`; family `classification`
- `lin_separable`: `make_blobs(n_samples=240, centers=[(-2, -2), (2, 2)], cluster_std=0.8, random_state=0)` → `(X, y)`; family `classification`
- `kmeans_4blobs`: `make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=0)` → keep `X`, **discard** labels (`y=None`); family `clustering`
- `kmeans_rings`: hand-crafted concentric rings — `n=150` points each on radius-5 and radius-2 circles: angles uniform in `[0, 2π)`, `x = r·cos(θ) + noise`, `y = r·sin(θ) + noise`, noise `N(0, 0.3)`; `y=None`; family `clustering`; note `"concentric rings — where K-Means fails"`

- [ ] **Step 1: Write the failing test** — `tests/app/test_datasets.py`

```python
import numpy as np
import pytest

from app.core.datasets import dataset_ids, get_dataset

def test_all_ids_present():
    assert set(dataset_ids()) == {
        "lin_clean_1f", "lin_noisy_1f", "lin_outliers_1f",
        "moons", "circles", "lin_separable",
        "kmeans_4blobs", "kmeans_rings",
    }

def test_unknown_id_raises():
    with pytest.raises(KeyError):
        get_dataset("nope")

@pytest.mark.parametrize("ds_id", list(dataset_ids()))
def test_shapes_and_determinism(ds_id):
    d1, d2 = get_dataset(ds_id), get_dataset(ds_id)
    assert d1.X.shape[0] <= 300 and d1.X.shape[1] <= 4
    assert np.array_equal(d1.X, d2.X)          # deterministic
    if d1.y is not None:
        assert d1.y.shape[0] == d1.X.shape[0]
    assert d1.note and d1.family

def test_clustering_sets_have_no_labels():
    for ds_id in ("kmeans_4blobs", "kmeans_rings"):
        assert get_dataset(ds_id).y is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.datasets'`

- [ ] **Step 3: Implement `app/core/datasets.py`**

Registry dict built at import by the factory functions above; `get_dataset` returns the stored `Data`. No Streamlit imports (pure; page layer caches).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/app/test_datasets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/datasets.py tests/app/test_datasets.py
git commit -m "feat: add seeded toy dataset registry"
```

---

### Task 4: Engine runner (`core/engines.py`)

**Files:**
- Create: `app/core/engines.py`
- Create: `tests/app/test_engines.py`

**Interfaces:**
- Consumes: `AlgorithmCard`, `Data`, `Fitted`, `PlayContext` from `app.core.card`.
- Produces: `run(card: AlgorithmCard, data: Data, params: dict, engine: Literal["scratch", "sklearn"]) -> Fitted` — calls `card.fit(data, params, engine)`, wraps the returned raw model as `Fitted(raw, kind)` where `kind` is `"transform"` iff `card.family == "dimensionality-reduction"`, else `"predict"`. Propagates exceptions from `card.fit` (never swallows).

- [ ] **Step 1: Write the failing test** — `tests/app/test_engines.py`

```python
import numpy as np
import pytest

from app.core.card import AlgorithmCard, Data
from app.core.engines import run

def dummy_card(fit_fn, family="classification"):
    return AlgorithmCard(
        id="d", title="D", family=family, when_to_use="", theory="t",
        sources=(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),),
        hypers=(), datasets=(), fit=fit_fn,
        metrics=lambda f, d: [("m", 1.0)], visualizations=(),
    )

def data(): return Data(X=np.zeros((4, 2)), y=None, note="n", family="clustering")

def test_run_wraps_raw_model_and_delegates_predict():
    class M:
        def predict(self, X): return X[:, 0]
    card = dummy_card(lambda d, p, e: M())
    f = run(card, data(), {}, "scratch")
    assert isinstance(f.raw, M)
    assert f.kind == "predict"
    assert list(f.predict(np.array([[7.0, 0.0]]))) == [7.0]

def test_run_sets_transform_kind_for_dim_reduction():
    class M:
        def transform(self, X): return X
    card = dummy_card(lambda d, p, e: M(), family="dimensionality-reduction")
    assert run(card, data(), {}, "sklearn").kind == "transform"

def test_run_propagates_fit_errors():
    def boom(d, p, e): raise ValueError("bad combo")
    card = dummy_card(boom)
    with pytest.raises(ValueError, match="bad combo"):
        run(card, data(), {}, "sklearn")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_engines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.engines'`

- [ ] **Step 3: Implement `app/core/engines.py`**

Per Interfaces block. No Streamlit imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/app/test_engines.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/core/engines.py tests/app/test_engines.py
git commit -m "feat: add engine runner wrapping fits as Fitted adapters"
```

---

### Task 5: Source extraction (`core/source_code.py`)

**Files:**
- Create: `app/core/source_code.py`
- Create: `tests/app/test_source_code.py`

**Interfaces:**
- Consumes: `app.paths.ROOT`.
- Produces: `get_class_source(rel_path: str, class_name: str) -> str` — returns the full source text of the class (docstring included) from the module file at repo-root-relative `rel_path`; raises `FileNotFoundError` if the file is missing, `ValueError("class not found")` if the class is absent. Module-level dict cache keyed `(rel_path, class_name, stat.st_mtime_ns, stat.st_size)` so edits to `src/models` show up without restarting the app.

- [ ] **Step 1: Write the failing test** — `tests/app/test_source_code.py`

```python
import pytest

from app.core.source_code import get_class_source

def test_extracts_real_class_from_src_models():
    src = get_class_source("src/models/linear_models.py", "LinearRegressionScratch")
    assert "class LinearRegressionScratch" in src
    assert "def fit" in src

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        get_class_source("src/models/does_not_exist.py", "X")

def test_missing_class_raises():
    with pytest.raises(ValueError, match="class not found"):
        get_class_source("src/models/linear_models.py", "NoSuchClass")

def test_cache_respects_file_edits(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("class A:\n    x = 1\n")
    rel = str(f)  # absolute path also allowed
    assert "x = 1" in get_class_source(rel, "A")
    f.write_text("class A:\n    x = 22\n")  # different size → new cache key
    assert "x = 22" in get_class_source(rel, "A")
    assert "x = 1" not in get_class_source(rel, "A")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_source_code.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.source_code'`

- [ ] **Step 3: Implement `app/core/source_code.py`**

Import the module via `importlib.util.spec_from_file_location` on `(ROOT / rel_path)` (absolute paths pass through unchanged), then `inspect.getsource(getattr(module, class_name))`. Cache: module-level dict `{(rel_path, class_name): (mtime_ns, size, source)}` — re-extract when mtime_ns or size differs from the stored entry. (Plain dict, not lru_cache, because the key must include the file stat.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/app/test_source_code.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/core/source_code.py tests/app/test_source_code.py
git commit -m "feat: add mtime-aware class source extraction from src/models"
```

---

### Task 6: Generic page renderer (`core/page.py`)

**Files:**
- Create: `app/core/page.py`
- Create: `tests/app/test_page_renderer.py`
- Create: `tests/app/smoke_runner.py`

**Interfaces:**
- Consumes: `AlgorithmCard`, `PlayContext`, widget specs from `app.core.card`; `run` from `app.core.engines`; `get_dataset` from `app.core.datasets`; `get_class_source` from `app.core.source_code`.
- Produces:
  - `resolve_hypers(hypers: tuple, values: dict) -> dict` — pure: merges resolved widget values over declared defaults (used by tests and the renderer).
  - `render_hypers(hypers: tuple) -> dict` — renders each spec to its Streamlit widget in the current sidebar section (`st.slider` / `st.selectbox` / `st.toggle`) and returns the resolved params dict.
  - `render_card(card: AlgorithmCard) -> None` — full page per spec §8: (1) `st.title(card.title)` + family badge + `when_to_use`; (2) `st.markdown(card.theory)`; (3) code section: for each `sources` entry one `st.expander` with `st.code(get_class_source(...), language="python")`, or — when `sklearn_only` — one expander showing `inspect.getsource(card.fit)`; (4) playground inside `@st.fragment`: sidebar section (`st.sidebar` subheader "Playground") with dataset picker (`st.selectbox` over `card.datasets`, caption = dataset note) + `render_hypers(card.hypers)`, then fits via `run` for both engines (sklearn-only cards: `sklearn` engine only) inside try/except → on exception `st.error` with the params snapshot and return early; then each viz `st.plotly_chart(viz(PlayContext(...)), use_container_width=True)` individually guarded by try/except; (5) metrics row: `st.columns` — for each `(label, value)` from `card.metrics(fitted, data)` per engine, a `st.metric(label=f"{label} — {engine}", value=f"{value:.3f}")`; (6) `card.notes` bullets via `st.markdown`. Dataset access goes through `st.cache_data(get_dataset)` (spec §12).
  - `run_card_page(card_id: str) -> None` — convenience used by `Home.py` and the smoke runner: `ensure_root_on_path()`; `render_card(get_card(card_id))`.

Layout rule: theory/code/notes render once per page-load; ONLY the playground block lives inside the fragment so slider moves re-run fits + viz, not theory rendering.

- [ ] **Step 1: Write the failing test** — `tests/app/test_page_renderer.py`

```python
import numpy as np
import plotly.graph_objects as go

from app.core.card import AlgorithmCard, Data, Slider
from app.core.page import resolve_hypers

def viz_card():
    def viz(ctx):
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1])
        return fig
    return AlgorithmCard(
        id="viz-dummy", title="Viz", family="classification", when_to_use="",
        theory="t", sources=(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),),
        hypers=(Slider("max_depth", 1, 20, 1, 3),), datasets=("moons",),
        fit=lambda d, p, e: type("M", (), {"predict": staticmethod(lambda X: X[:, 0])})(),
        metrics=lambda f, d: [("acc", 0.9)], visualizations=(viz,),
    )

def test_resolve_hypers_defaults_and_overrides():
    card = viz_card()
    assert resolve_hypers(card.hypers, {}) == {"max_depth": 3}
    assert resolve_hypers(card.hypers, {"max_depth": 9}) == {"max_depth": 9}

def test_viz_is_pure_function_of_context():
    card = viz_card()
    data = Data(X=np.array([[0.0, 0.0], [1.0, 1.0]]), y=np.array([0, 1]),
                note="n", family="classification")
    from app.core.engines import run
    s = run(card, data, {}, "scratch"); k = run(card, data, {}, "sklearn")
    from app.core.card import PlayContext
    fig = card.visualizations[0](PlayContext(data, {}, s, k))
    assert isinstance(fig, go.Figure)
```

And `tests/app/smoke_runner.py` (the AppTest harness — a real file, rendered per card):

```python
"""Rendered by AppTest.from_file with SMOKE_CARD_ID env var set."""
import os
from app.paths import ensure_root_on_path

ensure_root_on_path()
from app.registry.discovery import get_card  # noqa: E402 (after bootstrap)
from app.core.page import run_card_page      # noqa: E402

run_card_page(os.environ["SMOKE_CARD_ID"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_page_renderer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.page'`

- [ ] **Step 3: Implement `app/core/page.py`**

Per Interfaces block. Widget dispatch: `Slider → st.slider(label=name, min_value=min, max_value=max, value=default, step=step, help=help)`, `Select → st.selectbox(label=name, options=list(options), index=options.index(default), help=help)`, `Toggle → st.toggle(label=name, value=default, help=help)`. `render_card` may import Streamlit at module top — this module is Streamlit-facing by design. Metrics formatting: value rendered to 3 decimals.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/app/test_page_renderer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/core/page.py tests/app/test_page_renderer.py tests/app/smoke_runner.py
git commit -m "feat: add generic card page renderer with playground fragment"
```

---

### Task 7: Card discovery (`registry/discovery.py`)

**Files:**
- Create: `app/registry/discovery.py`
- Create: `tests/app/test_discovery.py`

**Interfaces:**
- Consumes: `AlgorithmCard` from `app.core.card`; `app.paths.ROOT`.
- Produces:
  - `discover_cards(directory: Path | None = None) -> list[AlgorithmCard]` — scans `app/registry/algorithms/*.py` (default), loads each module by file path with `importlib.util.spec_from_file_location` (unique module name per file, e.g. `app_cards_<stem>`), reads its module-level `card` attribute, calls `card.validate()`; raises `ValueError("duplicate card id")` on id collisions. Skips files whose name starts with `_`.
  - `get_card(card_id: str) -> AlgorithmCard` — raises `KeyError` if absent.
  - `all_cards() -> list[AlgorithmCard]` — cached at module level after first call.

- [ ] **Step 1: Write the failing test** — `tests/app/test_discovery.py`

```python
import pytest

from app.registry.discovery import all_cards, discover_cards, get_card

def test_finds_real_cards_once_authored():
    ids = {c.id for c in all_cards()}
    assert "linear-regression" in ids

def test_unknown_card_raises():
    with pytest.raises(KeyError):
        get_card("nope")

def test_duplicate_ids_rejected(tmp_path):
    body = (
        "from app.core.card import AlgorithmCard, Slider\n"
        "card = AlgorithmCard(id='dup', title='T', family='classification', "
        "when_to_use='', theory='t', sources=(('src/models/tree_models.py', "
        "('DecisionTreeClassifierScratch',)),), hypers=(), datasets=(), "
        "fit=lambda d,p,e: None, metrics=lambda f,d: [('m',1.0)], visualizations=())\n"
    )
    (tmp_path / "a_one.py").write_text(body)
    (tmp_path / "a_two.py").write_text(body)
    with pytest.raises(ValueError, match="duplicate card id"):
        discover_cards(tmp_path)

def test_no_notebook_references_anywhere():
    # G5: the app and its tests must never reference notebooks/
    from pathlib import Path
    import app, tests
    roots = [Path(app.__path__[0]), Path(tests.__path__[0])]
    for root in roots:
        for f in root.rglob("*.py"):
            assert "notebooks" not in f.read_text(encoding="utf-8"), f
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_discovery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.registry'`

- [ ] **Step 3: Implement `app/registry/discovery.py`**

Per Interfaces block. Note: `tests` also needs to be importable as a namespace package for the G5 test — it is, because repo root is on `sys.path` and `tests/` has no `__init__.py`.

- [ ] **Step 4: Run test to verify it fails on the missing card, passes the rest**

Run: `uv run pytest tests/app/test_discovery.py -v`
Expected: `test_finds_real_cards_once_authored` FAIL (`assert "linear-regression" in ids` — no cards authored yet); the other 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/registry/discovery.py tests/app/test_discovery.py
git commit -m "feat: add card registry discovery with duplicate-id guard"
```

---

### Task 8: Home entry point (`app/Home.py`)

**Files:**
- Create: `app/Home.py`
- Create: `tests/app/test_home.py`

**Interfaces:**
- Consumes: `ensure_root_on_path`, `discover_cards`, `render_card`.
- Produces: the runnable app (`uv run streamlit run app/Home.py`). Home page function `render_home()` shows: title "Traditional ML Concepts", a one-paragraph explanation of the playground concept, family-grouped list of card titles with their `when_to_use` lines, and a note that `notebooks/` is a separate experimentation environment (no links, per G5).

- [ ] **Step 1: Write the failing test** — `tests/app/test_home.py`

```python
from streamlit.testing.v1 import AppTest

def test_home_boots_without_exception():
    at = AppTest.from_file("app/Home.py", default_timeout=30)
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/app/test_home.py -v`
Expected: FAIL — `FileNotFoundError` (app/Home.py missing)

- [ ] **Step 3: Implement `app/Home.py`**

```python
from app.paths import ensure_root_on_path

ensure_root_on_path()
import streamlit as st
from app.core.page import render_card
from app.registry.discovery import discover_cards

st.set_page_config(page_title="Traditional ML Concepts", layout="wide")

def render_home():
    st.title("Traditional ML Concepts")
    st.markdown(...landing content per Interfaces...)

cards = discover_cards()
st.navigation(
    [st.Page(render_home, title="Home", icon="🏠", default=True)]
    + [st.Page(lambda c=c: render_card(c), title=c.title, icon="📘", url_path=c.id)
       for c in cards]
).run()
```

(The lambda-with-default-arg closure is required — late binding would render only the last card.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/app/test_home.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/Home.py tests/app/test_home.py
git commit -m "feat: add Home entry with registry-driven navigation"
```

---

### Task 9: Card — Linear Regression (dual engine) + contract tests

**Files:**
- Create: `app/registry/algorithms/linear_regression.py`
- Create: `tests/app/test_cards_contract.py`
- Create: `app/components/scatter.py`

**Interfaces:**
- Consumes: all `core` modules; `LinearRegressionScratch`, `RidgeRegressionScratch`, `LassoRegressionScratch` from `src.models.linear_models`; sklearn `LinearRegression`, `Ridge`, `Lasso`, `r2_score`, `mean_squared_error`.
- Produces: card id `linear-regression`; `app/components/scatter.py` → `labeled_scatter(X: np.ndarray, y: np.ndarray, title: str) -> go.Figure` (markers, axis titles "x₀…xₙ"/"y") — reused by Tasks 10–12.

Card content:
- `theory`: markdown + LaTeX with these required elements: model form `$$\\hat{y} = Xw + b$$`, MSE objective `$$\\min_w \\tfrac{1}{n}\\|y - Xw - b\\|_2^2$$`, normal equation `$$w = (X^\\top X)^{-1} X^\\top y$$`, ridge closed form `$$w = (X^\\top X + \\alpha I)^{-1} X^\\top y$$`, lasso objective `$$\\min_w \\tfrac{1}{n}\\|y - Xw\\|_2^2 + \\alpha\\|w\\|_1$$`. One paragraph each: what it is, the three variants, when to use.
- `sources`: `(("src/models/linear_models.py", ("LinearRegressionScratch", "RidgeRegressionScratch", "LassoRegressionScratch")),)`
- `hypers`: `Select("algorithm", ("linear", "ridge", "lasso"), "linear", "variant; alpha applies only to ridge/lasso")`, `Slider("alpha", 0.0, 5.0, 0.2, 1.0, "regularization strength (ridge/lasso only)")`, `Toggle("fit_intercept", True, "include bias term b")`
- `datasets`: `("lin_clean_1f", "lin_noisy_1f", "lin_outliers_1f")`
- `fit` glue (the ONLY non-determined body in this task — exact mapping):

```python
def fit(data, params, engine):
    variant = params["algorithm"]
    if engine == "scratch":
        cls = {"linear": LinearRegressionScratch, "ridge": RidgeRegressionScratch,
               "lasso": LassoRegressionScratch}[variant]
        kwargs = {"fit_intercept": params["fit_intercept"]}
        if variant != "linear":
            kwargs["alpha"] = params["alpha"]
        return cls(**kwargs).fit(data.X, data.y)
    # sklearn side — closed-form estimators are deterministic; only Lasso takes
    # random_state (used by its coordinate-selection path)
    sk_cls = {"linear": LinearRegression, "ridge": Ridge, "lasso": Lasso}[variant]
    kwargs = {"fit_intercept": params["fit_intercept"]}
    if variant != "linear":
        kwargs["alpha"] = params["alpha"]
    if variant == "lasso":
        kwargs["random_state"] = 0
    return sk_cls(**kwargs).fit(data.X, data.y)
```

- `metrics`: per engine — `("R²", r2_score(y, pred))`, `("MSE", mean_squared_error(y, pred))`
- `visualizations` (2): **"Fit overlay"** — `labeled_scatter` + both engines' prediction lines as two traces (`scratch` red dashed, `sklearn` blue solid; sort X before line plotting); **"Residuals"** — residual vs predicted scatter per engine (two traces).
- `notes`: e.g. "watch ridge/lasso coefficients shrink as α grows on `lin_outliers_1f`", "lasso zeroes coefficients — sparsity", "scratch vs sklearn lines should overlap on clean data".

- [ ] **Step 1: Write the failing tests** — `tests/app/test_cards_contract.py` (parametrized — later cards join automatically)

```python
import os
import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from app.core.card import PlayContext
from app.core.datasets import get_dataset
from app.core.engines import run
from app.registry.discovery import all_cards

ALL = all_cards()

def test_at_least_one_card():
    assert len(ALL) >= 1

def test_ids_unique_and_slugs():
    ids = [c.id for c in ALL]
    assert len(ids) == len(set(ids))
    assert all(i.replace("-", "").isalnum() for i in ids)

@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
@pytest.mark.parametrize("engine", ["scratch", "sklearn"])
def test_fit_and_metrics_work_on_every_dataset(card, engine):
    if card.sklearn_only and engine == "scratch":
        pytest.skip("sklearn-only card")
    for ds_id in card.datasets:
        data = get_dataset(ds_id)
        fitted = run(card, data, {h.name: h.default for h in card.hypers}, engine)
        values = card.metrics(fitted, data)
        assert values and all(isinstance(v, float) for _, v in values)

@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_every_viz_returns_figure(card):
    data = get_dataset(card.datasets[0])
    params = {h.name: h.default for h in card.hypers}
    fits = {e: run(card, data, params, e)
            for e in ["sklearn"] + ([] if card.sklearn_only else ["scratch"])}
    ctx = PlayContext(data, params,
                      fits.get("scratch"), fits["sklearn"])
    for viz in card.visualizations:
        assert isinstance(viz(ctx), go.Figure)

@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_contract_invariants(card):
    card.validate()  # raises on any violation

@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_card_page_smoke(card):
    os.environ["SMOKE_CARD_ID"] = card.id
    at = AppTest.from_file("tests/app/smoke_runner.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/app/test_cards_contract.py -v`
Expected: FAIL — `ModuleNotFoundError` / `KeyError: 'linear-regression'` (card not authored yet)

- [ ] **Step 3: Implement `app/components/scatter.py` + `app/registry/algorithms/linear_regression.py`**

Per Interfaces block and the fit-glue body above. Page loads the card module; discovery picks it up automatically.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/app -v`
Expected: PASS — including `test_discovery.py::test_finds_real_cards_once_authored` (now finds the card)

- [ ] **Step 5: Commit**

```bash
git add app/registry/algorithms/linear_regression.py app/components/scatter.py tests/app/test_cards_contract.py
git commit -m "feat: linear regression card with dual-engine playground"
```

---

### Task 10: Card — Decision Tree (dual engine)

**Files:**
- Create: `app/registry/algorithms/decision_tree.py`
- Create: `app/components/boundary.py`

**Interfaces:**
- Consumes: core modules; `DecisionTreeClassifierScratch` from `src.models.tree_models`; sklearn `DecisionTreeClassifier`, `accuracy_score`.
- Produces: card id `decision-tree`; `app/components/boundary.py` → `decision_boundary(fig: go.Figure, predict_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, name: str, colors) -> go.Figure` — paints a filled contour of `predict_fn` over a 120×120 meshgrid spanning X's range ±0.5, plus data-point scatter; called twice (once per engine) on two `go.Figure`s used as side-by-side subplots.

Card content:
- `theory`: LaTeX for entropy `$$H(y) = -\\sum_c p_c \\log_2 p_c$$`, gini `$$G(y) = 1 - \\sum_c p_c^2$$`, information gain `$$IG = H(parent) - \\sum_{branches} \\tfrac{n_b}{n} H(branch_b)$$`; paragraphs on recursive splitting + stopping criteria + overfitting.
- `sources`: `(("src/models/tree_models.py", ("DecisionTreeClassifierScratch",)),)`
- `hypers`: `Slider("max_depth", 1, 20, 1, 3)`, `Slider("min_samples_split", 2, 20, 1, 2)`, `Select("criterion", ("gini", "entropy"), "gini")`
- `datasets`: `("moons", "circles", "lin_separable")`
- `fit` glue: scratch → `DecisionTreeClassifierScratch(max_depth=params["max_depth"], min_samples_split=params["min_samples_split"], criterion=params["criterion"])`; sklearn → `DecisionTreeClassifier(max_depth=..., min_samples_split=..., criterion=..., random_state=0)`
- `metrics`: `("Accuracy", accuracy_score(y, pred))` per engine
- `visualizations` (2): **"Decision boundaries"** — one figure with 1×2 subplots: scratch boundary | sklearn boundary (via `decision_boundary`); **"Accuracy vs max_depth"** — refits internally with `train_test_split(test_size=0.3, random_state=0)`: both engines' train and test accuracy across `max_depth = 1..20` (4 traces: scratch-train, scratch-test, sklearn-train, sklearn-test) — the overfit demo.
- `notes`: e.g. "max_depth=20 on `moons`: train accuracy → 1.0 while test collapses — memorization", "criterion barely changes boundaries — try it", "min_samples_split fights overfitting from the other side".

- [ ] **Step 1: Write the failing tests**

Append to `tests/app/test_cards_contract.py` (the parametrized suite auto-discovers the new card):

```python
@pytest.mark.parametrize("card", [c for c in all_cards() if c.id == "decision-tree"])
def test_tree_overfit_curve_shows_test_gap(card):
    # on moons with max_depth=20, train accuracy must exceed test accuracy (memorization)
    from sklearn.metrics import accuracy_score
    data = get_dataset("moons")
    params = {h.name: h.default for h in card.hypers} | {"max_depth": 20}
    fitted = run(card, data, params, "sklearn")
    Xtr, Xte, ytr, yte = train_test_split(data.X, data.y, test_size=0.3, random_state=0)
    train_acc = accuracy_score(ytr, fitted.raw.predict(Xtr))
    test_acc = accuracy_score(yte, fitted.raw.predict(Xte))
    assert train_acc >= test_acc
```

(`train_test_split` imported at the top of the file with the other sklearn imports; the fit on the full `moons` set scoring higher on its own training rows than on unseen test rows is the memorization signal.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/app/test_cards_contract.py -v`
Expected: FAIL — `KeyError: 'decision-tree'` (card not authored yet)

- [ ] **Step 3: Implement `app/components/boundary.py` + `app/registry/algorithms/decision_tree.py`**

Per Interfaces block. Boundary painter builds an 80×80 meshgrid with `np.linspace(x0.min()-0.5, x0.max()+0.5, 80)` per axis, `Z = np.array([predict_fn(row[None, :]) for row in grid_points])` reshaped to (80, 80), drawn with `go.Contour`; scatter points on top (reuses `labeled_scatter` from `app/components/scatter.py`).

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/app -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/registry/algorithms/decision_tree.py app/components/boundary.py tests/app/test_cards_contract.py
git commit -m "feat: decision tree card with boundary and overfit-curve visualizations"
```

---

### Task 11: Card — KNN (dual engine)

**Files:**
- Create: `app/registry/algorithms/knn.py`

**Interfaces:**
- Consumes: core modules; `KNNClassifierScratch(n_neighbors, metric, p, weights)` from `src.models.knn_models`; sklearn `KNeighborsClassifier`, `accuracy_score`.
- Produces: card id `knn`.

Card content:
- `theory`: LaTeX — euclidean `$$d(x, x') = \\sqrt{\\sum_j (x_j - x'_j)^2}$$`, manhattan `$$d(x, x') = \\sum_j |x_j - x'_j|$$`, minkowski `$$d_p(x, x') = \\left(\\sum_j |x_j - x'_j|^p\\right)^{1/p}$$`, majority vote / distance-weighted vote formulas; paragraphs on k's bias-variance role and weight schemes.
- `sources`: `(("src/models/knn_models.py", ("KNNClassifierScratch",)),)`
- `hypers`: `Slider("n_neighbors", 1, 25, 1, 5)`, `Select("weights", ("uniform", "distance"), "uniform")`, `Select("metric", ("euclidean", "manhattan", "minkowski"), "euclidean")`, `Slider("p", 1, 5, 1, 2, "minkowski power; used only when metric='minkowski'")`
- `datasets`: `("moons", "circles", "lin_separable")`
- `fit` glue: scratch → `KNNClassifierScratch(n_neighbors=..., metric=..., p=..., weights=...)`; sklearn → `KNeighborsClassifier(n_neighbors=..., weights=..., metric=..., p=...)` (sklearn's `p` is accepted regardless of metric; no `random_state` param exists — do not pass one)
- `metrics`: `("Accuracy", accuracy_score(y, pred))` per engine
- `visualizations` (2): **"Decision boundaries"** — same 1×2-subplot pattern as Task 10; **"Accuracy vs k"** — refits internally across `n_neighbors = 1..25` on a 70/30 split, both engines × train/test (4 traces).
- `notes`: e.g. "k=1 → jagged boundary, zero train error, worst test error", "distance weighting smooths noisy neighborhoods", "compare metrics on `moons` — euclidean vs manhattan changes the boundary geometry".

- [ ] **Step 1: Write the failing tests**

Append to `tests/app/test_cards_contract.py`:

```python
@pytest.mark.parametrize("card", [c for c in all_cards() if c.id == "knn"])
def test_knn_k1_overfits_and_k25_underfits_on_moons(card):
    data = get_dataset("moons")
    base = {h.name: h.default for h in card.hypers}
    p1 = base | {"n_neighbors": 1}
    p25 = base | {"n_neighbors": 25}
    s1 = run(card, data, p1, "sklearn")
    s25 = run(card, data, p25, "sklearn")
    from sklearn.metrics import accuracy_score
    acc1 = card.metrics(s1, data)[0][1]
    acc25 = card.metrics(s25, data)[0][1]
    assert acc1 > acc25  # memorization at k=1 vs heavy smoothing at k=25
```

(This asserts on full-data training accuracy, which is exactly the memorization signal — train accuracy at k=1 is 1.0.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/app/test_cards_contract.py -v`
Expected: FAIL — `KeyError: 'knn'`

- [ ] **Step 3: Implement `app/registry/algorithms/knn.py`**

Per Interfaces block. Boundary viz reuses `app/components/boundary.py`.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/app -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/registry/algorithms/knn.py tests/app/test_cards_contract.py
git commit -m "feat: KNN card with dual-engine boundary and k-curve visualizations"
```

---

### Task 12: Card — K-Means (sklearn-only) + degenerate-combo guard

**Files:**
- Create: `app/registry/algorithms/kmeans.py`
- Create: `tests/app/test_kmeans_degenerate.py`

**Interfaces:**
- Consumes: core modules; sklearn `KMeans`, `silhouette_score`.
- Produces: card id `kmeans` with `sklearn_only=True`, `sources=()` (code section shows `inspect.getsource(card.fit)` — the renderer's sklearn-only fallback from Task 6).

Card content:
- `theory`: LaTeX — objective `$$\\min_C \\sum_{i=1}^{n} \\|x_i - \\mu_{c_i}\\|^2$$` (inertia), Lloyd iteration (assign → update) `$$\\mu_j \\leftarrow \\tfrac{1}{|C_j|} \\sum_{x_i \\in C_j} x_i$$`; paragraphs on initialization sensitivity (`n_init`), the concentric-rings failure, and k choice.
- `hypers`: `Slider("n_clusters", 1, 10, 1, 4)`, `Select("n_init", (1, 10), 10, "restarts; more = stabler centroids")`, `Slider("max_iter", 10, 500, 10, 300)`
- `datasets`: `("kmeans_4blobs", "kmeans_rings")`
- `fit` glue: `KMeans(n_clusters=params["n_clusters"], n_init=params["n_init"], max_iter=params["max_iter"], random_state=0).fit(X)` (y is None for clustering datasets)
- `metrics(fitted, data)`: `("Inertia", fitted.raw.inertia_)`, `("Silhouette", silhouette_score(data.X, fitted.raw.labels_) if len(set(fitted.raw.labels_)) > 1 else float("nan"))`
- `visualizations` (2): **"Clusters & centroids"** — scatter colored by `labels_` + centroid markers (`cluster_centers_`); **"Elbow (inertia vs k)"** — refits internally with k = 1..8 (`n_init=10, random_state=0`), inertia line + marker at current `n_clusters`.
- `notes`: e.g. "on `kmeans_rings`, k=2 splits by radius sector — K-Means assumes convex blobs", "raise `n_init` if centroids land differently across reruns".

- [ ] **Step 1: Write the failing tests** — `tests/app/test_kmeans_degenerate.py`

```python
import os
import pytest
from streamlit.testing.v1 import AppTest

from app.registry.discovery import get_card

def test_kmeans_card_is_sklearn_only():
    card = get_card("kmeans")
    assert card.sklearn_only is True
    assert card.sources == ()

def test_degenerate_combo_shows_error_not_crash():
    # n_clusters=300 > 300 rows: fit fails → page must show st.error, not raise
    os.environ["SMOKE_CARD_ID"] = "kmeans"
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("tests/app/smoke_runner.py", default_timeout=60)
    at.sidebar.selectbox[0].set_value("kmeans_rings")  # any dataset
    # find the n_clusters slider in the fragment and push it out of range
    sliders = [s for s in at.sidebar.slider if s.label == "n_clusters"]
    sliders[0].set_value(300)
    at.run()
    assert not at.exception
    assert len(at.error) >= 1
```

(The renderer's try/except around `run` from Task 6 is what converts the fit failure into `st.error`; this test pins that end-to-end through the real page.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/app/test_kmeans_degenerate.py -v`
Expected: FAIL — `KeyError: 'kmeans'`

- [ ] **Step 3: Implement `app/registry/algorithms/kmeans.py`**

Per Interfaces block. The renderer's try/except around `run` (Task 6) is what converts the fit failure into `st.error` — no extra handling needed here.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/app -v`
Expected: PASS — including every `test_cards_contract.py` parametrization now covering 4 cards × datasets × engines

- [ ] **Step 5: Commit**

```bash
git add app/registry/algorithms/kmeans.py tests/app/test_kmeans_degenerate.py
git commit -m "feat: K-Means sklearn-only card with elbow and cluster visualizations"
```

---

### Task 13: Full gates + manual acceptance

**Files:**
- Modify: `tests/app/test_home.py` (add card-completeness gate)

- [ ] **Step 1: Extend the Home smoke test**

```python
def test_slice1_cards_all_present():
    # nav completeness is guaranteed by construction in Home.py (nav list is built
    # from discover_cards()); assert the source of truth directly
    from app.registry.discovery import all_cards
    assert {c.id for c in all_cards()} == {
        "linear-regression", "decision-tree", "knn", "kmeans",
    }
```

- [ ] **Step 2: Run the whole suite**

Run: `uv run pytest tests/app -v`
Expected: PASS, zero skips except the intentional `sklearn-only card` skips in the dual-engine fit test

- [ ] **Step 3: Manual acceptance run** (human step — the slice's exit proof per spec §15)

```bash
uv run streamlit run app/Home.py
```

Checklist (verify each, then stop — this is proportionate verification, not endless probing):
1. Home lists 4 algorithms grouped by family.
2. Linear Regression page: theory LaTeX renders; source shows the three scratch classes; switch algorithm → ridge; drag α 0.2 → 5.0 on `lin_outliers_1f` — overlay lines change, metrics update, < 1 s per move.
3. Decision Tree page: boundary subplots appear; drag max_depth 3 → 20 on `moons` — boundary goes jagged, overfit curve shows the train/test gap.
4. KNN page: switch metric to manhattan — boundary geometry changes; k=1 vs k=25 visibly different.
5. K-Means page: code section shows the fit glue; `kmeans_rings` with k=2 shows the sector-split failure; set n_clusters to a huge value → friendly `st.error`, page still usable.
6. `notebooks/` untouched: `git status` shows no changes under `notebooks/`, `src/`, `docs/`, `examples/`.

- [ ] **Step 4: Final commit**

```bash
git add tests/app/test_home.py
git commit -m "test: assert home navigation covers all slice-1 cards"
```

---

## Slice 2 preview (not in this plan)

Remaining cards are mechanical: logistic regression, ridge/lasso (already inside the linear card — decide then whether to split), hierarchical clustering, DBSCAN (all sklearn-only until scratch exists), SVM, random forest, boosting family, PCA, ensembles, NN basics. Each card task will mirror Tasks 9–12: theory content → hypers → fit glue → viz → contract tests auto-join.
