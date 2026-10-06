# Streamlit ML Study Companion — Slice 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the remaining 10 algorithm cards (mechanical authoring) plus the 4 registry datasets they need, so every notebook-family algorithm has a page and all contract tests pass.

**Architecture:** Zero core-framework changes. Every card is one module in `app/registry/algorithms/` picked up automatically by discovery, contract tests, and navigation — this slice is the proof of spec G2 ("adding an algorithm = adding one file"). The only `core/` edit is `datasets.py` growing by the 4 spec-§10 datasets slice 1 deferred.

**Tech Stack:** Python ≥ 3.14 (uv), Streamlit ≥ 1.65, plotly ≥ 6.8, scikit-learn, numpy, pytest + `streamlit.testing.v1.AppTest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-streamlit-ml-app-design.md` (§10 datasets, §15 slice table; the slice-1 plan is the pattern reference for every card task)

## Scratch-inventory facts (verified 2026-10-05, drive every card's engine mode)

`src/models/` contains real implementations only in: `linear_models.py` (`LinearRegressionScratch`, `RidgeRegressionScratch`, `LassoRegressionScratch`, `LogisticRegressionScratch`), `tree_models.py` (`Node`, `DecisionTreeClassifierScratch`, `DecisionTreeRegressorScratch`), `knn_models.py` (`KNNClassifierScratch`, `KNNRegressorScratch`). `clustering_models.py`, `dimensionality_reduction.py`, `ensemble_models.py`, `svm_models.py` are empty placeholders; no naive-bayes module exists. Therefore: **logistic regression is the only dual-engine card in this slice; all nine others are sklearn-only** (`sources=()` + `sklearn_only=True`; the renderer's code section then shows `inspect.getsource(card.fit)` per spec §9). When scratch implementations land later, each card upgrades by editing the card file only.

## Global Constraints

- Never modify or create files under `notebooks/`, `src/`, `docs/`, `examples/`, `assets/` (spec §1 Scope; G5: app/tests never reference `notebooks/` at all).
- No `__init__.py` files anywhere under `app/` or `tests/` — PEP 420 namespace packages (spec §14).
- Every random element seeded: dataset factories via `np.random.default_rng(0)` / sklearn `random_state=0`; every sklearn estimator constructed **with** `random_state=0` **when the estimator has that parameter** — `DBSCAN`, `AgglomerativeClustering`, `GaussianNB` have none; pass nothing (slice-1 KNN precedent).
- Toy datasets: ≤ 300 rows, ≤ 4 features (spec §10).
- All charts are plotly `go.Figure`; all file handling `pathlib.Path`; imports absolute `app.core...` / `app.registry...`; commands run `uv run ...` from the repo root.
- Card ids are unique kebab-case slugs; `validate()` (called by discovery and the contract suite) enforces non-empty id/title/theory, unique hyper names, defaults in bounds, select default in options, and `sources` empty ⇔ `sklearn_only` True.
- `metrics` must return `list[tuple[str, float]]` — cast every value with `float()`.
- sklearn-only cards: viz code uses `ctx.sklearn` only (`PlayContext.scratch` is `None`).
- The app imports only `sklearn`, `src.models`, stdlib, numpy, plotly. `xgboost`/`lightgbm` exist in `pyproject.toml` from earlier repo work but are **not** the app's stack (spec Tech Stack) — no card uses them.

## Decisions locked (implementer cannot change)

1. **Ridge/lasso stay inside the linear-regression card** — no split (spec slice-2 preview left this open; splitting adds no lesson).
2. **XGBoost / LightGBM / CatBoost get no cards** — not in the spec's tech stack; the boosting family is represented by sklearn's `GradientBoostingClassifier`.
3. **No dendrogram visualization** — hierarchical clustering gets scatter-by-labels + silhouette-vs-k instead (mechanical authoring priority; plotly figure_factory drift risk, spec §16).
4. **Voting-ensemble members are fixed**: `LogisticRegression(max_iter=1000)`, `DecisionTreeClassifier(max_depth=3, random_state=0)`, `KNeighborsClassifier(n_neighbors=5)` — diversity of model *classes* is the lesson; member tuning is not.

## Review Focus

Five failure modes the spec implies but no single task's tests fully exercise:

1. **In-bounds degenerate combos** — unlike slice 1, some sliders *can* reach legitimate failure states (DBSCAN `eps=0.1` → every point noise; silhouette on a single cluster). Expected: graceful metrics / no crash, page usable (spec §11). Pinned by Task 8 Step 1 (DBSCAN all-noise test) and the metrics guards in Tasks 7–8.
2. **Multi-class through the whole stack** — `blobs_noisy` has 3 classes; the scratch logistic implementation is binary-only (sigmoid). Expected: a card declares only datasets its *weakest* engine supports; the contract fit test (card × dataset × engine) fails fast otherwise. Pinned by Task 2's dataset choice + existing contract suite.
3. **Estimator nondeterminism in the new families** (RF/GB bootstrap, MLP adam) — Expected: identical predictions across identical fits. Pinned by Task 12 Step 1 (cross-card determinism test).
4. **Transform-family wiring** — PCA must flow through `Fitted.transform`, not `predict` (`engines.run` sets `kind="transform"` iff family is `"dimensionality-reduction"`). Pinned by Task 9 Step 1.
5. **Registry growth breaking dataset invariants** — the 4 new factories must stay deterministic, ≤ 300×4. Pinned automatically: `test_datasets.py` parametrizes over `dataset_ids()`, so new ids join its shape/determinism checks.

---

## File Structure

```
app/core/datasets.py                        # Task 1 — +4 factories (lin_2f, blobs_noisy, var_blobs, pca_correlated_4f)
app/registry/algorithms/linear_regression.py  # Task 1 — datasets tuple += "lin_2f" (one line)
app/registry/algorithms/logistic_regression.py  # Task 2 (dual)
app/registry/algorithms/svm.py              # Task 3 (sklearn-only)
app/registry/algorithms/random_forest.py    # Task 4 (sklearn-only)
app/registry/algorithms/gradient_boosting.py  # Task 5 (sklearn-only)
app/registry/algorithms/naive_bayes.py      # Task 6 (sklearn-only)
app/registry/algorithms/hierarchical_clustering.py  # Task 7 (sklearn-only)
app/registry/algorithms/dbscan.py           # Task 8 (sklearn-only)
app/registry/algorithms/pca.py              # Task 9 (sklearn-only)
app/registry/algorithms/neural_network.py   # Task 10 (sklearn-only)
app/registry/algorithms/voting_ensemble.py  # Task 11 (sklearn-only)
tests/app/test_datasets.py                  # Task 1 — expected id set += 4
tests/app/test_cards_contract.py            # Tasks 2–11 — one existence guard per card; Task 12 — determinism test
tests/app/test_home.py                      # Task 12 — expected card set = all 14 ids
```

## Interfaces (produced by slice 1 — every card task consumes these)

- `AlgorithmCard(...)` — construct with keyword args: `id, title, family, when_to_use, theory, sources, hypers, datasets, row_cap=None, sklearn_only=False, fit, metrics, visualizations, notes=(), grid_resolution=40`. Families in use: `"regression" | "classification" | "clustering" | "dimensionality-reduction" | "ensembles"`.
- Widget specs: `Slider(name, min, max, step, default, help="")`, `Select(name, options: tuple, default, help="")` (options may hold ints/floats/tuples), `Toggle(name, default, help="")`.
- `Data(X, y|None, note, family)` frozen; `y is None` for clustering and dimensionality-reduction datasets.
- `run(card, data, params, engine) -> Fitted` — wraps `card.fit`; `Fitted.kind == "transform"` iff `card.family == "dimensionality-reduction"`, else `"predict"`. `Fitted.predict/transform` delegate to `raw`; `Fitted.raw` exposes estimator attributes (`labels_`, `explained_variance_ratio_`, `cluster_centers_`, …).
- `PlayContext(data, params, scratch, sklearn)` frozen — viz functions receive it; sklearn-only cards get `scratch=None`.
- `labeled_scatter(X: np.ndarray, y: np.ndarray, title: str) -> go.Figure` (branches on feature count); `decision_boundary(fig, predict_fn, X, name, resolution=40) -> go.Figure` (mutates `fig`; classifiers only — needs `predict`).
- `get_dataset(id) -> Data` (KeyError on unknown), `dataset_ids() -> tuple[str, ...]`.
- `tests/app/test_cards_contract.py` parametrizes over `all_cards()` — every authored card joins all contract tests automatically; per-task RED uses a 2-line existence guard appended to that file (slice-1 pattern).
- Contract expectations for every card: `fit` succeeds on every declared dataset × applicable engine at default params; `metrics` returns ≥ 1 `(label, float)`; every viz returns a `go.Figure` at defaults on `datasets[0]`; AppTest page smoke boots without exception.

Per-card authoring cycle (used by Tasks 2–11; written out fully once in Task 2):

1. Append the card's existence guard to `tests/app/test_cards_contract.py` → run → FAIL (`AssertionError`, card not authored).
2. Create `app/registry/algorithms/<card>.py` per that task's content block.
3. `uv run pytest tests/app -v` → PASS (contract suite auto-covers the new card; expect the run to take longer — each card adds fit/viz/smoke params).
4. Commit with the task's message.

---

### Task 1: Remaining spec-§10 datasets + linear card adopts `lin_2f`

**Files:**
- Modify: `app/core/datasets.py`, `tests/app/test_datasets.py`, `app/registry/algorithms/linear_regression.py` (datasets tuple only)
- Test: `tests/app/test_datasets.py`

**Interfaces:**
- Produces: 4 new registry ids consumed by Tasks 2–11 — `lin_2f` (regression), `blobs_noisy` (classification, **3 classes**), `var_blobs` (clustering), `pca_correlated_4f` (dimensionality-reduction, `y=None`).

- [ ] **Step 1: Write the failing test** — in `tests/app/test_datasets.py`, extend the expected set in `test_all_ids_present` with `"lin_2f"`, `"blobs_noisy"`, `"var_blobs"`, `"pca_correlated_4f"`, and add:

```python
def test_dim_reduction_sets_have_no_labels():
    assert get_dataset("pca_correlated_4f").y is None
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/app/test_datasets.py -v`
Expected: FAIL — `KeyError: 'lin_2f'` (first missing id hit); the existing 8 ids still pass their parametrized checks.

- [ ] **Step 3: Implement the 4 factories in `app/core/datasets.py`** (exact values):

```python
def _lin_2f():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(160, 2))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 1 + rng.normal(0, 0.5, 160)
    return Data(X=X, y=y, family="regression",
                note="two features — both engines should recover w ≈ (2.0, −1.5)")

def _blobs_noisy():
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=2.2, random_state=0)
    return Data(X=X, y=y, family="classification",
                note="three overlapping classes — no clean boundary, watch engines disagree")

def _var_blobs():
    X, _ = make_blobs(n_samples=300, centers=[(-2, -2), (2, -2), (0, 2.5)],
                      cluster_std=[0.4, 0.4, 2.0], random_state=0)
    return Data(X=X, y=None, family="clustering",
                note="unequal cluster spreads — density and variance assumptions show")

def _pca_correlated_4f():
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.9, 0.7, 0.4],
                    [0.9, 1.0, 0.8, 0.5],
                    [0.7, 0.8, 1.0, 0.6],
                    [0.4, 0.5, 0.6, 1.0]])
    X = rng.multivariate_normal(np.zeros(4), cov, size=240)
    return Data(X=X, y=None, family="dimensionality-reduction",
                note="four correlated features — two components should capture most variance")
```

Register all four in the registry dict; in `linear_regression.py` change `datasets` to `("lin_clean_1f", "lin_noisy_1f", "lin_outliers_1f", "lin_2f")`.

- [ ] **Step 4: Run to verify GREEN**

Run: `uv run pytest tests/app/test_datasets.py tests/app/test_cards_contract.py -v`
Expected: PASS — dataset parametrization now covers 12 ids; the linear card's contract params run on `lin_2f` too (both engines must fit a 2-feature regression).

- [ ] **Step 5: Commit**

```bash
git add app/core/datasets.py tests/app/test_datasets.py app/registry/algorithms/linear_regression.py
git commit -m "feat: add remaining spec-10 toy datasets and wire lin_2f into linear card"
```

---

### Task 2: Card — Logistic Regression (dual engine) — the exemplar card cycle

**Files:**
- Create: `app/registry/algorithms/logistic_regression.py`
- Modify: `tests/app/test_cards_contract.py` (existence guard)

**Interfaces:**
- Consumes: `LogisticRegressionScratch(learning_rate=0.01, max_iter=1000, tol=1e-4)` from `src.models.linear_models` (gradient descent, binary sigmoid — verify `.fit(X, y)` against the class docstring before wiring); sklearn `LogisticRegression(C, max_iter)`.
- Produces: card id `logistic-regression`, the last dual-engine card.

Card content:
- `theory` LaTeX (required elements): sigmoid `$$\\sigma(z) = \\tfrac{1}{1 + e^{-z}}$$`, model `$$\\hat{p} = \\sigma(w^\\top x + b)$$`, log-loss `$$L = -\\tfrac{1}{n}\\sum_i \\left[y_i \\log \\hat{p}_i + (1-y_i)\\log(1-\\hat{p}_i)\\right]$$`, GD update `$$w \\leftarrow w - \\eta \\nabla_w L$$`; paragraphs: scratch learns by plain gradient descent, sklearn minimizes L2-regularized log-loss (strength `1/C`) with a quasi-Newton solver — `C` has **no** scratch counterpart, so that slider moving only one boundary is the lesson.
- `sources`: `(("src/models/linear_models.py", ("LogisticRegressionScratch",)),)`
- `hypers`:
```python
Slider("C", 0.01, 10.0, 0.01, 1.0, "inverse regularization — sklearn only, scratch has none"),
Slider("learning_rate", 0.001, 1.0, 0.001, 0.1, "scratch gradient-descent step"),
Slider("max_iter", 100, 5000, 100, 1000),
```
- `datasets`: `("moons", "circles", "lin_separable")` — binary datasets only; the scratch sigmoid cannot fit the 3-class `blobs_noisy` (Review Focus #2).
- `fit` glue:
```python
def fit(data, params, engine):
    if engine == "scratch":
        return LogisticRegressionScratch(
            learning_rate=params["learning_rate"], max_iter=params["max_iter"],
        ).fit(data.X, data.y)
    return LogisticRegression(
        C=params["C"], max_iter=params["max_iter"], random_state=0,
    ).fit(data.X, data.y)
```
- `metrics`: `("Accuracy", float(accuracy_score(data.y, fitted.predict(data.X))))`
- `visualizations` (2): **"Decision boundaries"** — the slice-1 dual pattern: `labeled_scatter` + `decision_boundary(..., card.grid_resolution)` per engine on 1×2 subplots (copy the layout block from `decision_tree.py::_boundaries`); **"Accuracy vs learning_rate"** — refits the scratch engine at 20 log-spaced learning rates in `[0.001, 1.0]` (train accuracy line) plus sklearn's accuracy as a horizontal reference line — sklearn's accuracy does not depend on the learning rate, which is the divergence made visible.
- `notes`: e.g. "drag C — only the sklearn boundary moves: the scratch implementation has no regularization", "learning_rate → 1.0: scratch gradient descent overshoots and wobbles", "max_iter = 100 on `moons`: scratch is still climbing when sklearn has converged".

- [ ] **Step 1: Write the failing existence guard** — append to `tests/app/test_cards_contract.py`:

```python
def test_logistic_regression_card_exists():
    assert any(c.id == "logistic-regression" for c in ALL)
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest tests/app/test_cards_contract.py::test_logistic_regression_card_exists -v`
Expected: FAIL — `assert any(...)` false.

- [ ] **Step 3: Implement the card** per the content block above.

- [ ] **Step 4: Run the full suite to verify GREEN**

Run: `uv run pytest tests/app -v`
Expected: PASS — the contract suite now covers `logistic-regression` (fit × 3 datasets × 2 engines, viz figures, validate, page smoke).

- [ ] **Step 5: Commit**

```bash
git add app/registry/algorithms/logistic_regression.py tests/app/test_cards_contract.py
git commit -m "feat: logistic regression card with dual-engine boundary and GD-convergence viz"
```

---

### Tasks 3–11: sklearn-only cards (same 5-step cycle as Task 2)

Each task: append its existence guard (`test_<snake>_card_exists`, same shape), watch RED, author the card module, full suite GREEN, commit. All cards: `sklearn_only=True`, `sources=()`, viz uses `ctx.sklearn` only, `float()`-cast metrics. Boundary-viz cards reuse the `decision_boundary` painter (single call — one engine) and `labeled_scatter`. Clustering cards have **no** `predict` — no boundary viz.

### Task 3: Card — SVM

- [ ] Guard: `test_svm_card_exists` → RED → implement → suite GREEN → commit `"feat: SVM card with margin and C-sensitivity playgrounds"`.

Card content:
- `theory`: hard margin `$$\\max_{w,b} \\tfrac{2}{\\|w\\|} \\;\\text{s.t.}\\; y_i(w^\\top x_i + b) \\ge 1$$`, hinge loss `$$\\max(0,\\; 1 - y_i(w^\\top x_i + b))$$`, RBF kernel `$$K(x, x') = e^{-\\gamma \\|x - x'\\|^2}$$`; paragraphs on C (margin/violation tradeoff) and γ (island-shaped regions).
- `hypers`: `Select("kernel", ("linear", "rbf", "poly"), "rbf")`, `Slider("C", 0.01, 10.0, 0.01, 1.0)`, `Select("gamma", ("scale", 0.1, 1.0, 5.0), "scale", "kernel width (rbf/poly)")`
- `datasets`: `("moons", "circles", "blobs_noisy")`
- `fit`: `SVC(kernel=params["kernel"], C=params["C"], gamma=params["gamma"], random_state=0).fit(data.X, data.y)`
- `metrics`: Accuracy.
- `visualizations` (2): single decision boundary (grid_resolution); **"Accuracy vs C"** — log-spaced sweep 0.01→10, train + test lines (refits internally).
- `notes`: "C small → wide margin, more train errors tolerated", "γ=5 on `moons` → islands around single points", "linear kernel cannot bend around `circles` — geometry beats capacity".

### Task 4: Card — Random Forest

- [ ] Guard: `test_random_forest_card_exists` → RED → implement → suite GREEN → commit `"feat: random forest card with bagging boundary and n_estimators curve"`.

Card content:
- `theory`: bootstrap aggregation `$$\\hat{f}(x) = \\tfrac{1}{B}\\sum_{b=1}^{B} T_b(x)$$`, variance of the mean `$$\\mathrm{Var}(\\bar{f}) = \\rho\\sigma^2 + \\tfrac{1-\\rho}{B}\\sigma^2$$` (ρ = tree correlation), feature subsampling; paragraph contrasting bagging (variance ↓) with the single overfitting tree on the Decision Tree page.
- `hypers`: `Slider("n_estimators", 5, 200, 5, 100)`, `Slider("max_depth", 1, 20, 1, 5, "cap each tree; forest averages the rest")`
- `datasets`: `("moons", "circles", "blobs_noisy")`
- `fit`: `RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=0).fit(data.X, data.y)`
- `metrics`: Accuracy.
- `visualizations` (2): boundary; **"Accuracy vs n_estimators"** — 5→200 step 5, train + test lines.
- `notes`: "one deep tree memorizes (`moons`) — the forest averages the memorization away", "depth 2 + 200 trees often beats depth 20 + 5 trees", "watch the boundary straighten as B grows".

### Task 5: Card — Gradient Boosting

- [ ] Guard: `test_gradient_boosting_card_exists` → RED → implement → suite GREEN → commit `"feat: gradient boosting card with stagewise-fit playgrounds"`.

Card content:
- `theory`: stagewise additive model `$$F_m(x) = F_{m-1}(x) + \\nu\\, h_m(x)$$` where each `h_m` fits the negative gradient (residuals) of the log-loss; shrinkage `$$\\nu$$` (learning_rate); paragraphs on boosting reducing *bias* stepwise vs bagging reducing *variance*, and why more trees eventually overfit.
- `hypers`: `Slider("n_estimators", 5, 200, 5, 100)`, `Slider("learning_rate", 0.01, 1.0, 0.01, 0.1, "shrinkage per stage")`, `Slider("max_depth", 1, 5, 1, 3)`
- `datasets`: `("moons", "blobs_noisy")`
- `fit`: `GradientBoostingClassifier(n_estimators=..., learning_rate=..., max_depth=..., random_state=0).fit(data.X, data.y)`
- `metrics`: Accuracy.
- `visualizations` (2): boundary; **"Accuracy vs n_estimators"** — 5→200 step 5 at the current learning_rate, train + test: train climbs to 1.0 while test peaks then decays — the boosting-overfit signature.
- `notes`: "learning_rate 1.0 + 200 trees: test accuracy decays — the classic boosting overfit", "low learning_rate needs more trees but lands steadier", "compare with the Random Forest page: boosting vs bagging".

### Task 6: Card — Naive Bayes

- [ ] Guard: `test_naive_bayes_card_exists` → RED → implement → suite GREEN → commit `"feat: naive bayes card with gaussian-coverage playgrounds"`.

Card content:
- `theory`: Bayes rule `$$P(y \\mid x) \\propto P(y) \\prod_j P(x_j \\mid y)$$`, Gaussian class-conditional `$$P(x_j \\mid y) = \\tfrac{1}{\\sqrt{2\\pi\\sigma_{jy}^2}} e^{-\\frac{(x_j - \\mu_{jy})^2}{2\\sigma_{jy}^2}}$$`, variance smoothing `$$\\sigma^2 + \\epsilon$$`; paragraphs on the independence assumption (ellipse contours aligned to axes) and what ε protects against.
- `hypers`: `Select("var_smoothing", (1e-9, 1e-7, 1e-5, 1e-3), 1e-9, "variance floor")`
- `datasets`: `("moons", "blobs_noisy")`
- `fit`: `GaussianNB(var_smoothing=params["var_smoothing"]).fit(data.X, data.y)` — no `random_state` param; pass nothing.
- `metrics`: Accuracy.
- `visualizations` (2): boundary; **"Confusion matrix"** — `go.Heatmap` of `sklearn.metrics.confusion_matrix(data.y, pred)` with class-index axes.
- `notes`: "axis-aligned ellipse decision regions = per-feature Gaussians", "on `moons` the crescents' features are correlated — the independence assumption bends the boundary", "smoothing only matters at 1e-3: watch regions merge".

### Task 7: Card — Hierarchical Clustering

- [ ] Guard: `test_hierarchical_clustering_card_exists` → RED → implement → suite GREEN → commit `"feat: hierarchical clustering card with linkage and k-choice playgrounds"`.

Card content:
- `theory`: agglomerative merging; linkage distances (ward / average / complete / single) defined as the inter-cluster distance used at each merge `$$d(A, B)$$` per linkage; paragraph: the model has **no `predict`** — clustering is transductive (labels exist only for fitted rows), and choosing k = cutting the merge hierarchy.
- `hypers`: `Select("linkage", ("ward", "average", "complete", "single"), "ward")`, `Slider("n_clusters", 2, 10, 1, 4)`
- `datasets`: `("kmeans_4blobs", "var_blobs")`
- `fit`: `AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params["linkage"]).fit(data.X)` — no `random_state`; pass nothing.
- `metrics` (guarded): `("Clusters", float(len(set(labels))))`; `("Silhouette", float(silhouette_score(data.X, labels)))` **only if** `len(set(labels)) >= 2`.
- `visualizations` (2): scatter colored by `labels_` (no centroids); **"Silhouette vs k"** — refit k = 2..8 at the current linkage, silhouette line + marker at current k.
- `notes`: "single linkage chains across `var_blobs`' sparse regions — ward refuses to", "ward ≈ K-Means on convex blobs: compare pages", "the silhouette peak marks the k the merge hierarchy wants".

### Task 8: Card — DBSCAN (+ the slice's in-bounds degenerate pin)

- [ ] Guard: `test_dbscan_card_exists` → RED → implement → suite GREEN → commit `"feat: DBSCAN card with density playgrounds and noise handling"` (`git add` the card module, `tests/app/test_cards_contract.py`, and `tests/app/test_dbscan_degenerate.py`).

Card content:
- `theory`: core point `$$|N_\\epsilon(x)| \\ge \\text{min\\_samples}$$`, density-reachability, clusters = connected core components, noise label `$$-1$$`; paragraph: no k needed — eps and min_samples *are* the model.
- `hypers`: `Slider("eps", 0.1, 3.0, 0.05, 0.5, "neighborhood radius")`, `Slider("min_samples", 2, 20, 1, 5, "core-point threshold")`
- `datasets`: `("kmeans_rings", "var_blobs")`
- `fit`: `DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit(data.X)` — no `random_state`; pass nothing.
- `metrics` (all guarded — Review Focus #1): `labels = fitted.raw.labels_`; `n_clusters = len(set(labels)) - (1 if -1 in labels else 0)`; return `("Clusters", float(n_clusters))`, `("Noise", float((labels == -1).mean()))`, and `("Silhouette", float(silhouette_score(data.X, labels)))` **only if** `n_clusters >= 2`.
- `visualizations` (2): scatter by label with noise as gray × markers; **"Clusters vs eps"** — sweep eps 0.1→3.0 (step 0.05) at current min_samples, count line + marker at current eps.
- `notes`: "on `kmeans_rings`, eps ≈ 0.5 finds both rings — the failure K-Means cannot fix", "eps = 0.1: everything is noise, zero clusters — the page must survive it (it's the pinned degenerate case)", "eps = 3.0: one blob eats the rings".

- [ ] **Degenerate pin (goes in `tests/app/test_dbscan_degenerate.py`, write it in Step 1 together with the guard):**

```python
def test_all_noise_combo_degrades_gracefully():
    card = get_card("dbscan")
    params = {h.name: h.default for h in card.hypers} | {"eps": 0.1}
    fitted = run(card, get_dataset("kmeans_rings"), params, "sklearn")
    values = dict(card.metrics(fitted, get_dataset("kmeans_rings")))
    assert values["Clusters"] == 0.0  # every point is noise; no crash, no NaN
```

Expected: `eps=0.1` on `kmeans_rings` (300 points spread over ~44 units of arc) yields no core points at `min_samples=5` → all noise; metrics return `Clusters = 0.0` without raising. If sklearn's geometry disagrees and a cluster forms, lower to `eps=0.05` — the pin is "smallest eps ⇒ 0 clusters, no exception", not the specific 0.1.

### Task 9: Card — PCA (transform-family wiring)

- [ ] Guard: `test_pca_card_exists` → RED → implement → suite GREEN → commit `"feat: PCA card with projection and explained-variance playgrounds"` (`git add` the card module, `tests/app/test_cards_contract.py`, and `tests/app/test_pca_card.py`).

Card content:
- `theory`: maximum-variance projection; covariance eigendecomposition `$$\\Sigma v = \\lambda v$$`, PC variance `$$\\lambda_i$$`, explained-variance ratio `$$\\lambda_i / \\sum_j \\lambda_j$$`; paragraphs: components are *ordered by variance*, the slider only chooses how many to keep; compression = keeping few components with most of the ratio.
- `hypers`: `Slider("n_components", 1, 4, 1, 2)`
- `datasets`: `("pca_correlated_4f",)` — family `"dimensionality-reduction"`, so `run()` returns `Fitted(kind="transform")`.
- `fit`: `PCA(n_components=params["n_components"], random_state=0).fit(data.X)`
- `metrics`: `("Explained variance", float(fitted.raw.explained_variance_ratio_.sum()))` — the sum *for the kept components* is the lesson (k=2 on this dataset should be visibly less than 4.0 but more than half).
- `visualizations` (2): **"Projection (PC1 × PC2)"** — `proj = fitted.transform(data.X)`; x = `proj[:, 0]`, y = `proj[:, 1]` if `proj.shape[1] > 1` else zeros (guard k=1); **"Explained variance ratio"** — refit a full 4-component PCA internally, per-component ratio bars + cumulative line, marker at the current k.
- `notes`: "k=2 keeps most of four features' information — that is compression", "drag k to 1: the projection collapses onto one axis", "the bar heights never move — ratios are dataset properties, the slider only moves the marker".

- [ ] **Transform pin (in `tests/app/test_pca_card.py`, written in Step 1 with the guard):**

```python
def test_pca_flows_through_transform():
    card = get_card("pca")
    data = get_dataset("pca_correlated_4f")
    fitted = run(card, data, {h.name: h.default for h in card.hypers}, "sklearn")
    assert fitted.kind == "transform"                      # Review Focus #4
    proj = fitted.transform(data.X)
    assert proj.shape == (240, 2)
    assert fitted.raw.explained_variance_ratio_[0] > fitted.raw.explained_variance_ratio_[1]
```

### Task 10: Card — Neural Network (basics)

- [ ] Guard: `test_neural_network_card_exists` → RED → implement → suite GREEN → commit `"feat: neural network card with capacity and regularization playgrounds"`.

Card content:
- `theory`: layer map `$$h^{(l)} = \\phi(W^{(l)} h^{(l-1)} + b^{(l)})$$`, output `$$\\hat{p} = \\mathrm{softmax}(W^{(L)} h^{(L-1)} + b^{(L)})$$`, L2 penalty `$$+ \\alpha \\|W\\|_2^2$$`; paragraphs: capacity = width/depth, adam optimizer, ConvergenceWarning at small `max_iter` is *normal* (undertraining, not an app error).
- `hypers`: `Select("hidden", ((4,), (16,), (32,)), (16,), "one hidden layer width")`, `Select("activation", ("relu", "tanh", "logistic"), "relu")`, `Slider("alpha", 0.0001, 1.0, 0.0001, 0.01, "L2 strength")`, `Slider("max_iter", 50, 500, 50, 200)`
- `datasets`: `("moons", "circles", "blobs_noisy")`
- `fit`: `MLPClassifier(hidden_layer_sizes=params["hidden"], activation=params["activation"], alpha=params["alpha"], max_iter=params["max_iter"], random_state=0).fit(data.X, data.y)`
- `metrics`: Accuracy.
- `visualizations` (2): boundary; **"Accuracy vs alpha"** — log-spaced sweep 0.0001→1.0, train + test lines (regularization straightens the boundary).
- `notes`: "max_iter = 50 → ConvergenceWarning in the console: the network is simply undertrained", "alpha → 1.0: the boundary straightens toward linear", "(4,) vs (32,) on `moons`: capacity to bend".

### Task 11: Card — Voting Ensemble

- [ ] Guard: `test_voting_ensemble_card_exists` → RED → implement → suite GREEN → commit `"feat: voting ensemble card with member-comparison playgrounds"`.

Card content:
- `theory`: hard vote = majority of predicted labels; soft vote = argmax of averaged probabilities `$$\\hat{p} = \\tfrac{1}{M} \\sum_m p_m(y \\mid x)$$`; paragraph: ensembles pay off when member *errors decorrelate* — hence one member of each bias family (linear / tree / instance-based).
- `hypers`: `Select("voting", ("hard", "soft"), "soft")`
- `datasets`: `("moons", "blobs_noisy")`
- `fit` glue (members fixed — Decisions §4):
```python
def fit(data, params, engine):
    members = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
    return VotingClassifier(estimators=members, voting=params["voting"]).fit(data.X, data.y)
```
- `metrics`: Accuracy.
- `visualizations` (2): boundary of the ensemble; **"Member vs ensemble accuracy"** — refit the 3 members + the ensemble internally, `go.Bar` with 4 bars.
- `notes`: "hard vs soft on `moons`: averaging probabilities changes close calls", "the depth-3 tree member underfits — the ensemble recovers some of it", "the ensemble can lose to its best member: diversity is not free".

---

### Task 12: Full gates + manual acceptance

**Files:**
- Modify: `tests/app/test_home.py`, `tests/app/test_cards_contract.py`

- [ ] **Step 1: Update the completeness gate + add the determinism gate**

In `tests/app/test_home.py`, rename `test_slice1_cards_all_present` → `test_all_cards_present` and set the expected set to all 14 ids: `"linear-regression"`, `"decision-tree"`, `"knn"`, `"kmeans"`, `"logistic-regression"`, `"svm"`, `"random-forest"`, `"gradient-boosting"`, `"naive-bayes"`, `"hierarchical-clustering"`, `"dbscan"`, `"pca"`, `"neural-network"`, `"voting-ensemble"`.

Append to `tests/app/test_cards_contract.py` (Review Focus #3):

```python
@pytest.mark.parametrize("card", ALL, ids=lambda c: c.id)
def test_fit_is_deterministic(card):
    # sklearn param drift / nondeterminism: identical fits must agree exactly
    data = get_dataset(card.datasets[0])
    params = {h.name: h.default for h in card.hypers}
    a = run(card, data, params, "sklearn")
    b = run(card, data, params, "sklearn")
    out = lambda f: (f.transform(data.X) if f.kind == "transform" else f.predict(data.X))
    assert np.array_equal(out(a), out(b))
```

(`import numpy as np` at the file top if absent; all 14 cards run on the sklearn engine here, and the kind-aware call keeps the PCA card — transform family, no `predict` — in scope.)

- [ ] **Step 2: Run the whole suite**

Run: `uv run pytest tests/app -v`
Expected: PASS — 14 cards × (fit × datasets × engines, viz, validate, smoke) + determinism; zero skips except any genuinely structural ones; allow a longer wall time (~4–6 min: 14 AppTest page boots with refit-heavy viz).

- [ ] **Step 3: Manual acceptance run** (human step — the slice's exit proof per spec §15; the human partner runs the app and reports back; no browser automation)

```bash
uv run streamlit run app/Home.py
```

Checklist:
1. Home lists all 14 algorithms grouped by family (regression / classification / clustering / dimensionality-reduction / ensembles).
2. Logistic Regression: scratch vs sklearn boundaries both render; C moves only the sklearn boundary.
3. SVM: rbf γ islands on `moons`; linear kernel fails `circles`.
4. Random Forest + Gradient Boosting: n_estimators curves show bagging smoothing vs boosting decay.
5. Hierarchical/DBSCAN: `kmeans_rings` under DBSCAN at eps≈0.5 finds both rings; eps=0.1 shows 0 clusters gracefully.
6. PCA: k=2 keeps most variance; projection updates with k.
7. `notebooks/` untouched: `git status` shows no changes under `notebooks/`, `src/`, `docs/`, `examples/`.

- [ ] **Step 4: Final commit**

```bash
git add tests/app/test_home.py tests/app/test_cards_contract.py
git commit -m "test: gate home navigation and determinism across all slice-2 cards"
```

---

## Slice 3 preview (not in this plan)

Optional per spec §15: algorithm-comparison mode (one metric table across all cards), quick-reference page. Also the natural home for any scratch implementations that land in `src/models/` — each upgrades its card to dual-engine by editing the card file only.
