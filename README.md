# Traditional ML Concepts

A personal study repository for classical machine learning, in three layers:

1. **Notebooks** (`notebooks/`) — seventeen worked notebooks covering theory, assumptions, hyperparameter behavior, and practical use for each algorithm.
2. **From-scratch implementations** (`src/models/`) — clean NumPy implementations of the core algorithms, written to be read rather than shipped.
3. **An interactive Streamlit app** (`app/`) — a study companion that fits every algorithm with my implementation and scikit-learn side by side, so the agreement — and the divergence — between the two engines is always visible.

I work primarily with LLM providers (OpenAI, Anthropic, Google). This repository is my standing reminder of the classical layer underneath: what each algorithm assumes, where it fails, and which hyperparameter actually moves the needle.

## The Streamlit companion

The app turns each algorithm into a single "card": a self-contained declaration of theory, hyperparameters, datasets, and fit glue. The generic renderer turns any card into a full study page.

```bash
uv sync
uv run streamlit run app/Home.py
```

Every algorithm page has the same structure:

- **Theory** — distilled mathematics with LaTeX, including the assumptions and failure modes.
- **The code** — for dual-engine cards, the from-scratch implementation read live from `src/models/` at render time; for the rest, the scikit-learn setup glue.
- **Playground** — twelve deterministic toy datasets, hyperparameter sliders, and plotly visualizations. Slider moves re-run the fits, not the page.
- **Metrics row** — both engines' headline numbers side by side on the current fit.

Two cross-cutting pages sit alongside the cards in the sidebar:

- **Compare** — one metric table per family, every algorithm fit at its declared defaults on a shared dataset, engines in adjacent columns. A card that does not declare the selected dataset shows `n/a (not declared)` — a gap is information, never a silent skip.
- **Quick Reference** — one dense table over all cards: family, engines, datasets, hyperparameter defaults, when to use.

### Engine status

Four cards currently run both engines; the rest run scikit-learn alone until their scratch twin lands in `src/models/`. Upgrading a card is a one-file edit — the app discovers it automatically.

| Algorithm | Family | Engines |
|---|---|---|
| Linear Regression (linear / ridge / lasso) | regression | scratch + sklearn |
| Logistic Regression | classification | scratch + sklearn |
| Decision Tree | classification | scratch + sklearn |
| K-Nearest Neighbors | classification | scratch + sklearn |
| Support Vector Machine | classification | sklearn only |
| Naive Bayes (Gaussian) | classification | sklearn only |
| Neural Network (basics) | classification | sklearn only |
| K-Means | clustering | sklearn only |
| Hierarchical Clustering | clustering | sklearn only |
| DBSCAN | clustering | sklearn only |
| PCA | dimensionality reduction | sklearn only |
| Random Forest | ensembles | sklearn only |
| Gradient Boosting | ensembles | sklearn only |
| Voting Ensemble | ensembles | sklearn only |

## Notebooks

Numbered in a intended reading order:

- **Linear models** — `01` linear regression
- **Classification** — `02` logistic regression, `03` decision trees, `05` SVM, `06` naive bayes, `07` KNN
- **Tree ensembles** — `04` random forest, `08` gradient boosting, `16` ensemble methods
- **Boosting libraries** — `09` XGBoost, `10` LightGBM, `11` CatBoost
- **Unsupervised** — `12` K-Means, `13` hierarchical clustering, `14` DBSCAN, `15` PCA
- **Neural networks** — `17` neural network basics

## From-scratch implementations

`src/models/` currently holds:

- `linear_models.py` — `LinearRegressionScratch`, `RidgeRegressionScratch`, `LassoRegressionScratch`, `LogisticRegressionScratch`
- `tree_models.py` — `DecisionTreeClassifierScratch`, `DecisionTreeRegressorScratch`
- `knn_models.py` — `KNNClassifierScratch`, `KNNRegressorScratch`

`clustering_models.py`, `svm_models.py`, `ensemble_models.py`, and `dimensionality_reduction.py` are placeholders reserved for the scratch twins of the sklearn-only cards above. `src/utils/` carries shared evaluation, preprocessing, hyperparameter-tuning, and visualization helpers.

## Concept guides

`other/` holds the cross-algorithm reference notes:

- `when_to_use_what.md` and `by_task_type.md` — algorithm selection by problem shape
- `model_comparison.md` — head-to-head trade-offs
- `hyperparameter_guide.md` — tuning strategies per family
- `assumptions_and_requirements.md` — the assumptions each algorithm makes
- `ml_vs_llm.md` — where classical ML still beats the modern stack

`docs/superpowers/` holds the design specs and slice plans for the Streamlit app.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mcikalmerdeka/traditional-ml-concepts.git
cd traditional-ml-concepts
uv sync
```

Then:

```bash
uv run streamlit run app/Home.py   # the study companion
uv run jupyter notebook            # the notebooks
```

Without uv, `pip install -r requirements.txt` works against the same dependency set.

## Tests

The app layer has a contract and smoke suite:

```bash
uv run pytest tests/app
```

153 passed, 10 skipped at merge. The skips are the scratch-engine contract checks for sklearn-only cards. Everything is deterministic — fixed seeds, no network — and a broken card fails in seconds at discovery rather than mid-session.

## Scope

This repository focuses on tabular and structured data with the classical stack. For unstructured data (text, images), deep learning or modern LLM approaches are the better tool — `other/ml_vs_llm.md` notes where classical ML is still the right call.

## License

MIT — see [LICENSE](LICENSE).
