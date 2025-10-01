# ML Fundamentals Revisited

As an AI engineer working primarily with LLM providers (OpenAI, Anthropic, Google), this repository serves as my **personal reminder handbook** for classical ML algorithms. It covers theory, implementation, and practical applications.

## 🚀 Why This Repository?

- **Complete Reference**: Each algorithm with theory, assumptions, and use cases
- **Dual Implementation**: NumPy from-scratch AND scikit-learn implementations
- **Production-Ready**: Complete hyperparameter tuning guides
- **Visual Learning**: Diagrams and visualizations for every concept
- **Bridge to Modern AI**: Understanding when classical ML beats LLMs

## 📚 Contents

### Supervised Learning

- **Regression**: Linear Regression, Ridge, Lasso, Elastic Net
- **Classification**: Logistic Regression, Naive Bayes, KNN
- **Tree-Based**: Decision Trees, Random Forest, Gradient Boosting
- **Advanced Ensemble**: XGBoost, LightGBM, CatBoost
- **Support Vector Machines**: SVM, SVR

### Unsupervised Learning

- **Clustering**: K-Means, Hierarchical, DBSCAN
- **Dimensionality Reduction**: PCA, t-SNE

### Other Topics

- Ensemble Methods
- Neural Network Basics
- Hyperparameter Tuning Strategies

## 🗂️ Repository Structure

- `notebooks/`: Jupyter notebooks with complete explanations
- `src/`: Clean Python implementations for import
- `docs/`: Conceptual guides and comparisons
- `assets/`: Visualizations and diagrams
- `examples/`: Real-world use case implementations

## 🛠️ Setup

This project uses `uv` for fast, reliable package management.

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-fundamentals-revisited.git
cd ml-fundamentals-revisited

# Install dependencies using uv
uv add --active numpy pandas scipy scikit-learn xgboost lightgbm catboost matplotlib seaborn plotly jupyter notebook ipywidgets tqdm joblib imbalanced-learn shap

# Launch Jupyter
jupyter notebook
```

### Alternative: Install from requirements.txt

```bash
# If you prefer to use requirements.txt
uv pip install -r requirements.txt
```

## 📖 How to Use This Repository

1. **Quick Lookup**: Check `QUICK_REFERENCE.md` for algorithm selection
2. **Deep Dive**: Open relevant notebook in `notebooks/`
3. **Implementation**: Import from `src/` for your projects
4. **Comparison**: Read `docs/model_comparison.md`

## 🎓 Learning Path

**Beginner**: 01 → 02 → 03 → 04 → 12
**Intermediate**: 05 → 08 → 09 → 15 → 16
**Advanced**: 10 → 11 → 17

## 📝 License

MIT License - Feel free to use for learning

---

**Note**: This repository focuses on tabular/structured data. For unstructured data (text, images), consider deep learning approaches or modern LLMs.
