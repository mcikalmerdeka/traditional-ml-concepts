"""
Behavioral tests for the from-scratch Random Forest (src/models/ensemble_models.py).

Each test targets one observable behavior of the public API:
fit/predict shapes, class handling, probability estimates, feature
importances, OOB estimation, hyperparameter validation, and determinism.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.models.ensemble_models import (
    RandomForestClassifierScratch,
    RandomForestRegressorScratch,
)


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def reg_data():
    X, y = make_regression(
        n_samples=300,
        n_features=8,
        n_informative=5,
        noise=10,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


# --- Classification -------------------------------------------------------


def test_classifier_predict_shape_and_labels(clf_data):
    X_train, X_test, y_train, y_test = clf_data
    rf = RandomForestClassifierScratch(n_estimators=15, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset(set(rf.classes_))


def test_classifier_classes_are_sorted_unique(clf_data):
    X_train, _ = clf_data[0], None
    y_train = clf_data[2]
    rf = RandomForestClassifierScratch(n_estimators=5, random_state=0)
    rf.fit(X_train, y_train)
    assert np.array_equal(rf.classes_, np.unique(y_train))


def test_classifier_handles_non_contiguous_labels():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 4)
    y = np.where(X[:, 0] > 0, 10, 20)  # labels 10/20, not 0/1
    rf = RandomForestClassifierScratch(n_estimators=10, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert set(np.unique(preds)) == {10, 20}


def test_classifier_accuracy_on_easy_problem(clf_data):
    X_train, X_test, y_train, y_test = clf_data
    rf = RandomForestClassifierScratch(
        n_estimators=25, max_depth=8, random_state=42
    )
    rf.fit(X_train, y_train)
    assert rf.score(X_test, y_test) > 0.8


def test_predict_proba_rows_sum_to_one(clf_data):
    X_train, X_test, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(n_estimators=15, random_state=42)
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(rf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_feature_importances_are_normalized_non_negative(clf_data):
    X_train, _, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(n_estimators=25, random_state=42)
    rf.fit(X_train, y_train)
    imp = rf.feature_importances_
    assert imp.shape == (X_train.shape[1],)
    assert (imp >= 0).all()
    assert np.isclose(imp.sum(), 1.0)


def test_informative_features_dominate_importances(clf_data):
    X_train, _, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(n_estimators=40, random_state=42)
    rf.fit(X_train, y_train)
    imp = rf.feature_importances_
    informative_mean = imp[:4].mean()  # first 4 are informative
    noise_mean = imp[5:].mean()  # remaining are noise
    assert informative_mean > noise_mean


def test_oob_score_sane_for_classifier(clf_data):
    X_train, _, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(
        n_estimators=25, oob_score=True, random_state=42
    )
    rf.fit(X_train, y_train)
    assert 0.0 < rf.oob_score_ <= 1.0


def test_oob_decision_function_shape(clf_data):
    X_train, _, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(
        n_estimators=15, oob_score=True, random_state=42
    )
    rf.fit(X_train, y_train)
    assert rf.oob_decision_function_.shape == (len(X_train), len(rf.classes_))


def test_same_seed_gives_identical_predictions(clf_data):
    X_train, X_test, y_train, _ = clf_data
    rf1 = RandomForestClassifierScratch(n_estimators=5, random_state=7)
    rf2 = RandomForestClassifierScratch(n_estimators=5, random_state=7)
    rf1.fit(X_train, y_train)
    rf2.fit(X_train, y_train)
    assert np.array_equal(rf1.predict(X_test), rf2.predict(X_test))


def test_different_seed_gives_different_predictions(clf_data):
    X_train, X_test, y_train, _ = clf_data
    rf1 = RandomForestClassifierScratch(n_estimators=3, random_state=1)
    rf2 = RandomForestClassifierScratch(n_estimators=3, random_state=2)
    rf1.fit(X_train, y_train)
    rf2.fit(X_train, y_train)
    assert (rf1.predict(X_test) != rf2.predict(X_test)).any()


def test_bootstrap_false_still_fits_and_predicts(clf_data):
    X_train, X_test, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(
        n_estimators=10, bootstrap=False, random_state=42
    )
    rf.fit(X_train, y_train)
    assert rf.predict(X_test).shape == (len(X_test),)


def test_oob_score_requires_bootstrap(clf_data):
    X_train, _, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(
        n_estimators=5, bootstrap=False, oob_score=True, random_state=42
    )
    with pytest.raises(ValueError, match="bootstrap"):
        rf.fit(X_train, y_train)


def test_invalid_criterion_raises(clf_data):
    X_train, _, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(n_estimators=3, criterion="bogus")
    with pytest.raises(ValueError, match="criterion"):
        rf.fit(X_train, y_train)


@pytest.mark.parametrize("mf", ["sqrt", "log2", 3, 0.5, None])
def test_max_features_variants_fit(clf_data, mf):
    X_train, X_test, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(
        n_estimators=5, max_features=mf, random_state=42
    )
    rf.fit(X_train, y_train)
    assert rf.predict(X_test).shape == (len(X_test),)


@pytest.mark.parametrize("mf", ["bogus", 0, -1, 99, 1.5, -0.5])
def test_invalid_max_features_raises(clf_data, mf):
    X_train, _, y_train, _ = clf_data
    rf = RandomForestClassifierScratch(n_estimators=3, max_features=mf)
    with pytest.raises(ValueError, match="max_features"):
        rf.fit(X_train, y_train)


# --- Regression -----------------------------------------------------------


def test_regressor_predict_shape_and_r2(reg_data):
    X_train, X_test, y_train, y_test = reg_data
    rf = RandomForestRegressorScratch(
        n_estimators=25, max_depth=8, random_state=42
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert preds.shape == y_test.shape
    assert rf.score(X_test, y_test) > 0.8


def test_regressor_oob_score_sane(reg_data):
    X_train, _, y_train, _ = reg_data
    rf = RandomForestRegressorScratch(
        n_estimators=25, oob_score=True, random_state=42
    )
    rf.fit(X_train, y_train)
    assert np.isfinite(rf.oob_score_)
    assert rf.oob_score_ > 0.5


def test_regressor_same_seed_deterministic(reg_data):
    X_train, X_test, y_train, _ = reg_data
    rf1 = RandomForestRegressorScratch(n_estimators=5, random_state=7)
    rf2 = RandomForestRegressorScratch(n_estimators=5, random_state=7)
    rf1.fit(X_train, y_train)
    rf2.fit(X_train, y_train)
    assert np.array_equal(rf1.predict(X_test), rf2.predict(X_test))
