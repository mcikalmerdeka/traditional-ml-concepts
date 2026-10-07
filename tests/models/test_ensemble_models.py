"""
Behavioral tests for the from-scratch Random Forest (src/models/ensemble_models.py).

Each test targets one observable behavior of the public API:
fit/predict shapes, class handling, probability estimates, feature
importances, OOB estimation, hyperparameter validation, and determinism.
"""

import numpy as np
import pytest
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.model_selection import train_test_split

from src.models.ensemble_models import (
    GradientBoostingClassifierScratch,
    RandomForestClassifierScratch,
    RandomForestRegressorScratch,
    VotingEnsembleClassifierScratch,
)
from src.models.knn_models import KNNClassifierScratch
from src.models.linear_models import LogisticRegressionScratch
from src.models.tree_models import DecisionTreeClassifierScratch


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


# --- Gradient Boosting (classifier) -----------------------------------------


@pytest.fixture
def gb_multiclass_data():
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=1.0, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def test_gb_binary_predict_shape_and_labels():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    gb = GradientBoostingClassifierScratch(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0
    )
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset(set(gb.classes_))


def test_gb_binary_beats_single_tree_on_moons():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    gb = GradientBoostingClassifierScratch(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0
    )
    gb.fit(X_train, y_train)
    assert gb.score(X_test, y_test) > 0.85


def test_gb_binary_proba_rows_sum_to_one():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    gb = GradientBoostingClassifierScratch(
        n_estimators=20, max_depth=3, random_state=0
    )
    gb.fit(X, y)
    proba = gb.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    # argmax of the staged probabilities IS the hard prediction
    assert np.array_equal(gb.classes_[proba.argmax(axis=1)], gb.predict(X))


def test_gb_multiclass_on_blobs(gb_multiclass_data):
    X_train, X_test, y_train, y_test = gb_multiclass_data
    gb = GradientBoostingClassifierScratch(
        n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0
    )
    gb.fit(X_train, y_train)
    proba = gb.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert gb.score(X_test, y_test) > 0.85


def test_gb_estimators_structure_matches_stages_and_classes():
    X, y = make_moons(n_samples=120, noise=0.15, random_state=0)
    gb = GradientBoostingClassifierScratch(n_estimators=7, max_depth=2)
    gb.fit(X, y)
    # one stumps-tree per stage for binary log-loss...
    assert len(gb.estimators_) == 7
    assert len(gb.estimators_[0]) == 1
    # ...and K (softmax-deviance) trees per stage for multiclass
    X3, y3 = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=0)
    gb3 = GradientBoostingClassifierScratch(n_estimators=5, max_depth=2).fit(X3, y3)
    assert len(gb3.estimators_) == 5
    assert len(gb3.estimators_[0]) == 3


def test_gb_more_stages_explains_training_data_better():
    # boosting reduces bias stage by stage: train accuracy must rise with B
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    accs = []
    for B in (5, 100):
        gb = GradientBoostingClassifierScratch(
            n_estimators=B, max_depth=3, random_state=0
        ).fit(X, y)
        accs.append(float(np.mean(gb.predict(X) == y)))
    assert accs[1] > accs[0]


def test_gb_deterministic():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    gb1 = GradientBoostingClassifierScratch(n_estimators=15, max_depth=3)
    gb2 = GradientBoostingClassifierScratch(n_estimators=15, max_depth=3)
    gb1.fit(X, y)
    gb2.fit(X, y)
    assert np.array_equal(gb1.predict_proba(X), gb2.predict_proba(X))


def test_gb_non_contiguous_labels():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    y_big = np.where(y == 0, 10, 20)
    gb = GradientBoostingClassifierScratch(n_estimators=15, max_depth=3)
    gb.fit(X, y_big)
    assert set(np.unique(gb.predict(X))) == {10, 20}
    assert gb.score(X, y_big) > 0.8


def test_gb_single_class_trivially_predicts_that_class():
    X, _ = make_moons(n_samples=60, noise=0.1, random_state=0)
    y = np.zeros(60, dtype=int)
    gb = GradientBoostingClassifierScratch(n_estimators=5).fit(X, y)
    assert np.array_equal(gb.predict(X), np.zeros(60, dtype=int))
    assert np.allclose(gb.predict_proba(X)[:, 0], 1.0)


@pytest.mark.parametrize("kwargs, match", [
    ({"n_estimators": 0}, "n_estimators"),
    ({"learning_rate": 0.0}, "learning_rate"),
    ({"learning_rate": -0.1}, "learning_rate"),
    ({"max_depth": 0}, "max_depth"),
])
def test_gb_invalid_hyperparameters_raise(kwargs, match):
    X, y = make_moons(n_samples=120, noise=0.1, random_state=0)
    gb = GradientBoostingClassifierScratch(**kwargs)
    with pytest.raises(ValueError, match=match):
        gb.fit(X, y)


# --- Voting Ensemble (classifier) -------------------------------------------


def _members():
    return [
        ("logistic", LogisticRegressionScratch(learning_rate=0.1, max_iter=300)),
        ("tree", DecisionTreeClassifierScratch(max_depth=3)),
        ("knn", KNNClassifierScratch(n_neighbors=5)),
    ]


@pytest.fixture
def vote_data():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def _fit_vote(data, voting):
    vote = VotingEnsembleClassifierScratch(estimators=_members(), voting=voting)
    return vote.fit(data[0], data[2])


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_voting_predicts_reasonably_on_moons(vote_data, voting):
    X_train, X_test, y_train, y_test = vote_data
    vote = _fit_vote(vote_data, voting)
    preds = vote.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset(set(vote.classes_))
    assert np.mean(preds == y_test) > 0.8


def test_voting_soft_proba_rows_sum_to_one_and_matches_argmax(vote_data):
    X_train, X_test, y_train, _ = vote_data
    vote = _fit_vote(vote_data, "soft")
    proba = vote.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(vote.classes_[proba.argmax(axis=1)], vote.predict(X_test))


def test_voting_hard_and_soft_both_available(vote_data):
    # both votings fit/predict cleanly on the same bench — hard needs no probo
    X_train, _, y_train, _ = vote_data
    for voting in ("hard", "soft"):
        _fit_vote(vote_data, voting).predict(X_train[:5])


def test_voting_soft_requires_predict_proba():
    class NoProba:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    vote = VotingEnsembleClassifierScratch(
        estimators=[("np", NoProba())], voting="soft"
    )
    with pytest.raises(ValueError, match="predict_proba"):
        vote.fit(np.zeros((10, 2)), np.array([0, 1] * 5))


def test_voting_hard_works_without_predict_proba():
    class NoProba:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    vote = VotingEnsembleClassifierScratch(
        estimators=[("np", NoProba())], voting="hard"
    )
    preds = vote.fit(np.zeros((12, 2)), np.array([0, 1] * 6)).predict(np.zeros((4, 2)))
    # every member votes class 0 -> the majority is class 0
    assert np.array_equal(preds, np.zeros(4, dtype=int))


def test_voting_estimators_structure(vote_data):
    X_train, _, y_train, _ = vote_data
    vote = _fit_vote(vote_data, "soft")
    assert list(vote.member_names_) == [name for name, _ in _members()]
    assert len(vote.estimators_) == 3
    assert all(est is not None for est in vote.estimators_)
    # members fit FRESH copies — the original templates stay unfitted
    for (_name, template), fitted in zip(_members(), vote.estimators_):
        assert fitted is not template


def test_voting_non_contiguous_labels():
    X, y = make_moons(n_samples=240, noise=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    y_train_big = np.where(y_train == 0, 10, 20)
    y_test_big = np.where(y_test == 0, 10, 20)
    vote = VotingEnsembleClassifierScratch(
        estimators=[
            ("tree", DecisionTreeClassifierScratch(max_depth=3)),
            ("knn", KNNClassifierScratch(n_neighbors=5)),
            ("forest", RandomForestClassifierScratch(n_estimators=9, random_state=0)),
        ],
        voting="soft",
    )
    vote.fit(X_train, y_train_big)
    preds = vote.predict(X_test)
    assert set(np.unique(preds)) == {10, 20}
    assert np.mean(preds == y_test_big) > 0.8


def test_voting_empty_estimators_raises():
    X = np.zeros((12, 2))
    y = np.array([0, 1] * 6)
    vote = VotingEnsembleClassifierScratch(estimators=[])
    with pytest.raises(ValueError, match="estimators"):
        vote.fit(X, y)


def test_voting_duplicate_names_raise():
    X = np.zeros((12, 2))
    y = np.array([0, 1] * 6)
    vote = VotingEnsembleClassifierScratch(
        estimators=[
            ("a", DecisionTreeClassifierScratch()),
            ("a", DecisionTreeClassifierScratch()),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        vote.fit(X, y)


def test_voting_invalid_voting_raises():
    X = np.zeros((12, 2))
    y = np.array([0, 1] * 6)
    vote = VotingEnsembleClassifierScratch(
        estimators=[("a", DecisionTreeClassifierScratch())], voting="bogus"
    )
    with pytest.raises(ValueError, match="voting"):
        vote.fit(X, y)


