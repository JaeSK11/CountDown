"""The LightGBM member: the seam, the imbalance handling, and persistence."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.models import ModelRegistry, load_model
from countdown.schema import LabelSpace

SPACE = LabelSpace.taxonomy("traffic_type")   # 8 classes


def _separable(n_per_class=40, classes=(1, 4, 6), n_features=6, seed=0):
    """Well-separated Gaussian blobs, one per canonical class id."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, cls in enumerate(classes):
        X.append(rng.normal(loc=k * 6.0, scale=1.0, size=(n_per_class, n_features)))
        y.append(np.full(n_per_class, cls, dtype=np.int64))
    return np.vstack(X), np.concatenate(y)


def test_fits_and_predicts_proba_in_label_space_order():
    X, y = _separable()
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE).fit(X, y)

    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 8)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_classes_never_trained_on_get_a_zero_column():
    """A member trained on ISCXVPN has never seen `browsing`; it must say zero, not shift."""
    X, y = _separable(classes=(1, 4, 6))
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE).fit(X, y)
    proba = m.predict_proba(X)

    for unseen in (0, 2, 3, 5, 7):
        assert np.allclose(proba[:, unseen], 0.0), f"class {unseen} should be zero"
    assert np.allclose(proba[:, [1, 4, 6]].sum(axis=1), 1.0)


def test_it_actually_learns_a_separable_problem():
    X, y = _separable()
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE, n_estimators=40).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.95
    assert set(m.predict_names(X)) == {SPACE.name(c) for c in (1, 4, 6)}


def test_balanced_weights_counteract_a_skewed_training_set():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (400, 4)), rng.normal(6, 1, (12, 4))])
    y = np.array([1] * 400 + [7] * 12, dtype=np.int64)

    balanced = ModelRegistry.create(
        "flow_gbdt", label_space=SPACE, n_estimators=60, class_weight="balanced"
    ).fit(X, y)
    # the rare class must still be recoverable, not swallowed by the majority
    assert (balanced.predict(X[400:]) == 7).mean() > 0.8


def test_a_single_class_training_set_degrades_instead_of_crashing():
    X, y = _separable(classes=(3,))
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE).fit(X, y)
    proba = m.predict_proba(X)
    assert np.allclose(proba[:, 3], 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_early_stopping_runs_when_a_val_fold_is_given():
    """Unlearnable labels: validation loss stops improving, so the fit must stop early."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = rng.choice([1, 4, 6], size=400).astype(np.int64)
    Xv = rng.normal(size=(120, 6))
    yv = rng.choice([1, 4, 6], size=120).astype(np.int64)

    m = ModelRegistry.create(
        "flow_gbdt", label_space=SPACE, n_estimators=500, early_stopping_rounds=5
    ).fit(X, y, val=(Xv, yv))
    assert m.best_iteration_ is not None and m.best_iteration_ < 500


def test_a_val_class_absent_from_train_does_not_break_early_stopping():
    X, y = _separable(classes=(1, 4))
    Xv, yv = _separable(classes=(1, 4, 6), n_per_class=10, seed=3)
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE, n_estimators=40).fit(
        X, y, val=(Xv, yv)
    )
    assert m.is_fitted


def test_feature_importances_are_named():
    X, y = _separable(n_features=4)
    names = ["a", "b", "c", "d"]
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE, n_estimators=30).fit(
        X, y, feature_names=names
    )
    imp = m.feature_importance()
    assert len(imp) == 4 and {f for f, _ in imp} == set(names)
    assert [g for _, g in imp] == sorted([g for _, g in imp], reverse=True)
    assert len(m.feature_importance(top_k=2)) == 2


def test_save_load_round_trip_reproduces_probabilities(tmp_path):
    X, y = _separable()
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE, n_estimators=30).fit(
        X, y, feature_names=[f"f{i}" for i in range(X.shape[1])]
    )
    path = m.save(tmp_path / "gbdt.joblib")

    back = load_model(path)
    assert back.name == "flow_gbdt"
    assert back.label_space == SPACE
    assert back.feature_names == [f"f{i}" for i in range(X.shape[1])]
    assert np.allclose(back.predict_proba(X), m.predict_proba(X))


def test_input_type_is_declared_so_the_harness_can_check_it():
    m = ModelRegistry.create("flow_gbdt", label_space=SPACE)
    assert m.input_type == "flow_stats"
    assert m.supports_target("anything")
