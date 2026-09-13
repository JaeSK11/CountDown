"""The BaseModel contract and the registry."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.models.base import BaseModel, ModelRegistry, load_model, register
from countdown.schema import LabelSpace


@register("_stub")
class _Stub(BaseModel):
    """Minimal member: predicts the training prior, in canonical column order."""

    input_type = "flow_stats"
    supports = "*"

    def _fit(self, X, y, sample_weight=None, val=None):
        counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
        self._prior = counts / counts.sum()

    def _predict_proba(self, X):
        return np.tile(self._prior, (len(X), 1))

    def _state(self):
        return {"prior": self._prior.tolist()}

    def _load_state(self, state):
        self._prior = np.asarray(state["prior"], dtype=np.float64)


@register("_narrow")
class _Narrow(_Stub):
    supports = {"traffic_type"}


SPACE = LabelSpace.taxonomy("traffic_type")


def _xy(n=40, classes=(2, 6)):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 5))
    y = np.array([classes[i % len(classes)] for i in range(n)], dtype=np.int64)
    return X, y


# --- registry ----------------------------------------------------------------------------

def test_registry_creates_by_name_and_lists_members():
    m = ModelRegistry.create("_stub", label_space=SPACE)
    assert isinstance(m, _Stub) and m.name == "_stub"
    assert "_stub" in ModelRegistry.list() and "flow_gbdt" in ModelRegistry.list()


def test_unknown_model_names_the_alternatives():
    with pytest.raises(KeyError, match="available"):
        ModelRegistry.create("nope")


# --- the label-space contract ------------------------------------------------------------

def test_fit_without_a_label_space_is_refused():
    m = ModelRegistry.create("_stub")
    with pytest.raises(ValueError, match="no label_space"):
        m.fit(*_xy())


def test_proba_is_label_space_wide_regardless_of_classes_seen():
    """Trained on 2 of 8 classes; still emits 8 columns, zeros for the rest."""
    X, y = _xy()
    m = ModelRegistry.create("_stub", label_space=SPACE).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 8)
    assert np.allclose(proba.sum(axis=1), 1.0)
    unseen = [i for i in range(8) if i not in (2, 6)]
    assert np.allclose(proba[:, unseen], 0.0)
    assert m.classes_ == list(SPACE.names) and m.n_classes_ == 8


def test_dataset_local_ids_are_caught_rather_than_silently_miscolumned():
    """Passing ds.labels() instead of ds.labels(space=...) must fail loudly."""
    X, _ = _xy()
    y_local = np.zeros(len(X), dtype=np.int64)
    y_local[-1] = 99
    m = ModelRegistry.create("_stub", label_space=SPACE)
    with pytest.raises(ValueError, match="outside the 'traffic_type' LabelSpace"):
        m.fit(X, y_local)


def test_predict_returns_canonical_ids_and_names():
    X, y = _xy(classes=(6, 6))
    m = ModelRegistry.create("_stub", label_space=SPACE).fit(X, y)
    assert set(m.predict(X).tolist()) == {6}
    assert set(m.predict_names(X)) == {SPACE.name(6)}


def test_a_member_returning_the_wrong_width_is_rejected():
    class _Bad(_Stub):
        def _predict_proba(self, X):
            return np.zeros((len(X), 3))

    m = _Bad(label_space=SPACE).fit(*_xy())
    with pytest.raises(ValueError, match="expected \\(N, 8\\)"):
        m.predict_proba(np.zeros((4, 5)))


def test_unfitted_predict_is_refused():
    with pytest.raises(ValueError, match="not fitted"):
        ModelRegistry.create("_stub", label_space=SPACE).predict_proba(np.zeros((2, 5)))


def test_target_support_is_enforced():
    m = ModelRegistry.create("_narrow", label_space=SPACE)
    assert m.supports_target("traffic_type")
    with pytest.raises(ValueError, match="does not support target"):
        m._check_target("app")


def test_mismatched_x_and_y_lengths_are_caught():
    m = ModelRegistry.create("_stub", label_space=SPACE)
    with pytest.raises(ValueError, match="rows but y has"):
        m.fit(np.zeros((5, 3)), np.zeros(4, dtype=np.int64))


# --- persistence -------------------------------------------------------------------------

def test_save_load_round_trip_carries_the_label_space(tmp_path):
    X, y = _xy()
    m = ModelRegistry.create("_stub", label_space=SPACE).fit(X, y, feature_names=list("abcde"))
    path = m.save(tmp_path / "m.joblib")

    back = load_model(path)
    assert isinstance(back, _Stub)
    assert back.label_space == SPACE           # column meaning travels with the artifact
    assert back.feature_names == list("abcde")
    assert back.trained_classes_.tolist() == [2, 6]
    assert np.allclose(back.predict_proba(X), m.predict_proba(X))


def test_saving_an_unfitted_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not fitted"):
        ModelRegistry.create("_stub", label_space=SPACE).save(tmp_path / "m.joblib")


def test_get_set_params():
    m = ModelRegistry.create("_stub", label_space=SPACE, alpha=1)
    assert m.get_params()["alpha"] == 1
    assert m.set_params(alpha=2).get_params()["alpha"] == 2
