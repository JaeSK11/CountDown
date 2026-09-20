"""CatBoost comparison model: same contract as ``flow_gbdt``."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.models import ModelRegistry, load_model
from countdown.schema import LabelSpace

pytest.importorskip("catboost")
SPACE = LabelSpace.from_names("toy", ["a", "b", "c", "never_seen"])


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    X[np.arange(n), y] += 3.0
    return X, y


def test_catboost_contract_unseen_class_column_and_round_trip(tmp_path):
    X, y = _data()
    m = ModelRegistry.create("flow_catboost", label_space=SPACE, iterations=60)
    assert m.params["class_weight"] is None                      # decision D5
    m.fit(X[:200], y[:200], val=(X[200:250], y[200:250]))
    proba = m.predict_proba(X[250:])
    assert proba.shape == (50, 4)
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-6)
    assert (proba[:, 3] == 0).all()                              # never-seen class keeps a zero column
    assert (proba.argmax(1) == y[250:]).mean() > 0.9
    restored = load_model(m.save(tmp_path / "cb.joblib"))
    np.testing.assert_allclose(restored.predict_proba(X[250:]), proba, atol=1e-9)
