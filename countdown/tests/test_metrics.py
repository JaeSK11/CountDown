"""Metrics, checked against sklearn references, plus the baseline that gives them meaning."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from countdown.eval.metrics import (
    baseline_metrics,
    compute_metrics,
    evaluate,
    metrics_excluding_tail,
)
from countdown.models.base import BaseModel, register
from countdown.schema import LabelSpace


@register("_fixed")
class _Fixed(BaseModel):
    """Returns a canned probability matrix, so metrics are tested, not a model."""

    def __init__(self, label_space=None, proba=None, **kw):
        super().__init__(label_space=label_space, **kw)
        self._proba = proba

    def _fit(self, X, y, sample_weight=None, val=None):
        pass

    def _predict_proba(self, X):
        return self._proba


SPACE = LabelSpace.taxonomy("tunnel_type")   # none / vpn / tor


def test_metrics_match_sklearn_references():
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 2])
    m = compute_metrics(y_true, y_pred, n_classes=3, class_names=["a", "b", "c"])

    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y_true, y_pred, average="weighted"))
    assert m["n"] == 8
    assert np.asarray(m["confusion"]).shape == (3, 3)
    assert np.asarray(m["confusion"]).sum() == 8
    assert m["per_class"]["a"]["support"] == 3


def test_per_class_entries_exist_for_classes_with_no_support():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y, n_classes=3, class_names=["a", "b", "c"])
    assert m["per_class"]["c"]["support"] == 0
    assert m["per_class"]["c"]["f1"] == 0.0
    # macro over PRESENT classes is perfect; over all declared classes it is not
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["macro_f1_all_classes"] == pytest.approx(2 / 3)


def test_majority_baseline_is_computed_from_train_not_test():
    y_train = np.array([0] * 90 + [1] * 10)
    y_test = np.array([1] * 50 + [0] * 50)
    base = baseline_metrics(y_train, y_test, n_classes=2, class_names=["a", "b"])
    assert base["most_frequent"]["accuracy"] == pytest.approx(0.5)   # predicts 0 always
    assert "stratified" in base


def test_imbalanced_accuracy_flatters_where_macro_f1_does_not():
    """Why macro-F1 is the headline: 95% accuracy, useless model."""
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.zeros(100, dtype=int)
    m = compute_metrics(y_true, y_pred, n_classes=2, class_names=["big", "small"])
    assert m["accuracy"] == pytest.approx(0.95)
    assert m["macro_f1"] < 0.5


def test_evaluate_reports_the_baseline_alongside_and_the_verdict():
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]])
    y = np.array([0, 1, 2, 0])
    model = _Fixed(label_space=SPACE, proba=proba).fit(np.zeros((4, 2)), y)

    rep = evaluate(model, np.zeros((4, 2)), y, space=SPACE, y_train=y)
    assert rep["macro_f1"] == pytest.approx(1.0)
    assert rep["target"] == "tunnel_type" and rep["n_classes"] == 3
    assert rep["beats_baseline"] is True
    assert rep["macro_f1"] > rep["baseline_macro_f1"]
    assert rep["mean_confidence"] == pytest.approx(np.mean(proba.max(axis=1)))


def test_evaluate_needs_a_label_space():
    model = _Fixed(label_space=None, proba=np.zeros((2, 3)))
    model.is_fitted = True
    with pytest.raises(ValueError, match="needs a LabelSpace"):
        evaluate(model, np.zeros((2, 2)), np.zeros(2, dtype=int))


def test_tail_exclusion_reports_both_populations():
    rep = {"per_class": {
        "big1": {"f1": 0.9, "support": 100},
        "big2": {"f1": 0.7, "support": 80},
        "tail": {"f1": 0.0, "support": 3},
    }}
    out = metrics_excluding_tail(rep, min_support=50)
    assert out["n_classes"] == 2 and out["n_classes_dropped"] == 1
    assert out["macro_f1"] == pytest.approx(0.8)     # the tail zero is excluded
    assert metrics_excluding_tail(rep, min_support=1000)["macro_f1"] is None
