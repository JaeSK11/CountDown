"""Phase 7, task 7.3 -- per-class decision bias (no retraining)."""

from __future__ import annotations

import numpy as np

from countdown.ensemble.decision import (
    apply_bias,
    bias_for,
    fit_class_bias,
    load_biases,
    macro_f1,
)
from countdown.eval.triage import breakdown_error


def test_apply_bias_renormalises_and_moves_the_argmax():
    p = np.array([[0.6, 0.4], [0.9, 0.1]])
    q = apply_bias(p, np.array([0.0, np.log(2.0)]))
    assert np.allclose(q.sum(1), 1.0)
    assert q[0].argmax() == 1 and q[1].argmax() == 0
    assert np.allclose(apply_bias(p, np.zeros(2)), p)


def test_fit_recovers_an_under_predicted_class():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    logits = rng.normal(size=(3000, 3))
    logits[np.arange(3000), y] += 1.5
    logits[:, 2] -= 1.2                     # class 2 is systematically under-scored
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    b = fit_class_bias(p, y)
    assert abs(b.mean()) < 1e-9
    assert b[2] > b[0] and b[2] > b[1]
    assert macro_f1(y, apply_bias(p, b).argmax(1), 3) > macro_f1(y, p.argmax(1), 3)


def test_bias_for_refuses_a_mismatched_label_space():
    cfg = {"E9": {"bias": {"a": 0.5, "b": -0.5}}}
    assert np.allclose(bias_for("E9", ["b", "a"], cfg), [-0.5, 0.5])   # by name, not position
    assert bias_for("E9", ["a", "c"], cfg) is None
    assert bias_for("E1", ["a"], {}) is None


def test_shipped_e1_bias_covers_the_traffic_types():
    e1 = load_biases()["E1"]["bias"]
    assert set(e1) == {"browsing", "email", "chat", "audio_streaming", "video_streaming",
                       "file_transfer", "voip", "p2p"}
    assert e1["file_transfer"] > 0


def test_breakdown_error_by_bytes():
    true, pred = ["a", "b"], ["b", "b"]
    assert breakdown_error(true, pred)["total_variation"] == 0.5
    # the misclassified flow is 1 % of the bytes
    assert breakdown_error(true, pred, [1, 99])["total_variation"] == 0.01
