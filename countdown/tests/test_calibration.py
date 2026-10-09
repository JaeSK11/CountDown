"""Phase 5.1: calibration lowers ECE, keeps the arg-max and zero columns, and holds D5."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.ensemble.calibration import (
    CalibratedModel, ProbaCalibrator, check_convention, ece, nll,
)
from countdown.models.base import BaseModel, ModelRegistry
from countdown.schema import LabelSpace


def _overconfident(n=4000, c=4, seed=0):
    """Labels drawn from p; reported probabilities are p sharpened -- a classic over-confident net."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, c)) * 1.5
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(c, p=row) for row in p])
    sharp = p ** 4
    sharp /= sharp.sum(1, keepdims=True)
    return sharp, y


def test_temperature_lowers_ece_and_nll_without_moving_the_argmax():
    p, y = _overconfident()
    cal = ProbaCalibrator("temperature").fit(p[:2000], y[:2000])
    q = cal.transform(p[2000:])
    assert cal.temperature_ > 1.5                       # it learnt to soften
    assert ece(q, y[2000:]) < 0.5 * ece(p[2000:], y[2000:])
    assert nll(q, y[2000:]) < nll(p[2000:], y[2000:])
    assert (q.argmax(1) == p[2000:].argmax(1)).all()
    np.testing.assert_allclose(q.sum(1), 1.0)


def test_untrained_class_columns_stay_exactly_zero():
    p, y = _overconfident(c=3)
    p = np.hstack([p, np.zeros((len(p), 1))])           # class 3: the member never saw it
    for method in ("temperature", "isotonic"):
        q = ProbaCalibrator(method, min_isotonic=10).fit(p, y).transform(p)
        assert (q[:, 3] == 0).all(), method


def test_isotonic_falls_back_to_temperature_on_thin_classes():
    p, y = _overconfident(n=600)
    cal = ProbaCalibrator("isotonic", min_isotonic=200).fit(p, y)
    assert cal.method_used_ == "temperature"
    assert ProbaCalibrator("isotonic", min_isotonic=5).fit(p, y).method_used_ == "isotonic"


def _gbdt(space, **kw):
    return ModelRegistry.create("flow_gbdt", label_space=space, n_estimators=30, **kw)


def test_calibrated_model_is_a_base_model_and_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)
    space = LabelSpace.from_names("toy", ["a", "b", "c"])
    base = _gbdt(space).fit(X[:400], y[:400])
    cal = CalibratedModel(base=base).fit(X[400:500], y[400:500])
    assert isinstance(cal, BaseModel) and cal.input_type == "flow_stats"
    out = cal.predict_proba(X[500:])
    assert out.shape == (100, 3)
    back = BaseModel.load(cal.save(tmp_path / "cal.joblib"))
    assert type(back) is CalibratedModel
    np.testing.assert_allclose(back.predict_proba(X[500:]), out)


def test_class_weight_convention_is_enforced():
    space = LabelSpace.from_names("toy", ["a", "b"])
    check_convention({"a": _gbdt(space)})
    with pytest.raises(ValueError, match="D5"):
        check_convention({"a": _gbdt(space), "b": _gbdt(space, class_weight="balanced")})
