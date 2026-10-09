"""Per-member probability calibration, and the ECE metric that says whether it worked.

Members emit ``predict_proba`` on different scales -- a softmax trained to the last epoch is
over-confident, a GBDT is often under-confident -- and averaging or stacking raw scores lets
the worst-calibrated member dominate (``PHASE-5.md`` §1).  So every member is calibrated on
held-out predictions before anything combines them.

Two methods:

``temperature``  one scalar ``T``: ``softmax(log p / T)``.  It cannot change the arg-max, so
                 it cannot cost accuracy, and one parameter cannot overfit a thin class.
``isotonic``     one-vs-rest isotonic per class, then renormalised.  More flexible, and
                 overfits a class with few calibration samples -- so it is used only when
                 every trained class has at least ``min_isotonic`` of them, and falls back
                 to temperature otherwise (``p2p`` is 10 captures, ``chia.net`` 16 flows).

Classes a member never trained on keep an exact zero column, before and after: the
``BaseModel`` contract is that a zero column means "this member has no opinion", and a
calibrator that smeared mass into it would break the combiner's reading of that.

The calibration split must carry the **true** class distribution -- never rebalanced -- and
every member must share one class-weight convention (D5: none).  ``check_convention`` is
what enforces the second; the first is the caller's (Phase 5 calibrates on out-of-fold
predictions, which are the training distribution by construction).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.models.base import BaseModel, register

log = get_logger(__name__)

_EPS = 1e-12


def ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error: ``sum_b |acc_b - conf_b| * n_b / n``."""
    proba = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y)
    if len(y) == 0:
        return float("nan")
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            total += abs(correct[sel].mean() - conf[sel].mean()) * sel.sum()
    return float(total / len(y))


def reliability(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    """Per-bin ``{lo, hi, n, confidence, accuracy}`` -- the data behind a reliability plot."""
    proba = np.asarray(proba, dtype=np.float64)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        sel = idx == b
        out.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(sel.sum()),
            "confidence": float(conf[sel].mean()) if sel.any() else float("nan"),
            "accuracy": float(correct[sel].mean()) if sel.any() else float("nan"),
        })
    return out


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    proba = np.asarray(proba, dtype=np.float64)
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], _EPS, None))))


def _tempered(proba: np.ndarray, T: float) -> np.ndarray:
    """``softmax(log p / T)`` over the non-zero entries; exact zeros stay zero."""
    proba = np.asarray(proba, dtype=np.float64)
    live = proba > 0
    z = np.where(live, np.log(np.where(live, proba, 1.0)) / T, -np.inf)
    z -= z.max(axis=1, keepdims=True)
    e = np.where(live, np.exp(z), 0.0)
    return e / e.sum(axis=1, keepdims=True)


class ProbaCalibrator:
    """Maps a member's ``predict_proba`` to calibrated probabilities.  Fit on held-out rows."""

    def __init__(self, method: str = "temperature", min_isotonic: int = 200) -> None:
        if method not in ("temperature", "isotonic", "none"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.min_isotonic = int(min_isotonic)
        self.method_used_: str | None = None
        self.temperature_: float = 1.0
        self._iso: dict[int, Any] = {}

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "ProbaCalibrator":
        from scipy.optimize import minimize_scalar

        proba = np.asarray(proba, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        method = self.method
        if method == "isotonic":
            classes, counts = np.unique(y, return_counts=True)
            if counts.min() < self.min_isotonic:
                log.info("[calibration] isotonic -> temperature: class %d has %d calibration "
                         "samples (< %d)", int(classes[counts.argmin()]), int(counts.min()),
                         self.min_isotonic)
                method = "temperature"
        if method == "temperature":
            res = minimize_scalar(lambda t: nll(_tempered(proba, t), y),
                                  bounds=(0.05, 20.0), method="bounded")
            self.temperature_ = float(res.x)
        elif method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            for c in range(proba.shape[1]):
                if proba[:, c].max() <= 0:
                    continue
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(proba[:, c], (y == c).astype(np.float64))
                self._iso[c] = iso
        self.method_used_ = method
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.method_used_ is None:
            raise ValueError("ProbaCalibrator is not fitted")
        proba = np.asarray(proba, dtype=np.float64)
        if self.method_used_ == "none":
            return proba
        if self.method_used_ == "temperature":
            return _tempered(proba, self.temperature_)
        out = np.zeros_like(proba)
        for c, iso in self._iso.items():
            out[:, c] = iso.predict(proba[:, c])
        out = np.where(proba > 0, np.maximum(out, _EPS), 0.0)
        return out / out.sum(axis=1, keepdims=True)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method, "method_used": self.method_used_}
        if self.method_used_ == "temperature":
            d["temperature"] = round(self.temperature_, 4)
        return d


def check_convention(models: dict[str, BaseModel] | list[BaseModel]) -> None:
    """Every member on one class-weight convention (D5: ``None``), or calibration is fed
    members sitting on different implicit priors and the combiner cannot tell."""
    items = models.items() if isinstance(models, dict) else ((m.name, m) for m in models)
    bad = {n: m.params.get("class_weight") for n, m in items
           if m.params.get("class_weight") is not None}
    if bad:
        raise ValueError(
            f"members off the shared class-weight convention (D5 = None): {bad}. "
            f"Calibrate members that share one prior, or the combiner mixes priors.")


@register("calibrated")
class CalibratedModel(BaseModel):
    """A fitted member plus a :class:`ProbaCalibrator` -- still a ``BaseModel``, so it drops
    in wherever a member does.  ``fit`` here fits **only the calibrator**, on a calibration
    split, and requires the base to be fitted already."""

    def __init__(self, label_space=None, base: BaseModel | None = None,
                 method: str = "temperature", **params: Any) -> None:
        if base is not None and label_space is None:
            label_space = base.label_space
        super().__init__(label_space=label_space, method=method, **params)
        self.base = base
        self.calibrator = ProbaCalibrator(method, **{k: v for k, v in params.items()
                                                     if k == "min_isotonic"})
        if base is not None:
            self.input_type = base.input_type
            self.expects_ndim = base.expects_ndim

    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        if self.base is None or not self.base.is_fitted:
            raise ValueError("CalibratedModel needs a fitted base model before calibration")
        check_convention([self.base])
        self.calibrator.fit(self.base.predict_proba(X), y)
        self.trained_classes_ = self.base.trained_classes_

    def fit_from_proba(self, proba: np.ndarray, y: np.ndarray) -> "CalibratedModel":
        """Calibrate on predictions already in hand (out-of-fold, in Phase 5)."""
        check_convention([self.base])
        self.calibrator.fit(proba, y)
        self.trained_classes_ = self.base.trained_classes_
        self.is_fitted = True
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.base.predict_proba(X))

    def _state(self) -> dict[str, Any]:
        return {"base": self.base.to_blob(), "calibrator": self.calibrator}

    def _load_state(self, state: dict[str, Any]) -> None:
        self.base = BaseModel.from_blob(state["base"])
        self.calibrator = state["calibrator"]
        self.input_type = self.base.input_type
        self.expects_ndim = self.base.expects_ndim
