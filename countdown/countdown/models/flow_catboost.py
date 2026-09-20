"""CatBoost on ``flow_stats`` -- the second GBDT in the flow-stats bake-off, not a member.

``MODEL-flow-stats.md`` asks for CatBoost on the scorecard for two mechanisms LightGBM
lacks: *ordered boosting* (less target leakage on small, imbalanced classes) and *oblivious
trees* (a strong regulariser and very fast inference).  It is registered so that it runs
through the same harness, split and metrics as ``flow_gbdt``; it must **not** be promoted
to a second ensemble member -- two boosted-tree models on one feature vector make highly
correlated errors and add almost nothing to the combiner.  One winner is the member.

Follows the same conventions as ``flow_gbdt``: 600 rounds with early stopping on the
validation fold, no class weights (decision D5), dense <-> canonical class-id remapping so
``predict_proba`` stays in ``LabelSpace`` order with zero columns for unseen classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.models.base import BaseModel, register

log = get_logger(__name__)


@register("flow_catboost")
class FlowStatsCatBoost(BaseModel):
    input_type = "flow_stats"
    supports = "*"

    def __init__(self, label_space=None, class_weight: str | None = None,
                 early_stopping_rounds: int = 50, seed: int = 42, **params: Any) -> None:
        super().__init__(label_space=label_space, class_weight=class_weight,
                         early_stopping_rounds=early_stopping_rounds, seed=seed, **params)
        self.params.setdefault("iterations", 600)
        self.params.setdefault("learning_rate", 0.1)
        self.params.setdefault("depth", 6)
        self.params.setdefault("thread_count", -1)
        self._model = None
        self._dense_to_canonical: np.ndarray | None = None
        self.best_iteration_: int | None = None

    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError("flow_catboost needs CatBoost: pip install catboost") from exc

        canonical = np.unique(y)
        self._dense_to_canonical = canonical
        remap = {int(c): i for i, c in enumerate(canonical)}
        y_dense = np.array([remap[int(v)] for v in y], dtype=np.int64)
        if len(canonical) < 2:
            log.warning("[flow_catboost] one class in training data; fitting a constant model")
            self._model = None
            return

        p = self.params
        kw = {k: v for k, v in p.items() if k not in ("class_weight", "early_stopping_rounds", "seed")}
        if p.get("class_weight") == "balanced":
            kw["auto_class_weights"] = "Balanced"
        elif p.get("class_weight") not in (None, "none"):
            raise ValueError(f"class_weight must be None or 'balanced', got {p['class_weight']!r}")
        model = CatBoostClassifier(loss_function="MultiClass", random_seed=int(p["seed"]),
                                   verbose=False, allow_writing_files=False, **kw)
        fit_kw: dict[str, Any] = {"sample_weight": sample_weight}
        if val is not None and len(val[1]):
            Xv, yv = np.asarray(val[0]), np.asarray(val[1])
            keep = np.isin(yv, canonical)
            if keep.any():
                fit_kw["eval_set"] = (Xv[keep], np.array([remap[int(v)] for v in yv[keep]], dtype=np.int64))
                fit_kw["early_stopping_rounds"] = int(p["early_stopping_rounds"])
        model.fit(np.asarray(X), y_dense, **fit_kw)
        self._model = model
        self.best_iteration_ = model.get_best_iteration()

    def _predict_proba(self, X) -> np.ndarray:
        n = np.asarray(X).shape[0]
        out = np.zeros((n, self.n_classes_), dtype=np.float64)
        assert self._dense_to_canonical is not None
        if self._model is None:
            out[:, int(self._dense_to_canonical[0])] = 1.0
            return out
        proba = np.asarray(self._model.predict_proba(np.asarray(X)))
        # CatBoost orders its columns by ``classes_``, which are our dense ids.
        dense = np.asarray(self._model.classes_, dtype=np.int64)
        out[:, self._dense_to_canonical[dense]] = proba
        return out

    def _state(self) -> dict[str, Any]:
        blob = None
        if self._model is not None:
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
                path = f.name
            self._model.save_model(path)
            blob = open(path, "rb").read()
            os.unlink(path)
        return {"cbm": blob, "dense_to_canonical": None if self._dense_to_canonical is None
                else self._dense_to_canonical.tolist(), "best_iteration": self.best_iteration_}

    def _load_state(self, state: dict[str, Any]) -> None:
        self._dense_to_canonical = np.asarray(state["dense_to_canonical"], dtype=np.int64)
        self.best_iteration_ = state.get("best_iteration")
        self._model = None
        if state.get("cbm"):
            from catboost import CatBoostClassifier

            self._model = CatBoostClassifier()
            self._model.load_model(blob=state["cbm"])
