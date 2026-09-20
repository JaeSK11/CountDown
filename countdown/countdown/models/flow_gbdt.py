"""LightGBM on flow statistics -- the universal baseline member.

The papers this project builds on used Random Forest for tabular flow features; gradient
boosting is a strict upgrade there (accuracy, inference cost, native class weighting, early
stopping), so this is the member that runs on *every* target and the floor that the Phase-4
deep models have to beat.  See ``phases/MODEL-DECISIONS.md``.

Two things it has to get right beyond calling LightGBM:

**Class imbalance.**  ISCX ``traffic_type`` is skewed roughly 30:1 (``browsing`` 25,997 vs
``audio_streaming`` 840), so plain accuracy is nearly uninformative -- predicting the two
biggest classes scores well.  Training uses balanced sample weights and the headline metric
is macro-F1.

**The label-space widening.**  LightGBM trains on the classes it actually sees, densely
indexed from zero.  The ensemble needs columns in :class:`~countdown.schema.LabelSpace`
order, whatever the training data happened to contain.  ``_predict_proba`` scatters
LightGBM's dense columns back into canonical positions, leaving zeros for unseen classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.models.base import BaseModel, register

log = get_logger(__name__)


def _accepts_eval_xy(fn: Any) -> bool:
    """True when this LightGBM exposes the post-4.6 ``eval_X`` / ``eval_y`` arguments."""
    import inspect

    try:
        return "eval_X" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level signature
        return False

#: Defaults tuned for tens of thousands of rows and ~56 features -- small enough not to
#: memorise 140 captures, with early stopping doing the real work when a val fold exists.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


@register("flow_gbdt")
class FlowStatsGBDT(BaseModel):
    """Multiclass LightGBM over the ``flow_stats`` feature table."""

    input_type = "flow_stats"
    supports = "*"

    def __init__(
        self,
        label_space=None,
        class_weight: str | dict | None = None,   # D5 (2026-09-19): no class weights, any member
        early_stopping_rounds: int = 50,
        seed: int = 42,
        **params: Any,
    ) -> None:
        super().__init__(
            label_space=label_space,
            class_weight=class_weight,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            **params,
        )
        self._booster = None
        #: Canonical ids of the columns LightGBM emits, in its own dense order.
        self._dense_to_canonical: np.ndarray | None = None
        self.best_iteration_: int | None = None

    # -- helpers -----------------------------------------------------------------------
    def _lgb_params(self) -> dict[str, Any]:
        p = dict(DEFAULT_PARAMS)
        for k, v in self.params.items():
            if k not in ("class_weight", "early_stopping_rounds", "seed"):
                p[k] = v
        p["random_state"] = int(self.params.get("seed", 42))
        return p

    @staticmethod
    def _balanced_weights(y: np.ndarray) -> np.ndarray:
        """``n / (k * count[class])`` -- sklearn's 'balanced', computed per sample."""
        classes, counts = np.unique(y, return_counts=True)
        w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
        return np.array([w[v] for v in y], dtype=np.float64)

    def _weights(self, y: np.ndarray, sample_weight: np.ndarray | None) -> np.ndarray | None:
        if sample_weight is not None:
            return np.asarray(sample_weight, dtype=np.float64)
        cw = self.params.get("class_weight")
        if cw == "balanced":
            return self._balanced_weights(y)
        if isinstance(cw, dict):
            return np.array([cw.get(int(v), 1.0) for v in y], dtype=np.float64)
        return None

    # -- BaseModel -------------------------------------------------------------------
    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        val: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        import lightgbm as lgb

        # LightGBM wants dense 0..k-1 ids; remember how to scatter them back.
        canonical = np.unique(y)
        self._dense_to_canonical = canonical
        remap = {int(c): i for i, c in enumerate(canonical)}
        y_dense = np.array([remap[int(v)] for v in y], dtype=np.int64)

        params = self._lgb_params()
        if len(canonical) < 2:
            # One class in train: LightGBM refuses, and there is nothing to learn anyway.
            # Keep it a working (degenerate) model so a thin fold does not abort a sweep.
            log.warning(
                "[flow_gbdt] only one class in training data (%s); fitting a constant model",
                self.label_space.name(int(canonical[0])) if self.label_space else canonical[0],
            )
            self._booster = None
            return

        model = lgb.LGBMClassifier(objective="multiclass", num_class=len(canonical), **params)
        # Feature names are deliberately NOT handed to LightGBM: X arrives as a bare
        # ndarray at predict time, and a booster fitted with names warns on every call.
        # Importances are positional, so self.feature_names maps them just as well.
        fit_kw: dict[str, Any] = {"sample_weight": self._weights(y, sample_weight)}

        if val is not None:
            Xv, yv = np.asarray(val[0]), np.asarray(val[1])
            unseen = set(np.unique(yv).tolist()) - set(canonical.tolist())
            if unseen:
                # Early stopping needs the val labels inside the trained class set.
                log.warning(
                    "[flow_gbdt] validation fold holds %d class(es) absent from train; "
                    "dropping those rows for early stopping only", len(unseen)
                )
                keep = ~np.isin(yv, list(unseen))
                Xv, yv = Xv[keep], yv[keep]
            if len(yv):
                yv_dense = np.array([remap[int(v)] for v in yv], dtype=np.int64)
                rounds = int(self.params.get("early_stopping_rounds", 50))
                fit_kw["eval_metric"] = "multi_logloss"
                fit_kw["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]
                # eval_set was deprecated in lightgbm 4.6; fall back for older versions.
                if _accepts_eval_xy(model.fit):
                    fit_kw["eval_X"], fit_kw["eval_y"] = Xv, yv_dense
                else:  # pragma: no cover - older lightgbm
                    fit_kw["eval_set"] = [(Xv, yv_dense)]

        model.fit(X, y_dense, **fit_kw)
        self._booster = model
        self.best_iteration_ = getattr(model, "best_iteration_", None)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = np.asarray(X).shape[0]
        out = np.zeros((n, self.n_classes_), dtype=np.float64)
        assert self._dense_to_canonical is not None

        if self._booster is None:  # degenerate single-class fit
            out[:, int(self._dense_to_canonical[0])] = 1.0
            return out

        dense = self._booster.predict_proba(X)
        for j, canonical_id in enumerate(self._dense_to_canonical):
            out[:, int(canonical_id)] = dense[:, j]
        return out

    # -- extras ------------------------------------------------------------------------
    def feature_importance(self, top_k: int | None = None) -> list[tuple[str, float]]:
        """``(feature, gain)`` pairs, most important first."""
        self._require_fitted()
        if self._booster is None:
            return []
        gains = np.asarray(self._booster.feature_importances_, dtype=np.float64)
        names = self.feature_names or [f"f{i}" for i in range(len(gains))]
        ranked = sorted(zip(names, gains), key=lambda kv: -kv[1])
        return ranked[:top_k] if top_k else ranked

    # -- persistence -------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        return {
            "booster": self._booster,
            "dense_to_canonical": (
                None if self._dense_to_canonical is None
                else self._dense_to_canonical.tolist()
            ),
            "best_iteration": self.best_iteration_,
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        self._booster = state.get("booster")
        d2c = state.get("dense_to_canonical")
        self._dense_to_canonical = None if d2c is None else np.asarray(d2c, dtype=np.int64)
        self.best_iteration_ = state.get("best_iteration")
