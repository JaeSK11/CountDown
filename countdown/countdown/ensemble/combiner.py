"""Combiners over members' (calibrated) ``predict_proba``: soft vote and stacking.

Inputs are a list of ``(N, C)`` blocks, one per member, every block in the same
``LabelSpace`` column order -- which the ``BaseModel`` contract guarantees, so building the
meta-feature matrix is a concatenation.  A member with no prediction for a row (not routed,
or no input for that flow) is passed as a row of NaNs: it becomes a zero block plus a
``missing`` indicator column for the stacker and is skipped by the vote.

``SoftVote``  weighted mean; weights = each member's out-of-fold macro-F1, normalised.
``Stacker``   a meta-learner on the concatenated blocks -- ``lgbm`` (the plan's LightGBM)
              or ``logreg`` (multinomial logistic regression on log-probabilities, the
              textbook stacker and the cheaper one at 120 classes).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from countdown.eval.metrics import compute_metrics

_EPS = 1e-6


def _present(block: np.ndarray) -> np.ndarray:
    return ~np.isnan(block).any(axis=1)


class SoftVote:
    def __init__(self, weights: str | Sequence[float] = "oof_f1") -> None:
        self.weights = weights
        self.weights_: np.ndarray | None = None

    def fit(self, blocks: list[np.ndarray], y: np.ndarray) -> "SoftVote":
        if isinstance(self.weights, str):
            if self.weights == "equal":
                w = np.ones(len(blocks))
            else:
                n = blocks[0].shape[1]
                w = np.array([compute_metrics(y[_present(b)], np.nan_to_num(b[_present(b)]).argmax(1),
                                              n)["macro_f1"] for b in blocks])
        else:
            w = np.asarray(self.weights, dtype=np.float64)
        w = np.clip(w, 1e-6, None)
        self.weights_ = w / w.sum()
        return self

    def predict_proba(self, blocks: list[np.ndarray]) -> np.ndarray:
        num = np.zeros_like(blocks[0], dtype=np.float64)
        den = np.zeros((blocks[0].shape[0], 1))
        for w, b in zip(self.weights_, blocks):
            ok = _present(b)
            num[ok] += w * b[ok]
            den[ok] += w
        den[den == 0] = 1.0
        out = num / den
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


class Stacker:
    def __init__(self, meta: str = "logreg", seed: int = 42, **params: Any) -> None:
        if meta not in ("lgbm", "logreg"):
            raise ValueError(f"unknown meta-learner {meta!r}")
        self.meta = meta
        self.seed = seed
        self.params = params
        self.model_ = None
        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self._has_missing = False

    def _features(self, blocks: list[np.ndarray]) -> np.ndarray:
        cols = []
        for b in blocks:
            ok = _present(b)
            v = np.where(ok[:, None], np.nan_to_num(b), 0.0)
            if self.meta == "logreg":
                v = np.where(ok[:, None], np.log(np.clip(v, _EPS, 1.0)), 0.0)
            cols.append(v)
            if self._has_missing:
                cols.append((~ok).astype(np.float64)[:, None])
        return np.hstack(cols)

    def fit(self, blocks: list[np.ndarray], y: np.ndarray) -> "Stacker":
        self.n_classes_ = blocks[0].shape[1]
        self._has_missing = any(not _present(b).all() for b in blocks)
        X = self._features(blocks)
        self.classes_ = np.unique(y)
        if self.meta == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            p = {"C": 1.0, "max_iter": 2000, **self.params}
            self.model_ = make_pipeline(StandardScaler(), LogisticRegression(**p))
        else:
            import lightgbm as lgb

            p = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15,
                 "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
                 "colsample_bytree": 0.8, "verbose": -1, "random_state": self.seed,
                 **self.params}
            self.model_ = lgb.LGBMClassifier(**p)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, blocks: list[np.ndarray]) -> np.ndarray:
        dense = self.model_.predict_proba(self._features(blocks))
        out = np.zeros((dense.shape[0], self.n_classes_))
        out[:, self.classes_] = dense
        return out


def make_combiner(kind: str, seed: int = 42) -> SoftVote | Stacker:
    """``vote`` | ``stack_logreg`` | ``stack_lgbm`` (``stacking`` = ``stack_logreg``)."""
    if kind in ("vote", "soft_vote"):
        return SoftVote()
    if kind in ("stacking", "stack_logreg"):
        return Stacker("logreg", seed=seed)
    if kind == "stack_lgbm":
        return Stacker("lgbm", seed=seed)
    raise ValueError(f"unknown combiner {kind!r}")
