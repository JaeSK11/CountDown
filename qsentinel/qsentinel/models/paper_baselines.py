"""The Draper-Gil et al. (ICISSP 2016) classifiers, as a replication baseline.

The paper used Weka's C4.5 (J48) and IBk with default settings and 10-fold CV.  Neither
has an exact scikit-learn equivalent, so the deviations are named here rather than buried:

**C4.5 -> ``DecisionTreeClassifier(criterion="entropy")``.**  Both split on information
gain, but J48 post-prunes with a pessimistic-error estimate (``confidenceFactor=0.25``)
and scikit-learn has no equivalent -- its ``ccp_alpha`` is cost-complexity pruning, a
different rule.  We therefore grow with J48's ``minNumObj=2`` leaf floor and leave the
tree unpruned, which if anything *favours* the baseline on training fit and costs it a
little generalisation.  This is the single largest fidelity gap in the replication.

**IBk -> ``KNeighborsClassifier``.**  Weka's IBk defaults to ``k=1`` and normalises every
attribute to [0, 1] inside its distance function, so the pipeline here is a
``MinMaxScaler`` followed by ``k=1``.  Getting this wrong is not a detail: the paper's
features mix seconds (``flowiat_min`` ~ 1e-5) with bytes per second (``fb_psec`` ~ 1e6),
and unscaled Euclidean distance would be decided entirely by ``fb_psec``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from qsentinel.config import get_logger
from qsentinel.models.base import BaseModel, register

log = get_logger(__name__)


class _SklearnBaseline(BaseModel):
    """Shared label-space bookkeeping for the two paper classifiers.

    scikit-learn estimators emit columns for the classes *they* saw, in their own order.
    The ensemble contract requires columns in ``LabelSpace`` order with a zero column for
    every class the member never trained on, so the scatter-back happens here once.
    """

    input_type = "timeonly"
    supports = "*"

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        self._est = None
        self._dense_to_canonical: np.ndarray | None = None

    @abstractmethod
    def _make_estimator(self):
        """A fresh, unfitted scikit-learn estimator."""

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        val: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        # `val` is accepted and ignored: neither C4.5 nor k-NN early-stops, and the paper
        # used plain 10-fold CV with no validation fold at all.
        canonical = np.unique(y)
        self._dense_to_canonical = canonical

        if canonical.size < 2:
            log.warning(
                "[%s] only one class in training data; fitting a constant model", self.name
            )
            self._est = None
            return

        est = self._make_estimator()
        # sample_weight is deliberately not forwarded: the paper trained unweighted, and a
        # reweighted baseline would not be the number we are trying to reproduce.
        if sample_weight is not None:
            log.warning("[%s] ignoring sample_weight -- the paper baseline is unweighted", self.name)
        est.fit(X, y)
        self._est = est

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = np.asarray(X).shape[0]
        out = np.zeros((n, self.n_classes_), dtype=np.float64)
        assert self._dense_to_canonical is not None

        if self._est is None:  # degenerate single-class fit
            out[:, int(self._dense_to_canonical[0])] = 1.0
            return out

        proba = self._est.predict_proba(X)
        for j, canonical_id in enumerate(self._est.classes_):
            out[:, int(canonical_id)] = proba[:, j]
        return out

    def _state(self) -> dict[str, Any]:
        return {
            "est": self._est,
            "dense_to_canonical": (
                None if self._dense_to_canonical is None else self._dense_to_canonical.tolist()
            ),
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        self._est = state.get("est")
        d2c = state.get("dense_to_canonical")
        self._dense_to_canonical = None if d2c is None else np.asarray(d2c, dtype=np.int64)


@register("flow_c45_paper")
class PaperC45(_SklearnBaseline):
    """C4.5 decision tree on the paper's 23 time-related features."""

    def _make_estimator(self):
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(
            criterion="entropy",                                   # C4.5 = information gain
            min_samples_leaf=int(self.params.get("min_samples_leaf", 2)),  # J48 minNumObj
            random_state=int(self.params.get("seed", 42)),
        )


@register("flow_knn_paper")
class PaperKNN(_SklearnBaseline):
    """k-NN on the paper's 23 time-related features, Weka-IBk-style."""

    def _make_estimator(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler

        return Pipeline(
            [
                ("scale", MinMaxScaler()),                          # IBk normalises to [0,1]
                ("knn", KNeighborsClassifier(
                    n_neighbors=int(self.params.get("k", 1)),        # IBk default k=1
                    n_jobs=-1,
                )),
            ]
        )
