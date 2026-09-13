"""The model contract every ensemble member implements, plus the registry.

Phase 3 ships one model.  This interface exists now anyway, because it is the seam that
lets Phase 4 add a CNN, an SSM, a byte transformer and a GNN, and Phase 5 calibrate and
stack any subset of them, without rewriting the members.

The load-bearing part is :meth:`BaseModel.predict_proba`, which returns ``(N, C)`` with
columns in :class:`~countdown.schema.LabelSpace` order.  Aligned columns are what make
stacking a matter of concatenation.  A model that reports its own idea of the class set
would force every downstream consumer to re-derive the mapping, and the failure mode is
silent: probabilities land in the wrong column and the combiner still runs.

Subclasses implement ``_fit`` / ``_predict_proba``, not ``fit`` / ``predict_proba`` -- the
base class owns the label-space bookkeeping so no member can get it wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from countdown.config import get_logger
from countdown.schema import LabelSpace

log = get_logger(__name__)

_REGISTRY: dict[str, type["BaseModel"]] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator: make a model constructible by name."""

    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return deco


class ModelRegistry:
    """Name -> model class.  Members self-register with :func:`register`."""

    @staticmethod
    def list() -> list[str]:
        return sorted(_REGISTRY)

    @staticmethod
    def get(name: str) -> type["BaseModel"]:
        if name not in _REGISTRY:
            raise KeyError(f"unknown model {name!r}; available: {ModelRegistry.list()}")
        return _REGISTRY[name]

    @staticmethod
    def create(name: str, **params: Any) -> "BaseModel":
        return ModelRegistry.get(name)(**params)


class BaseModel(ABC):
    """One ensemble member.

    Attributes
    ----------
    name
        Registry key, set by :func:`register`.
    input_type
        Which feature representation ``X`` must come from -- ``flow_stats`` here,
        ``packet_seq`` / ``bytes`` / ``image`` / ``graph`` for the Phase-4 members.  The
        training harness checks this rather than trusting the config.
    expects_ndim
        The rank of that representation.  Checked by the harness alongside ``input_type``.
    supports
        Targets the model can train, or ``"*"`` for any.
    label_space
        The canonical column ordering of ``predict_proba``.  Required before ``fit``.
    """

    name: str = "base"
    input_type: str = "flow_stats"
    supports: set[str] | str = "*"
    #: Rank of the ``X`` this model consumes: 2 for a ``(N, F)`` table, 3 for a
    #: ``(N, T, C)`` sequence, 4 for a ``(N, C, H, W)`` image.  The training harness
    #: checks ``X.ndim`` against this instead of assuming every member is tabular --
    #: a member that silently accepts the wrong rank fails much later and much worse.
    expects_ndim: int = 2

    def __init__(self, label_space: LabelSpace | None = None, **params: Any) -> None:
        self.label_space = label_space
        self.params: dict[str, Any] = dict(params)
        self.feature_names: list[str] | None = None
        #: Classes actually present in the training labels, as canonical ids.  Classes in
        #: the space but not here get a zero probability column -- see ``predict_proba``.
        self.trained_classes_: np.ndarray | None = None
        self.is_fitted: bool = False

    # -- introspection -----------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        n = len(self.label_space) if self.label_space else "?"
        return f"{type(self).__name__}(name={self.name!r}, classes={n}, fitted={self.is_fitted})"

    @property
    def classes_(self) -> list[str]:
        """Canonical class names, in ``predict_proba`` column order."""
        self._require_space()
        return list(self.label_space.names)  # type: ignore[union-attr]

    @property
    def n_classes_(self) -> int:
        self._require_space()
        return len(self.label_space)  # type: ignore[arg-type]

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def set_params(self, **kw: Any) -> "BaseModel":
        self.params.update(kw)
        return self

    def supports_target(self, target: str) -> bool:
        return self.supports == "*" or target in self.supports

    # -- guards ------------------------------------------------------------------------
    def _require_space(self) -> None:
        if self.label_space is None:
            raise ValueError(
                f"{type(self).__name__} has no label_space; construct it with "
                f"ModelRegistry.create({self.name!r}, label_space=...) so that "
                f"predict_proba columns have a defined meaning"
            )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(f"{type(self).__name__} is not fitted; call fit() first")

    def _check_target(self, target: str) -> None:
        if not self.supports_target(target):
            raise ValueError(
                f"model {self.name!r} does not support target {target!r} "
                f"(supports: {self.supports})"
            )

    # -- the contract ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        val: tuple[np.ndarray, np.ndarray] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "BaseModel":
        """Fit on ``y`` given as **canonical label-space ids**.

        Use ``FlowDataset.labels(space=...)`` to produce them.  Passing dataset-local ids
        is the mistake this contract exists to prevent, so it is checked: ids outside the
        space are a hard error rather than a silent off-by-one in the columns.
        """
        self._require_space()
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if y.size == 0:
            raise ValueError("cannot fit on an empty training set")
        if y.min() < 0 or y.max() >= self.n_classes_:
            raise ValueError(
                f"y contains ids outside the {self.label_space.target!r} LabelSpace "  # type: ignore[union-attr]
                f"(0..{self.n_classes_ - 1}): saw {int(y.min())}..{int(y.max())}. "
                f"Did you pass ds.labels() instead of ds.labels(space=...)?"
            )
        if feature_names is not None:
            self.feature_names = list(feature_names)
        self.trained_classes_ = np.unique(y)
        self._fit(X, y, sample_weight=sample_weight, val=val)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``(N, len(label_space))``, rows summing to 1, columns in label-space order.

        Classes the training data never contained keep a zero column: a member trained on
        ISCXVPN has never seen ``browsing``, and reporting zero for it is what lets its
        output be stacked with a member that has.
        """
        self._require_fitted()
        proba = np.asarray(self._predict_proba(np.asarray(X)), dtype=np.float64)
        if proba.ndim != 2 or proba.shape[1] != self.n_classes_:
            raise ValueError(
                f"{type(self).__name__}._predict_proba returned {proba.shape}, expected "
                f"(N, {self.n_classes_}) in LabelSpace order"
            )
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Canonical label-space ids of the arg-max class."""
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def predict_names(self, X: np.ndarray) -> list[str]:
        self._require_space()
        return self.label_space.decode(self.predict(X))  # type: ignore[union-attr]

    # -- to implement ------------------------------------------------------------------
    @abstractmethod
    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        val: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Fit on canonical ids.  ``self.trained_classes_`` is already set."""

    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``(N, len(label_space))`` in label-space order, zero columns for unseen classes."""

    # -- persistence -------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        """Extra state a subclass needs restored.  Override alongside ``_load_state``."""
        return {}

    def _load_state(self, state: dict[str, Any]) -> None:
        """Inverse of ``_state``."""

    def save(self, path: str | Path) -> Path:
        """Persist the model **and its label space** to one joblib file.

        The space travels with the artifact deliberately.  A saved ``predict_proba`` whose
        column meaning lives only in a config file somewhere is a latent Phase-5 bug.
        """
        import joblib

        self._require_fitted()
        self._require_space()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "countdown_model": self.name,
                "cls": f"{type(self).__module__}.{type(self).__qualname__}",
                "label_space": self.label_space.to_dict(),  # type: ignore[union-attr]
                "params": self.params,
                "feature_names": self.feature_names,
                "trained_classes": (
                    None if self.trained_classes_ is None else self.trained_classes_.tolist()
                ),
                "state": self._state(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Restore a model saved by :meth:`save`, resolving its class via the registry."""
        import joblib

        blob = joblib.load(Path(path))
        # "qsentinel_model" is the pre-rename key; blobs saved before the project was
        # renamed still load.
        name = blob.get("countdown_model") or blob["qsentinel_model"]
        target_cls = ModelRegistry.get(name) if cls is BaseModel else cls
        model = target_cls(
            label_space=LabelSpace.from_dict(blob["label_space"]), **blob.get("params", {})
        )
        model.feature_names = blob.get("feature_names")
        tc = blob.get("trained_classes")
        model.trained_classes_ = None if tc is None else np.asarray(tc, dtype=np.int64)
        model._load_state(blob.get("state", {}))
        model.is_fitted = True
        return model


def load_model(path: str | Path) -> BaseModel:
    """Load any saved member without knowing its class up front."""
    return BaseModel.load(path)
