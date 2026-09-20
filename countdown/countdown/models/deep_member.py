"""Shared skin for the *recommended* deep members (decision D3, 2026-09-19).

Replicated paper baselines keep their own paper-faithful loops; every recommended member
trains through :mod:`countdown.training.deep` so that they are compared on identical
optimisation -- AdamW, linear warmup/decay, gradient clipping, AMP, model selection on
validation macro-F1.  This class is the glue between that trainer and the
:class:`~countdown.models.base.BaseModel` contract for members whose input is one dense
array: subclasses implement :meth:`_build` (the network) and, if needed, :meth:`_prep`
(array -> network input), and nothing else.

Class weights are **off** for every member (decision D5): Phase 5 calibrates on the true
prior.  ``class_weight="balanced"`` is still accepted so the choice can be measured.

The final epoch is reported unless early stopping is requested (``early_stop_patience >
0``) or ``restore_best=True`` is passed: on the small-capture tasks these members serve, a
grouped validation fold is a handful of captures, and selecting an epoch on it measurably
hurts (VPN traffic type: 0.80 +/- 0.21 restored vs 0.93 +/- 0.03 final, five seeds).  The
validation curve is still recorded in ``fit_result_.history``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.models.base import BaseModel
from countdown.training.deep import DeepConfig, DeepTrainer

log = get_logger(__name__)

#: ``params`` keys that are routed to :class:`DeepConfig`, with the aliases the rest of the
#: project already uses (``lr``, ``early_stop_patience``, ``workers``).
_TRAINER_KEYS = {
    "epochs": "epochs", "batch_size": "batch_size", "eval_batch_size": "eval_batch_size",
    "lr": "learning_rate", "learning_rate": "learning_rate", "weight_decay": "weight_decay",
    "warmup_ratio": "warmup_ratio", "max_grad_norm": "max_grad_norm",
    "early_stop_patience": "patience", "patience": "patience", "seed": "seed",
    "device": "device", "amp": "amp", "workers": "num_workers", "num_workers": "num_workers",
    "checkpoint_path": "checkpoint_path", "log_every": "log_every",
    "restore_best": "restore_best",
}


class ArrayDataset:
    """``{"x": tensor, "y": label}`` rows over one dense array.  Imports torch lazily."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        self.X = X
        self.y = None if y is None else np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def labels(self) -> np.ndarray:
        return self.y if self.y is not None else np.zeros(len(self), dtype=np.int64)

    def __getitem__(self, i: int) -> dict:
        import torch

        item = {"x": torch.from_numpy(np.ascontiguousarray(self.X[i]))}
        if self.y is not None:
            item["y"] = torch.tensor(int(self.y[i]), dtype=torch.long)
        return item


class DeepArrayMember(BaseModel):
    """A recommended member: one dense input array, trained by :class:`DeepTrainer`."""

    #: Trainer defaults a subclass overrides; anything in ``params`` wins over these.
    deep_defaults: dict[str, Any] = {}

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        self.params.setdefault("class_weight", None)
        self.net = None
        self.fit_result_ = None
        self._sample_shape: tuple[int, ...] | None = None

    # -- to implement ------------------------------------------------------------------
    def _build(self, n_classes: int, sample_shape: tuple[int, ...]):
        raise NotImplementedError

    def _prep(self, X: np.ndarray) -> np.ndarray:
        """Array as the network wants it.  Default: float32, unchanged shape."""
        return np.asarray(X, dtype=np.float32)

    # -- plumbing ----------------------------------------------------------------------
    def _deep_config(self) -> DeepConfig:
        kw: dict[str, Any] = dict(self.deep_defaults)
        for key, value in self.params.items():
            if key in _TRAINER_KEYS and value is not None:
                kw[_TRAINER_KEYS[key]] = value
        cw = self.params.get("class_weight")
        if cw not in (None, "none", "balanced"):
            raise ValueError(f"class_weight must be None or 'balanced', got {cw!r}")
        kw["class_weighted_loss"] = cw == "balanced"
        # Select on validation only when early stopping was asked for.  With a fixed
        # schedule the final epoch is reported, as the papers do: on the small-capture
        # tasks these members serve, a grouped val fold is too noisy to pick an epoch with
        # (see DeepConfig.restore_best).  An explicit ``restore_best`` param still wins.
        kw.setdefault("restore_best", bool(kw.get("patience", 0)))
        return DeepConfig(**kw)

    @staticmethod
    def _forward(model, batch):
        return model(batch["x"])

    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        Xp = self._prep(X)
        self._sample_shape = tuple(int(d) for d in Xp.shape[1:])
        self.net = self._build(self.n_classes_, self._sample_shape)
        cfg = self._deep_config()

        class_weights = None
        if cfg.class_weighted_loss:
            counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
            class_weights = np.where(
                counts > 0, len(y) / (np.count_nonzero(counts) * np.maximum(counts, 1)), 0.0)

        val_ds = None
        if val is not None and len(val[1]):
            val_ds = ArrayDataset(self._prep(val[0]), val[1])
        self.fit_result_ = DeepTrainer(cfg).fit(
            self.net, ArrayDataset(Xp, y), n_classes=self.n_classes_, val_ds=val_ds,
            forward_fn=self._forward, class_weights=class_weights,
        )
        r = self.fit_result_
        log.info("[%s] fit %.1fs, best epoch %d, val macro-F1 %.4f, %.2fM params",
                 self.name, r.train_seconds, r.best_epoch, r.best_val_macro_f1, r.n_params / 1e6)

    def _predict_proba(self, X) -> np.ndarray:
        return DeepTrainer(self._deep_config()).predict_proba(
            self.net, ArrayDataset(self._prep(X)), forward_fn=self._forward)

    def n_params(self) -> int:
        return 0 if self.net is None else sum(p.numel() for p in self.net.parameters())

    # -- persistence -------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        if self.net is None:
            return {}
        r = self.fit_result_
        return {
            "state_dict": {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()},
            "sample_shape": list(self._sample_shape or ()),
            "history": [] if r is None else r.history,
            "best_epoch": None if r is None else r.best_epoch,
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        import torch

        self._sample_shape = tuple(state["sample_shape"])
        self.net = self._build(self.n_classes_, self._sample_shape)
        self.net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state["state_dict"].items()})
