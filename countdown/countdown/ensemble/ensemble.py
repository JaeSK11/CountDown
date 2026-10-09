"""The ``Ensemble``: calibrated members + a combiner for one expert, with a per-flow breakdown.

``fit`` does the whole leakage-safe protocol on the expert's outer **train** rows:

1. per member, K-fold out-of-fold predictions over train (``stacking.oof_predict``);
2. per member, a refit on all of train -- the model that scores test and new flows;
3. per member, a calibrator fit on its OOF predictions (true class distribution, D5);
4. the combiner fit on the calibrated OOF blocks.

The single temperature per member is fit on the same OOF rows the combiner then trains on.
One scalar cannot memorise rows, so this is not the leak §5.3 is about; isotonic would be,
which is one more reason it is not the default.

Member predictions are the expensive part (a GPU member is 5 + 1 fits), so ``fit`` can
cache each member's OOF/test predictions and refit model under ``cache_dir`` and resume
from them -- a dropped GPU (``PHASE-4-HANDOFF.md`` §0) then costs one member, not the run.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.ensemble.calibration import ProbaCalibrator, check_convention
from countdown.ensemble.combiner import make_combiner
from countdown.ensemble.stacking import oof_predict
from countdown.ensemble.views import ExpertData, build_expert_data
from countdown.models.base import BaseModel

log = get_logger(__name__)


@dataclass
class MemberPreds:
    name: str
    oof: np.ndarray          # (len(train_rows), C)
    fold: np.ndarray
    test: np.ndarray         # (len(test_rows), C)
    seconds: float


def _rows_digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def member_predictions(data: ExpertData, name: str, n_folds: int = 5, seed: int = 42,
                       cache_dir: str | Path | None = None,
                       keep_model: bool = True) -> tuple[MemberPreds, BaseModel | None]:
    """OOF over train + refit-on-train test predictions for one member (cached if asked)."""
    view = data.views[name]
    digest = _rows_digest(data.train_rows, data.test_rows, [n_folds, seed])
    npz = Path(cache_dir) / f"{name}.npz" if cache_dir else None
    blob = Path(cache_dir) / f"{name}.joblib" if cache_dir else None
    if npz is not None and npz.exists():
        d = np.load(npz, allow_pickle=False)
        if str(d["digest"]) == digest:
            log.info("[ensemble] %s: cached predictions %s", name, npz)
            model = BaseModel.load(blob) if keep_model and blob.exists() else None
            return MemberPreds(name, d["oof"], d["fold"], d["test"], float(d["seconds"])), model
        log.warning("[ensemble] %s: cache %s is for another split; recomputing", name, npz)

    t0 = time.time()
    tr = data.train_rows

    def fit_predict(fit_rows: np.ndarray, held: np.ndarray) -> np.ndarray:
        # Each finished fold is kept on disk: a GIN fold is ~15 min, and losing four of
        # them to one interrupted process is what per-member caching alone allowed.
        f = Path(cache_dir) / f"{name}.fold-{_rows_digest(fit_rows, held)}.npy" if cache_dir else None
        if f is not None and f.exists():
            log.info("[ensemble] %s: cached fold %s", name, f.name)
            return np.load(f)
        p = view.predict(view.fit(fit_rows), held)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, p)
        return p

    oof = oof_predict(fit_predict, tr, data.y[tr], data.groups[tr], len(data.space),
                      n_folds, seed)
    model = view.fit(tr)
    test = view.predict(model, data.test_rows)
    preds = MemberPreds(name, oof.proba, oof.fold, test, time.time() - t0)
    if npz is not None:
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, oof=preds.oof, fold=preds.fold, test=preds.test,
                 seconds=preds.seconds, digest=digest)
        model.save(blob)
    return preds, model


class Ensemble:
    """Calibrated members of one expert plus a combiner.  See the module docstring."""

    def __init__(self, data: ExpertData, members: list[str] | None = None,
                 combiner: str = "stack_logreg", calibrate: str | bool = "temperature",
                 n_folds: int = 5, seed: int = 42) -> None:
        self.data = data
        self.members = list(members or data.views)
        missing = [m for m in self.members if m not in data.views]
        if missing:
            raise KeyError(f"{data.expert} has no view for {missing}; skipped: {data.skipped}")
        self.combiner_kind = combiner
        self.calibrate = "temperature" if calibrate is True else (calibrate or "none")
        self.n_folds = n_folds
        self.seed = seed
        self.preds: dict[str, MemberPreds] = {}
        self.models: dict[str, BaseModel] = {}
        self.calibrators: dict[str, ProbaCalibrator] = {}
        self.combiner = None

    @classmethod
    def build(cls, expert: str = "E1", members: list[str] | None = None,
              combiner: str = "stacking", calibrate: str | bool = True, seed: int = 42,
              n_folds: int = 5, **data_kw: Any) -> "Ensemble":
        from countdown.experts import get_expert

        spec = get_expert(expert)
        members = list(members) if members else list(spec.deployed_members)
        data = build_expert_data(spec, members=members, seed=seed, **data_kw)
        return cls(data, members=members, combiner=combiner, calibrate=calibrate,
                   n_folds=n_folds, seed=seed)

    # -- fit ---------------------------------------------------------------------------
    def fit(self, cache_dir: str | Path | None = None) -> "Ensemble":
        for m in self.members:
            preds, model = member_predictions(self.data, m, self.n_folds, self.seed, cache_dir)
            self.preds[m] = preds
            if model is not None:
                self.models[m] = model
        if self.models:
            check_convention(self.models)
        y_tr = self.data.y[self.data.train_rows]
        for m in self.members:
            ok = ~np.isnan(self.preds[m].oof).any(axis=1)
            self.calibrators[m] = ProbaCalibrator(self.calibrate).fit(
                self.preds[m].oof[ok], y_tr[ok])
        if len(self.members) == 1:
            # One member (E1 ships the GBDT alone): a weight-1 vote is the calibrated member.
            self.combiner = make_combiner("vote", seed=self.seed).fit(self._blocks("oof"), y_tr)
        else:
            self.combiner = make_combiner(self.combiner_kind, seed=self.seed).fit(
                self._blocks("oof"), y_tr)
        return self

    def _blocks(self, which: str) -> list[np.ndarray]:
        out = []
        for m in self.members:
            p = getattr(self.preds[m], which)
            ok = ~np.isnan(p).any(axis=1)
            q = np.full_like(p, np.nan)
            q[ok] = self.calibrators[m].transform(p[ok])
            out.append(q)
        return out

    # -- predict -----------------------------------------------------------------------
    def combine(self, member_proba: dict[str, np.ndarray]) -> np.ndarray:
        """Final proba from raw member probabilities (calibrated here, then combined)."""
        blocks = []
        for m in self.members:
            p = np.asarray(member_proba[m], dtype=np.float64)
            ok = ~np.isnan(p).any(axis=1)
            q = np.full_like(p, np.nan)
            q[ok] = self.calibrators[m].transform(p[ok])
            blocks.append(q)
        return self.combiner.predict_proba(blocks)

    def predict_rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Member probabilities for rows of the expert's table, from the refit models."""
        return {m: self.data.views[m].predict(self.models[m], rows) for m in self.members}

    def test_proba(self) -> np.ndarray:
        return self.combine({m: self.preds[m].test for m in self.members})

    def explain(self, member_proba: dict[str, np.ndarray], top_k: int = 1) -> list[dict[str, Any]]:
        """``{label, confidence, per_member: {name: {label, confidence}}}`` per row."""
        names = self.data.space.names
        final = self.combine(member_proba)
        out = []
        for i in range(final.shape[0]):
            per = {}
            for m in self.members:
                p = self.calibrators[m].transform(member_proba[m][i:i + 1])[0] \
                    if not np.isnan(member_proba[m][i]).any() else None
                per[m] = None if p is None else {"label": names[int(p.argmax())],
                                                 "confidence": round(float(p.max()), 4)}
            out.append({"label": names[int(final[i].argmax())],
                        "confidence": round(float(final[i].max()), 4), "per_member": per})
        return out

    # -- persistence -------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Members (as model blobs), calibrators and combiner -- not the training data."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "expert": self.data.expert, "label_space": self.data.space.to_dict(),
            "members": self.members, "combiner_kind": self.combiner_kind,
            "calibrate": self.calibrate, "models": {m: b.to_blob() for m, b in self.models.items()},
            "calibrators": self.calibrators, "combiner": self.combiner,
        }, path)
        return path

    def describe(self) -> dict[str, Any]:
        return {
            "expert": self.data.expert, "members": self.members, "combiner": self.combiner_kind,
            "calibration": {m: c.describe() for m, c in self.calibrators.items()},
            "weights": (None if not hasattr(self.combiner, "weights_") else
                        dict(zip(self.members, np.round(self.combiner.weights_, 4).tolist()))),
            "data": self.data.describe(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return json.dumps({"expert": self.data.expert, "members": self.members,
                           "combiner": self.combiner_kind}, indent=None)
