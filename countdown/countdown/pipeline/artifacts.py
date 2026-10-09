"""Deployed experts: everything Stage 4 needs at inference time, in one file per expert.

A ``DeployedExpert`` holds the fitted members, each member's calibrator, the combiner, the
**exact input recipe** each member was trained on (extractor, its parameters, windows) and the
confidence threshold for the E2 -> E1 fallback (``PHASE-5.md`` section 10.5).  It is exported
from a finished Phase-5 run (``scripts/export_models.py``) -- no retraining -- and written to
``models_store/<expert>.joblib`` (gitignored: the E2 file carries a 44M-parameter GNN).

Provenance, stated: the members are the Phase-5 **refit-on-train** models, i.e. trained on the
outer train split of the evaluation protocol, not on all labelled data.  That keeps the
reported test numbers valid for exactly these artifacts; a refit on everything is a later step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.ensemble.calibration import ProbaCalibrator
from countdown.ensemble.combiner import SoftVote
from countdown.models.base import BaseModel
from countdown.schema import LabelSpace

log = get_logger(__name__)

DEFAULT_STORE = Path(__file__).resolve().parents[2] / "models_store"
#: Members the ``fast`` mode keeps: the GBDT floor of every expert -- no GPU, no graphs.
FAST_MEMBERS = ("flow_gbdt",)


@dataclass
class MemberRecipe:
    """How to build a member's input from flows, exactly as at training time."""

    features: str                      # extractor name, or "byte_matrix" for the graph member
    feature_params: dict[str, Any] = field(default_factory=dict)
    windows: list[float] | None = None


@dataclass
class DeployedExpert:
    name: str
    target: str
    label_space: LabelSpace
    members: dict[str, BaseModel]
    recipes: dict[str, MemberRecipe]
    calibrators: dict[str, ProbaCalibrator]
    combiner: SoftVote
    #: Confidence below which a routed flow falls back (``full`` / ``fast`` mode); None = never.
    fallback_threshold: dict[str, float | None] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def member_names(self, mode: str = "full") -> list[str]:
        names = list(self.members)
        if mode == "fast":
            names = [m for m in names if m in FAST_MEMBERS] or names[:1]
        return names

    def combine(self, member_proba: dict[str, np.ndarray]) -> np.ndarray:
        """Calibrate each member present, then vote; an absent member (key missing or a NaN
        row) is skipped, which is how ``fast`` mode and flows without byte input work."""
        n = next(iter(member_proba.values())).shape[0]
        blocks = []
        for m in self.members:
            p = member_proba.get(m)
            if p is None:
                blocks.append(np.full((n, len(self.label_space)), np.nan))
                continue
            ok = ~np.isnan(p).any(axis=1)
            q = np.full_like(p, np.nan, dtype=np.float64)
            if ok.any():
                q[ok] = self.calibrators[m].transform(p[ok])
            blocks.append(q)
        return self.combiner.predict_proba(blocks)

    # -- persistence -------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "name": self.name, "target": self.target, "label_space": self.label_space.to_dict(),
            "members": {m: b.to_blob() for m, b in self.members.items()},
            "recipes": {m: vars(r) for m, r in self.recipes.items()},
            "calibrators": self.calibrators, "combiner": self.combiner,
            "fallback_threshold": self.fallback_threshold, "provenance": self.provenance,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DeployedExpert":
        import joblib

        d = joblib.load(Path(path))
        return cls(
            name=d["name"], target=d["target"], label_space=LabelSpace.from_dict(d["label_space"]),
            members={m: BaseModel.from_blob(b) for m, b in d["members"].items()},
            recipes={m: MemberRecipe(**r) for m, r in d["recipes"].items()},
            calibrators=d["calibrators"], combiner=d["combiner"],
            fallback_threshold=d.get("fallback_threshold", {}), provenance=d.get("provenance", {}),
        )


def load_store(store: str | Path | None = None) -> dict[str, DeployedExpert]:
    store = Path(store) if store else DEFAULT_STORE
    out = {}
    for f in sorted(store.glob("E*.joblib")):
        out[f.stem] = DeployedExpert.load(f)
    if not out:
        raise FileNotFoundError(f"no deployed experts in {store}; run scripts/export_models.py")
    return out


def export_expert(name: str, run_dir: str | Path, keep: float = 0.90, seed: int = 42,
                  n_folds: int = 5) -> DeployedExpert:
    """Build ``name``'s deployed expert from a finished Phase-5 run's cached predictions.

    Uses the expert's ``deploy`` members and the calibrated soft vote (the CV-selected
    combiner on every expert).  ``keep`` sets the fallback threshold: the confidence quantile
    that keeps that share of the expert's own (out-of-fold) flows -- 0.90 is the operating
    point recommended in ``PHASE-5.md`` section 10.5.
    """
    import json

    from countdown.ensemble import Ensemble
    from countdown.experts import get_expert

    spec = get_expert(name)
    run_dir = Path(run_dir)
    run_cfg = json.loads((run_dir / "ensemble.json").read_text())
    overrides = _overrides_used(name)
    ens = Ensemble.build(name, members=list(spec.deployed_members), combiner="vote",
                         calibrate="temperature", seed=seed, n_folds=n_folds,
                         overrides=overrides)
    ens.fit(cache_dir=run_dir.parent / f"preds-s{seed}")

    recipes = {}
    for m in ens.members:
        ms = spec.member(m)
        ov = overrides.get(m, {})
        recipes[m] = MemberRecipe(
            features=ms.features, feature_params={**ms.feature_params, **ov.get("feature_params", {})},
            windows=list(ov.get("windows", ms.windows) or []) or None)

    thresholds: dict[str, float | None] = {}
    y_oof = ens.data.y[ens.data.train_rows]
    for mode in ("full", "fast"):
        names = [m for m in ens.members if mode == "full" or m in FAST_MEMBERS] or ens.members[:1]
        vote = SoftVote().fit([ens._blocks("oof")[ens.members.index(m)] for m in names], y_oof)
        conf = vote.predict_proba([ens._blocks("oof")[ens.members.index(m)] for m in names]).max(1)
        thresholds[mode] = float(np.quantile(conf, 1.0 - keep))

    test = ens.test_proba()
    y_te = ens.data.y[ens.data.test_rows]
    from countdown.eval.metrics import compute_metrics

    test_f1 = compute_metrics(y_te, test.argmax(1), len(ens.data.space))["macro_f1"]
    return DeployedExpert(
        name=name, target=spec.target, label_space=ens.data.space,
        members=dict(ens.models), recipes=recipes, calibrators=dict(ens.calibrators),
        combiner=ens.combiner, fallback_threshold=thresholds,
        provenance={
            "exported": time.strftime("%Y-%m-%d %H:%M:%S"), "from_run": str(run_dir),
            "dataset": spec.dataset, "trained_on": "outer train split (refit-on-train)",
            "n_train": int(len(ens.data.train_rows)), "test_macro_f1": round(float(test_f1), 4),
            "combiner": "calibrated soft vote", "keep_for_threshold": keep,
            "run_args": run_cfg.get("args"),
        })


def _overrides_used(name: str) -> dict[str, dict[str, Any]]:
    """The per-flow input overrides the Phase-5 runs trained with."""
    from countdown.ensemble.views import PER_FLOW_OVERRIDES

    return PER_FLOW_OVERRIDES.get(name, {})
