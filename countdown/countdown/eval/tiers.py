"""Readiness tiers: score predictions on the decision they drive, not the raw label.

Confusing two classes in the same tier (``audio_streaming`` for ``video_streaming``) changes
nothing about what the customer migrates first, so it should not cost anything.  The mapping
lives in ``configs/readiness_tiers.yaml``; models keep predicting raw classes and this module
folds them into tiers afterwards.

Tier probability is the sum of its classes' probabilities, not a relabel of the argmax: a flow
at 0.4 ``email`` / 0.35 ``file_transfer`` is a confident ``migrate_first`` even though no single
class is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from countdown.eval.metrics import compute_metrics

DEFAULT_TIERS_PATH = Path(__file__).resolve().parents[2] / "configs" / "readiness_tiers.yaml"


def load_tiers(path: str | Path = DEFAULT_TIERS_PATH) -> dict[str, list[str]]:
    """``{tier: [class, ...]}`` in priority order; every class may appear in only one tier."""
    with open(path) as fh:
        tiers = yaml.safe_load(fh)["tiers"]
    seen: set[str] = set()
    for tier, classes in tiers.items():
        dup = seen.intersection(classes)
        if dup:
            raise ValueError(f"classes {sorted(dup)} are in more than one tier (last: {tier})")
        seen.update(classes)
    return {tier: list(classes) for tier, classes in tiers.items()}


def tier_index(class_names: Sequence[str], tiers: dict[str, list[str]]) -> np.ndarray:
    """For each class position, the index of its tier.  Every class must be mapped."""
    lookup = {c: i for i, classes in enumerate(tiers.values()) for c in classes}
    missing = [c for c in class_names if c not in lookup]
    if missing:
        raise ValueError(f"classes with no tier: {missing}")
    return np.array([lookup[c] for c in class_names], dtype=np.int64)


def to_tier_proba(
    proba: np.ndarray, class_names: Sequence[str], tiers: dict[str, list[str]]
) -> np.ndarray:
    """``(n, n_classes)`` class probabilities -> ``(n, n_tiers)`` by summing within each tier."""
    idx = tier_index(class_names, tiers)
    out = np.zeros((proba.shape[0], len(tiers)), dtype=proba.dtype)
    np.add.at(out.T, idx, np.asarray(proba).T)
    return out


def tier_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_names: Sequence[str],
    tiers: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """The usual metrics dict, computed on tiers instead of raw classes."""
    tiers = tiers if tiers is not None else load_tiers()
    idx = tier_index(class_names, tiers)
    y_tier = idx[np.asarray(y_true)]
    y_pred = to_tier_proba(proba, class_names, tiers).argmax(axis=1)
    return compute_metrics(y_tier, y_pred, len(tiers), class_names=list(tiers))
