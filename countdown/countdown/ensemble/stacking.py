"""Out-of-fold base predictions -- the leakage-safe input to a stacking combiner.

The trap (``PHASE-5.md`` §5.3): a meta-learner trained on base predictions made over the
rows those bases were trained on learns how well each member *memorised*, not how well it
generalises.  So for each of K group-aware folds of the training set, the member is fit on
the other K-1 folds and predicts the held fold; the assembled out-of-fold (OOF) matrix
trains the calibrators and the meta-learner, and the member is then refit on all of train
for inference.

Folds are ``StratifiedGroupKFold`` over the expert's group key (``source_file`` for ISCX,
``capture_day`` for CSTNET), so a capture never sits on both sides of a fold any more than
it does across the outer train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from countdown.config import get_logger

log = get_logger(__name__)

#: ``fit_predict(train_rows, predict_rows) -> (len(predict_rows), C)``.
FitPredict = Callable[[np.ndarray, np.ndarray], np.ndarray]


def group_folds(y: np.ndarray, groups: np.ndarray, n_folds: int = 5,
                seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(fit_idx, held_idx), ...]`` positions into ``y``; groups never straddle a fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    n_groups = len(set(groups.tolist()))
    # Stratification needs every class in every fold: MobileApp has 3 training captures
    # per class, so it gets 3 folds (2 captures per class to fit on), not 5.
    min_class = int(np.bincount(np.unique(y, return_inverse=True)[1]).min())
    k = min(int(n_folds), n_groups, min_class)
    if k < 2:
        raise ValueError(f"cannot make out-of-fold predictions from {n_groups} group(s) "
                         f"with {min_class} sample(s) in the thinnest class")
    if k < n_folds:
        log.warning("[stacking] %d groups, thinnest class %d: %d folds instead of %d",
                    n_groups, min_class, k, n_folds)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [(np.sort(a), np.sort(b)) for a, b in splitter.split(np.zeros(len(y)), y, groups)]
    for a, b in folds:
        leaked = set(groups[a].tolist()) & set(groups[b].tolist())
        if leaked:  # pragma: no cover - StratifiedGroupKFold guarantees this
            raise AssertionError(f"group leak inside an OOF fold: {sorted(leaked)[:5]}")
    return folds


@dataclass
class OOF:
    """Out-of-fold predictions for ``rows`` (global row ids), in ``rows`` order."""

    rows: np.ndarray
    proba: np.ndarray
    fold: np.ndarray            # which fold predicted each row


def oof_predict(fit_predict: FitPredict, rows: np.ndarray, y: np.ndarray, groups: np.ndarray,
                n_classes: int, n_folds: int = 5, seed: int = 42) -> OOF:
    """Run ``fit_predict`` once per fold; every row is predicted by a model that never saw it.

    ``rows``, ``y`` and ``groups`` are parallel: ``y[i]`` / ``groups[i]`` describe global row
    ``rows[i]``.  ``fit_predict`` receives global row ids.
    """
    rows = np.asarray(rows)
    proba = np.full((len(rows), n_classes), np.nan)
    fold_of = np.full(len(rows), -1, dtype=np.int64)
    for k, (fit_pos, held_pos) in enumerate(group_folds(y, groups, n_folds, seed)):
        log.info("[stacking] fold %d: fit %d rows, predict %d", k, len(fit_pos), len(held_pos))
        p = np.asarray(fit_predict(rows[fit_pos], rows[held_pos]), dtype=np.float64)
        if p.shape != (len(held_pos), n_classes):
            raise ValueError(f"fold {k}: fit_predict returned {p.shape}, "
                             f"expected {(len(held_pos), n_classes)}")
        proba[held_pos] = p
        fold_of[held_pos] = k
    if (fold_of < 0).any():  # pragma: no cover - the folds partition the rows
        raise AssertionError("some rows were never predicted out of fold")
    return OOF(rows=rows, proba=proba, fold=fold_of)
