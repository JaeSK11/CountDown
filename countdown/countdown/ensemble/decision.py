"""Per-class decision bias: move an expert's argmax without retraining (Phase 7, task 7.3).

E1's calibrated soft vote under-predicts ``file_transfer`` (``PHASE-6.md`` section 10).  The fix
here is a decision rule, not a training change -- D5 (no class weights) stands: the members, their
calibrators and their probabilities are untouched, and the label is

    argmax_k  log p_k + b_k

with one bias per class, fit by coordinate ascent on the expert's **out-of-fold** predictions
(grouped CV on the train split), never on test.  The biases live in
``configs/decision_bias.yaml`` -- plain numbers, tracked in git -- so the published model zoo
(whose pickles are hash-pinned) stays valid and a bias can be dropped with
``--no-decision-bias``.

The reported probability is the re-normalised ``p_k * exp(b_k)``, so ``confidence`` stays the
probability of the label actually returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from countdown.config import get_logger

log = get_logger(__name__)

DEFAULT_BIAS_PATH = Path(__file__).resolve().parents[2] / "configs" / "decision_bias.yaml"
GRID = np.linspace(-3.0, 3.0, 25)


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """``p * exp(b)``, re-normalised per row; NaN rows (abstentions) stay NaN."""
    q = np.asarray(proba, dtype=np.float64) * np.exp(np.asarray(bias, dtype=np.float64))[None, :]
    return q / q.sum(axis=1, keepdims=True)


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int, weight: np.ndarray | None = None) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, average="macro", labels=range(k), sample_weight=weight,
                          zero_division=0))


def fit_class_bias(proba: np.ndarray, y: np.ndarray, iters: int = 3,
                   grid: Sequence[float] = GRID) -> np.ndarray:
    """Coordinate ascent on macro-F1 over one bias per class (mean-centred).  Deterministic:
    classes in order, a move is taken only if it strictly improves the score."""
    k = proba.shape[1]
    logp = np.log(np.clip(proba, 1e-12, None))
    b = np.zeros(k)
    best = macro_f1(y, logp.argmax(1), k)
    for _ in range(iters):
        moved = False
        for c in range(k):
            keep = b[c]
            for v in grid:
                b[c] = v
                s = macro_f1(y, (logp + b).argmax(1), k)
                if s > best + 1e-9:
                    best, keep, moved = s, v, True
            b[c] = keep
        if not moved:
            break
    return b - b.mean()


def load_biases(path: str | Path = DEFAULT_BIAS_PATH) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as fh:
        return (yaml.safe_load(fh) or {}).get("experts", {})


def bias_for(expert: str, class_names: Sequence[str],
             biases: dict[str, dict[str, Any]] | None = None) -> np.ndarray | None:
    """The bias vector for ``expert`` in ``class_names`` order, or None.  A config whose class
    list does not match the deployed label space is refused rather than applied to the wrong
    columns."""
    entry = (load_biases() if biases is None else biases).get(expert)
    if not entry:
        return None
    b = entry["bias"]
    if set(b) != set(class_names):
        log.warning("decision_bias.yaml: %s classes %s do not match the deployed %s; ignored",
                    expert, sorted(b), sorted(class_names))
        return None
    return np.array([float(b[c]) for c in class_names])
