"""Scoring, with macro-F1 as the headline and a baseline alongside it.

Accuracy is close to useless on these targets.  Pooled ISCX ``traffic_type`` runs about
30:1 between its largest and smallest class, so a model that only ever answers
``browsing`` or ``p2p`` scores ~70% accuracy while being worthless.  Macro-F1 weights every
class equally and is what every table in this project reports.

For the same reason a majority-class baseline is computed next to every result: a macro-F1
of 0.42 means nothing until you know the baseline is 0.09.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from countdown.schema import LabelSpace


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Accuracy, macro/weighted F1, per-class P/R/F1/support, and the confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    names = list(class_names) if class_names is not None else [str(i) for i in labels]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # macro over classes PRESENT in y_true; a class with no support would otherwise
        # drag the mean toward zero for being absent rather than for being wrong.
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                labels=sorted(set(np.unique(y_true).tolist())),
                average="macro", zero_division=0,
            )
        ),
        "macro_f1_all_classes": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": names,
    }


def baseline_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    class_names: Sequence[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Majority-class and stratified-random baselines on the same test set."""
    from sklearn.dummy import DummyClassifier

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    X_train = np.zeros((len(y_train), 1))
    X_test = np.zeros((len(y_test), 1))

    out: dict[str, Any] = {}
    for strategy in ("most_frequent", "stratified"):
        dummy = DummyClassifier(strategy=strategy, random_state=seed)
        dummy.fit(X_train, y_train)
        m = compute_metrics(y_test, dummy.predict(X_test), n_classes, class_names)
        out[strategy] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        }
    return out


def evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    space: LabelSpace | None = None,
    y_train: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Score a fitted model on ``(X, y)`` in canonical label-space ids.

    ``y_train`` is optional but worth passing: the majority baseline is only honest when
    its majority class comes from the training set, not from the test set it is scored on.
    """
    space = space or getattr(model, "label_space", None)
    if space is None:
        raise ValueError("evaluate() needs a LabelSpace (pass space= or bind one to the model)")

    y = np.asarray(y)
    proba = model.predict_proba(X)
    y_pred = np.argmax(proba, axis=1)

    report = compute_metrics(y, y_pred, len(space), space.names)
    report["target"] = space.target
    report["n_classes"] = len(space)

    base = baseline_metrics(
        y_train if y_train is not None else y, y, len(space), space.names, seed=seed
    )
    report["baseline"] = base
    report["baseline_macro_f1"] = base["most_frequent"]["macro_f1"]
    report["baseline_accuracy"] = base["most_frequent"]["accuracy"]
    report["beats_baseline"] = bool(report["macro_f1"] > report["baseline_macro_f1"])

    # Mean max-probability: a cheap confidence read the Phase-5 router will want.
    report["mean_confidence"] = float(np.mean(np.max(proba, axis=1)))
    return report


def metrics_excluding_tail(
    report: dict[str, Any], min_support: int = 50
) -> dict[str, Any]:
    """Re-derive macro-F1 over classes with at least ``min_support`` test samples.

    CSTNET's app target has a severe tail (``chia.net`` has 16 flows in total), so the
    120-class macro-F1 is dominated by classes with a handful of samples.  Reporting both
    numbers is honest; reporting only the flattering one is not.
    """
    kept = {
        n: d for n, d in report["per_class"].items() if d["support"] >= min_support
    }
    if not kept:
        return {"min_support": min_support, "n_classes": 0, "macro_f1": None}
    return {
        "min_support": min_support,
        "n_classes": len(kept),
        "n_classes_dropped": len(report["per_class"]) - len(kept),
        "macro_f1": float(np.mean([d["f1"] for d in kept.values()])),
        "classes": sorted(kept),
    }
