#!/usr/bin/env python
"""Replicate Draper-Gil et al., ICISSP 2016, and check the published numbers.

    python scripts/replicate_drapergil.py --ftm 15 30 60 120

What is reproduced
------------------
* the paper's **features** (23 time-related columns, ``timeonly`` extractor),
* the paper's **classifiers** (C4.5 proxy, k-NN / IBk proxy),
* the paper's **flow-timeout sweep** (15/30/60/120 s, by segmenting the timeline),
* the paper's **scenarios** -- A1 VPN-vs-non-VPN, A2 characterise each side separately,
  B one flat pass over the joint traffic x tunnel label,
* the paper's **protocol** -- stratified 10-fold CV drawn at random over flows.

That last point is the reason this script exists twice over.  Random 10-fold CV lets
windows cut from one conversation -- often one TCP connection -- fall on both sides of the
split, so it measures partly memorisation.  Every configuration is therefore run under
**both** protocols: ``paper`` (random 10-fold, to check their number) and ``grouped``
(folds split by ``source_file``, to get an honest one).  The gap between the two columns
is the finding.

Known fidelity gaps, all reported rather than hidden:
* ISCXVPN2016 as distributed ships **no browsing captures**, so the paper's 7th category
  cannot be reproduced at all, and its single ``p2p`` capture is a VPN one.
* scikit-learn has no J48 pessimistic-error pruning (see ``models/paper_baselines.py``).
* the activity threshold behind ``active``/``idle`` is unstated in the paper; 5 s is
  ISCXFlowMeter's inherited default.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qsentinel.config import get_logger
from qsentinel.data import load
from qsentinel.features import get_extractor
from qsentinel.flows.segment import segment_flows
from qsentinel.models import ModelRegistry
from qsentinel.schema import LabelSpace

log = get_logger("replicate")

#: Values read off the paper (Section 5) for the configurations it states outright.
#: ``avg_precision`` is macro-averaged precision, the metric the paper reports.
PUBLISHED: dict[tuple[str, str, int], float] = {
    ("A1", "flow_c45_paper", 15): 0.898,   # mean of VPN 0.890 / non-VPN 0.906
    ("A1", "flow_c45_paper", 120): 0.874,  # mean of 0.860 / 0.887
    ("A1", "flow_knn_paper", 15): 0.847,   # mean of 0.848 / 0.846
    ("A1", "flow_knn_paper", 120): 0.826,  # mean of 0.815 / 0.837
    ("A2-nonvpn", "flow_c45_paper", 15): 0.89,
    ("A2-vpn", "flow_c45_paper", 15): 0.84,
    ("B", "flow_c45_paper", 15): 0.783,    # "highest average Pr from the different ftm values"
    ("B", "flow_knn_paper", 15): 0.711,
}

MODELS = ("flow_c45_paper", "flow_knn_paper")


# ----------------------------------------------------------------------------------------
# folds
# ----------------------------------------------------------------------------------------
def paper_folds(y: np.ndarray, n_splits: int, seed: int):
    """The paper's protocol: stratified 10-fold CV drawn at random over flows."""
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y
    )


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Same fold count, but no capture file may straddle a fold boundary."""
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups
    )


# ----------------------------------------------------------------------------------------
# one cross-validated configuration
# ----------------------------------------------------------------------------------------
def cross_validate(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, class_names: list[str],
    model_name: str, protocol: str, n_splits: int, seed: int,
    model_params: dict | None = None,
) -> dict:
    """Macro precision / recall / F1 pooled over folds, plus per-class precision.

    Predictions are pooled across folds before scoring rather than averaged fold-by-fold:
    with thin classes some folds carry none of a class, and averaging per-fold macro
    scores would weight those folds as if the class did not exist.
    """
    from sklearn.metrics import precision_recall_fscore_support

    space = LabelSpace.from_names("replication", class_names)
    folds = (
        paper_folds(y, n_splits, seed) if protocol == "paper"
        else grouped_folds(y, groups, n_splits, seed)
    )

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    n_folds = 0

    for train_idx, test_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            continue  # a grouped fold can leave one class in train; nothing to learn
        model = ModelRegistry.create(
            model_name, label_space=space, seed=seed, **(model_params or {})
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred_all.append(model.predict(X[test_idx]))
        y_true_all.append(y[test_idx])
        n_folds += 1

    if not n_folds:
        return {"error": "no usable folds", "n_folds": 0}

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    labels = np.arange(len(class_names))
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    seen = sup > 0  # classes with no test samples must not drag the macro average to 0

    return {
        "n_folds": n_folds,
        "n_test": int(len(y_true)),
        "avg_precision": float(p[seen].mean()),
        "avg_recall": float(r[seen].mean()),
        "macro_f1": float(f[seen].mean()),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": {
            class_names[i]: {"precision": round(float(p[i]), 4),
                       "recall": round(float(r[i]), 4),
                       "support": int(sup[i])}
            for i in labels if seen[i]
        },
    }


# ----------------------------------------------------------------------------------------
# scenario construction
# ----------------------------------------------------------------------------------------
def build_scenarios(flows) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Label vectors for each of the paper's scenarios, as (row mask, class names).

    Returns ``{scenario: (labels_or_None_per_row, names)}`` where a ``-1`` label marks a
    row the scenario excludes.
    """
    traffic = np.array([f.label_fields["traffic_type"] for f in flows])
    is_vpn = np.array([bool(f.label_fields["is_vpn"]) for f in flows])

    out: dict[str, tuple[np.ndarray, list[str]]] = {}

    # A1 -- VPN detection.
    out["A1"] = (is_vpn.astype(np.int64), ["non-vpn", "vpn"])

    # A2 -- characterise each side on its own.
    for side, mask in (("nonvpn", ~is_vpn), ("vpn", is_vpn)):
        names = sorted(set(traffic[mask]))
        idx = {n: i for i, n in enumerate(names)}
        y = np.full(len(flows), -1, dtype=np.int64)
        y[mask] = [idx[t] for t in traffic[mask]]
        out[f"A2-{side}"] = (y, names)

    # B -- one flat pass over the joint label.
    joint = np.array([("vpn-" if v else "") + t for t, v in zip(traffic, is_vpn)])
    names = sorted(set(joint))
    idx = {n: i for i, n in enumerate(names)}
    out["B"] = (np.array([idx[j] for j in joint], dtype=np.int64), names)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="iscxvpn")
    ap.add_argument("--ftm", type=float, nargs="+", default=[15, 30, 60, 120],
                    help="flow-timeout values to sweep (seconds)")
    ap.add_argument("--scenarios", nargs="+", default=["A1", "A2-nonvpn", "A2-vpn", "B"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--protocols", nargs="+", default=["paper", "grouped"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--activity-timeout", type=float, default=5.0)
    ap.add_argument("--min-packets", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("runs/replicate_drapergil.json"))
    args = ap.parse_args()

    ds = load(args.dataset, target="traffic_type")
    log.info("loaded %d flows from %s", len(ds), args.dataset)

    extractor = get_extractor("timeonly", activity_timeout=args.activity_timeout)
    names = list(extractor.feature_names)

    results: dict = {
        "dataset": args.dataset,
        "features": {"extractor": "timeonly", "n": len(names),
                     "activity_timeout": args.activity_timeout},
        "folds": args.folds,
        "seed": args.seed,
        "runs": [],
    }

    for ftm in args.ftm:
        windows, seg_stats = segment_flows(ds.flows, ftm, min_packets=args.min_packets)
        log.info("ftm=%gs -> %d windows (%d dropped)", ftm,
                 seg_stats["windows_kept"], seg_stats["windows_dropped_min_packets"])

        X = extractor.transform(windows)
        groups = np.array([w.meta.get("source_file", w.flow_id) for w in windows])
        scenarios = build_scenarios(windows)

        for scenario in args.scenarios:
            if scenario not in scenarios:
                continue
            y_all, class_names = scenarios[scenario]
            keep = y_all >= 0
            Xs, ys, gs = X[keep], y_all[keep], groups[keep]

            for model_name in args.models:
                for protocol in args.protocols:
                    res = cross_validate(Xs, ys, gs, class_names, model_name,
                                         protocol, args.folds, args.seed)
                    row = {"ftm": ftm, "scenario": scenario, "model": model_name,
                           "protocol": protocol, "n_classes": len(class_names),
                           "segmentation": seg_stats, **res}
                    row["published_avg_precision"] = PUBLISHED.get(
                        (scenario, model_name, int(ftm))
                    )
                    results["runs"].append(row)
                    log.info(
                        "  %-10s %-16s %-8s avgPr=%.4f  macroF1=%.4f  (pub %s)",
                        scenario, model_name, protocol,
                        res.get("avg_precision", float("nan")),
                        res.get("macro_f1", float("nan")),
                        row["published_avg_precision"],
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
