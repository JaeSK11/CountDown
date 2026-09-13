#!/usr/bin/env python
"""DoD 2 -- separate the *feature* effect from the *model* effect.

    python scripts/ablate_features_vs_model.py

`flow_gbdt` beats the Draper-Gil baseline, but that comparison changes two things at once:
the classifier (C4.5 -> LightGBM) **and** the representation (23 time-only columns -> 56
`flow_stats` columns).  Until they are varied independently, "our model is better" and "our
features are better" are the same number.

This runs the full grid -- {timeonly, flow_stats} x {C4.5, k-NN, LightGBM} -- with the fold
assignment, segmentation and scenario held fixed, so each effect is readable on its own:

    feature effect = (flow_stats, M) - (timeonly, M)      for a fixed model M
    model effect   = (timeonly, LGBM) - (timeonly, C4.5)  for a fixed representation

Note the two representations are *not* nested: `flow_stats` adds packet-size statistics and
first-N signed sizes but drops the paper's active/idle spans, so this is a comparison of two
designs, not of a subset against its superset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Run as a script (`python scripts/...`), so the repo root is not on sys.path by default.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from countdown.config import get_logger
from countdown.data import load
from countdown.features import get_extractor
from countdown.flows.segment import segment_flows
from scripts.replicate_drapergil import build_scenarios, cross_validate

log = get_logger("ablate")

FEATURES = ("timeonly", "flow_stats")
MODELS = ("flow_c45_paper", "flow_knn_paper", "flow_gbdt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="iscxvpn")
    ap.add_argument("--ftm", type=float, default=15.0)
    ap.add_argument("--scenarios", nargs="+", default=["A1", "A2-nonvpn", "B"])
    ap.add_argument("--features", nargs="+", default=list(FEATURES))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--protocols", nargs="+", default=["paper", "grouped"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--min-packets", type=int, default=2)
    ap.add_argument("--gbdt-rounds", type=int, default=200,
                    help="LightGBM n_estimators; capped so the grid finishes in "
                         "bounded time, and applied identically in every cell")
    ap.add_argument("--gbdt-class-weight", default="balanced",
                    choices=["balanced", "none"],
                    help="flow_gbdt ships class_weight=balanced; the paper models "
                         "train unweighted, so 'none' is the like-for-like setting")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("runs/ablate_features_vs_model.json"))
    args = ap.parse_args()

    ds = load(args.dataset, target="traffic_type")
    windows, seg = segment_flows(ds.flows, args.ftm, min_packets=args.min_packets)
    log.info("ftm=%gs -> %d windows", args.ftm, seg["windows_kept"])

    groups = np.array([w.meta.get("source_file", w.flow_id) for w in windows])
    scenarios = build_scenarios(windows)

    # Build each representation once and reuse it across every scenario/model/protocol.
    X_by_feature = {}
    for feat in args.features:
        ex = get_extractor(feat)
        X_by_feature[feat] = ex.transform(windows)
        log.info("%-11s -> %d columns", feat, X_by_feature[feat].shape[1])

    results = {"dataset": args.dataset, "ftm": args.ftm, "folds": args.folds,
               "seed": args.seed, "gbdt_rounds": args.gbdt_rounds,
               "gbdt_class_weight": args.gbdt_class_weight,
               "segmentation": seg, "runs": []}

    for scenario in args.scenarios:
        if scenario not in scenarios:
            continue
        y_all, class_names = scenarios[scenario]
        keep = y_all >= 0
        for feat in args.features:
            Xs, ys, gs = X_by_feature[feat][keep], y_all[keep], groups[keep]
            for model_name in args.models:
                for protocol in args.protocols:
                    cw = None if args.gbdt_class_weight == "none" else "balanced"
                    mp = ({"n_estimators": args.gbdt_rounds, "class_weight": cw}
                          if model_name == "flow_gbdt" else {})
                    res = cross_validate(Xs, ys, gs, class_names, model_name,
                                         protocol, args.folds, args.seed, model_params=mp)
                    results["runs"].append({
                        "scenario": scenario, "features": feat,
                        "n_features": int(Xs.shape[1]), "model": model_name,
                        "protocol": protocol, "n_classes": len(class_names), **res,
                    })
                    log.info("  %-10s %-11s %-16s %-8s avgPr=%.4f macroF1=%.4f",
                             scenario, feat, model_name, protocol,
                             res.get("avg_precision", float("nan")),
                             res.get("macro_f1", float("nan")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
