#!/usr/bin/env python
"""Fit an expert's per-class decision bias on its out-of-fold predictions (Phase 7, task 7.3).

    python scripts/fit_decision_bias.py                 # E1 -> configs/decision_bias.yaml
    python scripts/fit_decision_bias.py --dry-run       # report only

Rebuilds the deployed expert's calibrated vote from the cached Phase-5 predictions (as
``scripts/export_models.py`` does -- no retraining), fits the bias on the out-of-fold
predictions of the train split, and reports test-split metrics before/after **by flows and by
bytes** (the test labels are used only for this report).  Writes
``results/phase7/decision_bias_<expert>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from countdown.ensemble.decision import DEFAULT_BIAS_PATH, apply_bias, fit_class_bias, macro_f1  # noqa: E402


def _report(y: np.ndarray, proba: np.ndarray, w: np.ndarray, names: list[str], focus: str) -> dict:
    k, f = len(names), names.index(focus)
    pred = proba.argmax(1)
    m = y == f
    true_f = np.bincount(y, minlength=k) / len(y)
    pred_f = np.bincount(pred, minlength=k) / len(y)
    true_b = np.bincount(y, weights=w, minlength=k) / w.sum()
    pred_b = np.bincount(pred, weights=w, minlength=k) / w.sum()
    return {
        "macro_f1": round(macro_f1(y, pred, k), 4),
        "macro_f1_bytes": round(macro_f1(y, pred, k, w), 4),
        f"{focus}_recall": round(float((pred[m] == f).mean()), 4),
        f"{focus}_recall_bytes": round(float(w[m & (pred == f)].sum() / w[m].sum()), 4),
        "class_mix_tv": round(float(0.5 * np.abs(pred_f - true_f).sum()), 4),
        "class_mix_tv_bytes": round(float(0.5 * np.abs(pred_b - true_b).sum()), 4),
        f"{focus}_gap_pts": round(float(100 * (pred_f - true_f)[f]), 2),
        f"{focus}_gap_pts_bytes": round(float(100 * (pred_b - true_b)[f]), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", default="E1")
    ap.add_argument("--focus", default="file_transfer", help="class reported in detail")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs/ensemble"))
    ap.add_argument("--out", default=str(DEFAULT_BIAS_PATH))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from countdown.data import load
    from countdown.ensemble import Ensemble
    from countdown.ensemble.views import PER_FLOW_OVERRIDES
    from countdown.experts import get_expert

    spec = get_expert(args.expert)
    ens = Ensemble.build(args.expert, members=list(spec.deployed_members), combiner="vote",
                         calibrate="temperature", overrides=PER_FLOW_OVERRIDES.get(args.expert, {}))
    ens.fit(cache_dir=Path(args.runs_dir) / args.expert / "preds-s42")
    names = list(ens.data.space.names)
    oof = ens.combiner.predict_proba(ens._blocks("oof"))
    test = ens.test_proba()
    y_tr, y_te = ens.data.y[ens.data.train_rows], ens.data.y[ens.data.test_rows]
    flows = load(spec.dataset).flows
    nbytes = np.array([float(np.asarray(f.sizes).sum()) for f in flows])
    if len(nbytes) != len(ens.data.y):
        raise SystemExit(f"{args.expert}: {len(nbytes)} flows vs {len(ens.data.y)} rows; byte join unsafe")
    w_tr, w_te = nbytes[ens.data.train_rows], nbytes[ens.data.test_rows]

    ok = ~np.isnan(oof).any(axis=1)
    bias = fit_class_bias(oof[ok], y_tr[ok])
    res = {"expert": args.expert, "dataset": spec.dataset, "classes": names,
           "bias": {c: round(float(b), 4) for c, b in zip(names, bias)},
           "fit_on": f"out-of-fold predictions of the train split ({int(ok.sum())} flows)",
           "objective": "macro-F1 (flows), coordinate ascent, grid -3..3 step 0.25",
           "oof": {"before": _report(y_tr[ok], oof[ok], w_tr[ok], names, args.focus),
                   "after": _report(y_tr[ok], apply_bias(oof[ok], bias), w_tr[ok], names, args.focus)},
           "test": {"before": _report(y_te, test, w_te, names, args.focus),
                    "after": _report(y_te, apply_bias(test, bias), w_te, names, args.focus)}}

    out_dir = ROOT / "results/phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"decision_bias_{args.expert}.json").write_text(json.dumps(res, indent=2) + "\n")
    md = [f"# Decision bias, {args.expert} ({spec.dataset})", "",
          f"Fit on {res['fit_on']}; objective {res['objective']}.", "",
          "| class | bias |", "| --- | --- |"] + [f"| {c} | {b:+.2f} |" for c, b in res["bias"].items()]
    md += ["", "| split | metric | before | after |", "| --- | --- | --- | --- |"]
    for split in ("oof", "test"):
        for key, before in res[split]["before"].items():
            md.append(f"| {split} | {key} | {before} | {res[split]['after'][key]} |")
    (out_dir / f"decision_bias_{args.expert}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    if not args.dry_run:
        path = Path(args.out)
        doc = yaml.safe_load(path.read_text()) if path.exists() else None
        doc = doc or {"version": 1, "experts": {}}
        doc["experts"][args.expert] = {"bias": res["bias"], "fit_on": res["fit_on"],
                                       "objective": res["objective"],
                                       "test_macro_f1": [res["test"]["before"]["macro_f1"],
                                                         res["test"]["after"]["macro_f1"]]}
        header = ("# Per-class decision biases (countdown/ensemble/decision.py, Phase 7 task 7.3).\n"
                  "# Written by scripts/fit_decision_bias.py from out-of-fold predictions; the label is\n"
                  "# argmax(log p + bias). Plain numbers, so the hash-pinned model zoo is unchanged.\n")
        path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
