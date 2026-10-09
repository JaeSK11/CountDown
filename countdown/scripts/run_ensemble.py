#!/usr/bin/env python
"""Phase 5: fit one expert's ensemble and write the ablation table.

    python -m scripts.run_ensemble --expert E1
    python scripts/run_ensemble.py --expert E2 --members flow_gbdt graph_gnn
    python scripts/run_ensemble.py --expert E1 --max-flows 3000 --n-folds 3   # smoke run

Per member: K-fold out-of-fold predictions over the outer train rows, then a refit on all of
train that scores test.  Predictions and the refit model are cached under
``runs/ensemble/<expert>/preds-s<seed>/`` and reused when the split matches, so a re-run
(or a run that lost the GPU half-way) only pays for the members it has not finished.

Output: ``runs/ensemble/<expert>/<run_id>/`` with ``ablation.json``, ``ablation.md``,
``ensemble.json`` (members, calibration, data audit), ``routing.json`` (where the router
sends this dataset's flows) and ``ensemble.joblib`` (the fitted stacking ensemble).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from countdown.ensemble import Ensemble, Router, build_expert_data  # noqa: E402
from countdown.eval.ablation import run_ablation, to_markdown  # noqa: E402
from countdown.eval.report import run_id  # noqa: E402
from countdown.experts import get_expert  # noqa: E402

#: Per-flow input overrides.  The image member's 224-px default is ~600 KB per window; per
#: flow on ISCX that is >100 GB, and a FlowPic of a 9-packet flow carries no detail a
#: 64-px histogram loses (64 is the smallest the network's pooling stack accepts).  One
#: 60 s window: the median ISCX flow is far shorter than 15 s, so 15/30/60 s windows would
#: be three copies of one image.  Larger batches / fewer epochs because there are ~10^5
#: images here, not the ~10^3 the scorecard trained on.  E3 keeps the scorecard's
#: construction (MobileApp is one flow per capture, so per-flow = per-capture windows).
OVERRIDES = {
    "E1": {"flow_image_cnn": {"feature_params": {"size": 64}, "windows": [60.0],
                              "params": {"batch_size": 256, "epochs": 15}}},
    "E2": {"graph_gnn": {"params": {"epochs": 20, "workers": 24}}},
    "E3": {},
}
#: Combiners scored by cross-validation on OOF.  A 120-class LightGBM meta-learner is
#: minutes per fit; five more per variant is not worth it on E2.
CV_COMBINERS = {"E1": ("vote", "stack_logreg", "stack_lgbm"), "E2": ("vote", "stack_logreg"),
                "E3": ("vote", "stack_logreg", "stack_lgbm")}
EXPECTED_ROUTE = {"E1": "E1", "E2": "E2", "E3": "E3"}
CONTEXT = {"E3": "mobile"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", choices=("E1", "E2", "E3"), required=True)
    ap.add_argument("--members", nargs="*", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-flows", type=int, default=None, help="subsample (smoke runs)")
    ap.add_argument("--runs-dir", default="runs/ensemble")
    ap.add_argument("--no-subsets", action="store_true", help="ablate the full member set only")
    args = ap.parse_args()

    t0 = time.time()
    expert = get_expert(args.expert)
    data = build_expert_data(expert, members=args.members, seed=args.seed,
                             overrides=OVERRIDES[args.expert], max_flows=args.max_flows)
    print(json.dumps(data.describe(), indent=2, default=str))

    root = Path(args.runs_dir) / args.expert
    tag = f"preds-s{args.seed}" + (f"-n{args.max_flows}" if args.max_flows else "")
    ens = Ensemble(data, combiner="stack_logreg", calibrate="temperature",
                   n_folds=args.n_folds, seed=args.seed).fit(cache_dir=root / tag)

    tr, te = data.train_rows, data.test_rows
    res = run_ablation({m: ens.preds[m].oof for m in ens.members},
                       {m: ens.preds[m].test for m in ens.members},
                       data.y[tr], data.y[te], len(data.space), subsets=not args.no_subsets,
                       seed=args.seed, fold=ens.preds[ens.members[0]].fold,
                       cv_combiners=CV_COMBINERS[args.expert])

    from countdown.data import load

    flows = load(expert.dataset).flows
    routing = Router().report(flows, expected=EXPECTED_ROUTE[args.expert],
                              context=CONTEXT.get(args.expert))

    out = root / run_id()
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(res, indent=2))
    (out / "ablation.md").write_text(to_markdown(res))
    desc = ens.describe()
    desc["member_seconds"] = {m: round(ens.preds[m].seconds, 1) for m in ens.members}
    desc["args"] = vars(args)
    desc["wall_seconds"] = round(time.time() - t0, 1)
    (out / "ensemble.json").write_text(json.dumps(desc, indent=2, default=str))
    (out / "routing.json").write_text(json.dumps(routing, indent=2))
    ens.save(out / "ensemble.joblib")

    print(to_markdown(res))
    bs = res["best_single"]
    print(f"best single (by test)  {bs['members'][0]:<34} macro-F1 {bs['macro_f1']:.4f}")
    for label in ("primary", "selected", "best_ensemble"):
        r = res[label]
        if r:
            print(f"{label:<22} {' + '.join(r['members'])} / {r['combiner']} / {r['calibration']}"
                  f"  macro-F1 {r['macro_f1']:.4f}  (CV {r.get('cv_macro_f1')})")
    print(f"exit criterion (primary or CV-selected ensemble >= best single): "
          f"{res['ensemble_beats_best_single']}")
    print(f"routing: {routing.get('share_expected', float('nan')):.1%} of {expert.dataset} flows "
          f"-> {EXPECTED_ROUTE[args.expert]}")
    print(f"run {out}  ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
