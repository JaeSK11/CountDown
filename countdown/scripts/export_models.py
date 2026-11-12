#!/usr/bin/env python
"""Export the deployed experts from the finished Phase-5 runs into ``models_store/``.

    python scripts/export_models.py                 # E1, E2, E3 from their latest runs
    python scripts/export_models.py --experts E2 --keep 0.95
    python scripts/export_models.py --all-rows --store runs/models-v2   # Phase 10: refit on all

No retraining: each expert is rebuilt from the run's cached member predictions and refit
models, with its ``deploy`` members, the calibrated soft vote, and the E2 -> E1 fallback
threshold at the ``--keep`` operating point (``PHASE-5.md`` section 10.5).

``--all-rows`` *does* train: every member of the deployed configuration is fit on all labelled
rows (5-fold OOF first, for the calibrators, vote weights and threshold), resumable from
``runs/ensemble/<E>/preds-all-s<seed>/``.  That is ``models-v2``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from countdown.pipeline.artifacts import DEFAULT_STORE, export_expert  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", nargs="+", default=["E1", "E2", "E3"])
    ap.add_argument("--runs-dir", default="runs/ensemble")
    ap.add_argument("--keep", type=float, default=0.90)
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    ap.add_argument("--all-rows", action="store_true", help="train on every labelled row (models-v2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for name in args.experts:
        if args.all_rows:
            exp = export_expert(name, keep=args.keep, seed=args.seed, all_rows=True,
                                cache_dir=Path(args.runs_dir) / name / f"preds-all-s{args.seed}")
        else:
            runs = sorted(p for p in (Path(args.runs_dir) / name).glob("2026*")
                          if (p / "ensemble.json").exists()
                          and not json.loads((p / "ensemble.json").read_text()).get("args", {}).get("outer_fold")
                          and json.loads((p / "ensemble.json").read_text()).get("args", {}).get("seed", 42) == args.seed)
            if not runs:
                raise SystemExit(f"no finished Phase-5 run for {name} under {args.runs_dir}/{name}")
            exp = export_expert(name, runs[-1], keep=args.keep, seed=args.seed)
        Path(args.store).mkdir(parents=True, exist_ok=True)
        path = exp.save(Path(args.store) / f"{name}.joblib")
        print(f"{name}: members {list(exp.members)}, thresholds {exp.fallback_threshold}, "
              f"test macro-F1 {exp.provenance['test_macro_f1']}, OOF {exp.provenance['oof_macro_f1']} -> {path}")
        print("   ", json.dumps(exp.provenance, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
