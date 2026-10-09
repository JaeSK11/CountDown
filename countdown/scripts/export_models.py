#!/usr/bin/env python
"""Export the deployed experts from the finished Phase-5 runs into ``models_store/``.

    python scripts/export_models.py                 # E1, E2, E3 from their latest runs
    python scripts/export_models.py --experts E2 --keep 0.95

No retraining: each expert is rebuilt from the run's cached member predictions and refit
models, with its ``deploy`` members, the calibrated soft vote, and the E2 -> E1 fallback
threshold at the ``--keep`` operating point (``PHASE-5.md`` section 10.5).
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
    args = ap.parse_args()
    for name in args.experts:
        runs = sorted(p for p in (Path(args.runs_dir) / name).glob("2026*") if (p / "ensemble.json").exists())
        if not runs:
            raise SystemExit(f"no finished Phase-5 run for {name} under {args.runs_dir}/{name}")
        exp = export_expert(name, runs[-1], keep=args.keep)
        path = exp.save(Path(args.store) / f"{name}.joblib")
        print(f"{name}: members {list(exp.members)}, thresholds {exp.fallback_threshold}, "
              f"test macro-F1 {exp.provenance['test_macro_f1']} -> {path}")
        print("   ", json.dumps(exp.provenance, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
