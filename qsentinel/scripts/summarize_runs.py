#!/usr/bin/env python3
"""Tabulate the runs in ``runs/`` -- the Phase-3 results table, regenerated from artifacts.

    scripts/summarize_runs.py                 # all runs, newest last
    scripts/summarize_runs.py --per-class     # add the per-class F1 breakdown
    scripts/summarize_runs.py --markdown      # paste-ready table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS = Path(__file__).resolve().parent.parent / "runs"


def _load(run_dir: Path) -> dict | None:
    mf = run_dir / "metrics.json"
    if not mf.exists():
        return None
    blob = json.loads(mf.read_text())
    metrics, split = blob.get("metrics", {}), blob.get("split", {})
    cfg = {}
    cf = run_dir / "config.yaml"
    if cf.exists():
        import yaml

        cfg = yaml.safe_load(cf.read_text()) or {}
    tail = metrics.get("tail_excluded") or {}
    return {
        "run": run_dir.name,
        "dataset": cfg.get("dataset_resolved", cfg.get("dataset", "?")),
        "target": metrics.get("target", "?"),
        "model": cfg.get("model", "?"),
        "classes": metrics.get("n_classes"),
        "n_test": metrics.get("n"),
        "macro_f1": metrics.get("macro_f1"),
        "baseline": metrics.get("baseline_macro_f1"),
        "accuracy": metrics.get("accuracy"),
        "weighted_f1": metrics.get("weighted_f1"),
        "tail_macro_f1": tail.get("macro_f1"),
        "tail_classes": tail.get("n_classes"),
        "group_key": split.get("group_key", "?"),
        "groups": f"{split.get('n_groups_train', '?')}/{split.get('n_groups_test', '?')}",
        "disjoint": split.get("groups_disjoint"),
        "absent": len(split.get("classes_absent_from_test") or []),
        "beats": metrics.get("beats_baseline"),
        "per_class": metrics.get("per_class", {}),
    }


def _fmt(v, nd=4):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    rows = [r for r in (_load(d) for d in sorted(args.runs_dir.glob("*/"))) if r]
    if not rows:
        print(f"no runs in {args.runs_dir}")
        return 1

    cols = [("dataset", 12), ("target", 13), ("classes", 7), ("n_test", 7),
            ("macro_f1", 9), ("baseline", 9), ("accuracy", 9), ("group_key", 12),
            ("groups", 9), ("absent", 6)]

    if args.markdown:
        print("| " + " | ".join(c for c, _ in cols) + " | beats |")
        print("|" + "|".join("---" for _ in range(len(cols) + 1)) + "|")
        for r in rows:
            print("| " + " | ".join(_fmt(r[c]) for c, _ in cols)
                  + f" | {'yes' if r['beats'] else 'NO'} |")
    else:
        print("  ".join(c.ljust(w) for c, w in cols))
        print("-" * (sum(w + 2 for _, w in cols)))
        for r in rows:
            print("  ".join(_fmt(r[c]).ljust(w) for c, w in cols))

    for r in rows:
        if not r["disjoint"]:
            print(f"\n!! {r['run']}: split is NOT group-disjoint")
        if r["absent"]:
            print(f"\n!! {r['run']}: {r['absent']} class(es) absent from the test fold")
        if not r["beats"]:
            print(f"\n!! {r['run']}: does NOT beat the majority baseline")

    if args.per_class:
        for r in rows:
            print(f"\n{r['dataset']} / {r['target']}  (macro-F1 {_fmt(r['macro_f1'])})")
            ranked = sorted(r["per_class"].items(), key=lambda kv: -kv[1]["f1"])
            for name, d in ranked[:25]:
                print(f"   {name:24s} F1={d['f1']:.3f}  P={d['precision']:.3f}  "
                      f"R={d['recall']:.3f}  n={d['support']}")
            if len(ranked) > 25:
                print(f"   ... {len(ranked) - 25} more classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
