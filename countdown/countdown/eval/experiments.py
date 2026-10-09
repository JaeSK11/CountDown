"""``countdown experiments run`` -- regenerate every results table (Phase 6, tasks 6.5-6.6).

Each experiment writes ``results/experiments/<name>.json`` and ``.md`` with **no timestamps or
run-specific paths**, so re-running on unchanged inputs reproduces the files byte for byte
(``tests/test_experiments.py`` checks it on the cheap ones).

==================  =======================================================================
``triage``          PQC-triage per dataset + synthetic PQC + predicted-breakdown fidelity
``cross_dataset``   train on one capture setting, test on another (traffic type, GBDT)
``ablation``        the Phase-5 ensemble ablation and per-target headline, tabulated from the
                    finished Phase-5 runs (``runs/ensemble/``) -- it re-reads their saved
                    predictions' scores; it does not re-train hours of GPU members
==================  =======================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "experiments"


def _triage() -> tuple[dict, str]:
    from countdown.eval.triage import build_report

    return build_report()


def _cross() -> tuple[dict, str]:
    from countdown.eval import cross_dataset

    res = cross_dataset.run()
    return res, cross_dataset.to_markdown(res)


def _ablation() -> tuple[dict, str]:
    from countdown.experts import get_expert

    out: dict[str, Any] = {}
    md = ["# Phase-5 ensemble ablation (per target)", "",
          "Macro-F1 on each expert's grouped test split, seed 42. `selected` = combiner chosen by "
          "cross-validation on out-of-fold predictions (never by test).", "",
          "| expert | target | best single | selected ensemble | deployed members | verdict |",
          "| --- | --- | --- | --- | --- | --- |"]
    for name in ("E1", "E2", "E3"):
        runs = sorted(p for p in (ROOT / "runs/ensemble" / name).glob("2026*")
                      if (p / "ablation.json").exists())
        if not runs:
            continue
        r = json.loads((runs[-1] / "ablation.json").read_text())
        spec = get_expert(name)
        bs, sel = r["best_single"], r["selected"]
        out[name] = {"target": spec.target, "dataset": spec.dataset,
                     "best_single": {k: bs[k] for k in ("members", "macro_f1", "accuracy")},
                     "selected": {k: sel[k] for k in ("members", "combiner", "calibration",
                                                      "macro_f1", "accuracy", "cv_macro_f1")},
                     "deployed": list(spec.deployed_members),
                     "beats_best_single": r["ensemble_beats_best_single"],
                     "rows": r["rows"]}
        md.append(f"| {name} | {spec.target} | {bs['members'][0]} {bs['macro_f1']:.4f} | "
                  f"{' + '.join(sel['members'])} ({sel['combiner']}, {sel['calibration']}) "
                  f"{sel['macro_f1']:.4f} | {' + '.join(spec.deployed_members)} | "
                  f"{'✓' if r['ensemble_beats_best_single'] else '✗'} |")
    return out, "\n".join(md) + "\n"


EXPERIMENTS: dict[str, Callable[[], tuple[dict, str]]] = {
    "triage": _triage, "cross_dataset": _cross, "ablation": _ablation,
}


def run(names: list[str] | None = None, out_dir: Path = OUT) -> dict[str, Path]:
    names = names or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise KeyError(f"unknown experiment(s) {unknown}; available: {list(EXPERIMENTS)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for n in names:
        res, md = EXPERIMENTS[n]()
        (out_dir / f"{n}.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")
        (out_dir / f"{n}.md").write_text(md)
        written[n] = out_dir / f"{n}.md"
    return written
