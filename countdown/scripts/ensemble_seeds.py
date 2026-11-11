#!/usr/bin/env python
"""Does an expert's ensemble beat its best single member across seeds, not just on one?

    python scripts/ensemble_seeds.py --expert E3              # -> results/ensemble_seeds/E3.{json,md}
    python scripts/ensemble_seeds.py --expert E2 --by fold    # -> E2-folds.{json,md}

Reads every finished ``runs/ensemble/<expert>/<run_id>/`` (``ensemble.json`` + ``ablation.json``)
written by ``scripts/run_ensemble.py``; the latest run per unit counts.

* ``--by seed`` (default): outer fold 0 runs, one per ``--seed``.  A seed changes the members'
  initialisation and -- where the class mix of groups ties -- the grouped split (see
  ``countdown.eval.splits.group_split``: E2's split does not move with the seed).
* ``--by fold``: seed-42 runs, one per ``--outer-fold``.  Folds are disjoint test sets.

Per seed, test macro-F1 of:

* ``gbdt``      -- ``flow_gbdt`` alone (what E1 ships, and the obvious fallback);
* ``best_single`` -- the best single member **on test** (optimistic for the single side);
* ``deployed``  -- the shipped configuration: soft vote over ``expert.deploy``, temperature-
  calibrated;
* ``selected``  -- the combiner chosen by CV on out-of-fold predictions (never by test).

The summary gives mean, std and range of each, the paired differences (deployed - gbdt,
deployed - best_single, selected - best_single) with a t-interval, win counts and an exact
two-sided sign test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _row(rows: list[dict], members: list[str], combiner: str, calibration: str) -> dict | None:
    for r in rows:
        if (sorted(r["members"]) == sorted(members) and r["combiner"] == combiner
                and r["calibration"] == calibration):
            return r
    return None


def collect(expert: str, runs_dir: Path, deploy: list[str], by: str = "seed",
            seed: int = 42) -> list[dict]:
    by_unit: dict[int, dict] = {}
    for d in sorted((runs_dir / expert).glob("*/ablation.json")):
        run = d.parent
        ens_path = run / "ensemble.json"
        if not ens_path.exists():
            continue
        ens = json.loads(ens_path.read_text())
        args = ens.get("args") or {}
        if args.get("max_flows") or args.get("members") not in (None, []) and \
                sorted(args["members"]) != sorted(deploy):
            continue                       # smoke runs and member subsets are not comparable
        ab = json.loads(d.read_text())
        rows = ab["rows"]
        gbdt = _row(rows, ["flow_gbdt"], "-", "none")
        dep = (_row(rows, deploy, "vote", "temperature") if len(deploy) > 1
               else _row(rows, deploy, "-", "temperature"))
        run_seed, fold = int(args.get("seed", 42)), int(args.get("outer_fold") or 0)
        if by == "seed" and fold != 0 or by == "fold" and run_seed != seed:
            continue
        unit = run_seed if by == "seed" else fold
        by_unit[unit] = {                  # sorted by run id = time: the latest run wins
            "unit": unit, "seed": run_seed, "outer_fold": fold, "run": str(run.relative_to(ROOT)),
            "n_test": ens.get("data", {}).get("n_test"),
            "gbdt": gbdt["macro_f1"] if gbdt else None,
            "best_single": ab["best_single"]["macro_f1"],
            "best_single_member": "+".join(ab["best_single"]["members"]),
            "deployed": dep["macro_f1"] if dep else None,
            "selected": ab["selected"]["macro_f1"],
            "selected_combiner": f"{ab['selected']['combiner']}/{ab['selected']['calibration']}",
        }
    return [by_unit[u] for u in sorted(by_unit)]


def _t975(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:              # pragma: no cover - scipy is in the env
        return 1.96


def _sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a: list[float], b: list[float]) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean = float(d.mean()) if n else 0.0
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = _t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)] if n > 1 else None,
            "wins": wins, "ties": n - wins - losses, "losses": losses,
            "sign_test_p": round(_sign_p(wins, losses), 4)}


def summarise(seeds: list[dict]) -> dict:
    ok = [s for s in seeds if s["gbdt"] is not None and s["deployed"] is not None]
    col = lambda k: [s[k] for s in ok]
    stats = {k: {"mean": round(float(np.mean(col(k))), 4), "sd": round(float(np.std(col(k), ddof=1)), 4)
                 if len(ok) > 1 else 0.0, "min": min(col(k)), "max": max(col(k))}
             for k in ("gbdt", "best_single", "deployed", "selected")} if ok else {}
    return {"n_seeds": len(ok), "n_units": len(ok), "per_seed": seeds, "stats": stats,
            "deployed_minus_gbdt": paired(col("deployed"), col("gbdt")),
            "deployed_minus_best_single": paired(col("deployed"), col("best_single")),
            "selected_minus_best_single": paired(col("selected"), col("best_single"))}


def markdown(expert: str, deploy: list[str], res: dict, by: str = "seed") -> str:
    dep = (f"soft vote over {' + '.join(deploy)}, temperature-calibrated" if len(deploy) > 1
           else f"{deploy[0]} alone, temperature-calibrated")
    if by == "seed":
        head = (f"# {expert}: ensemble vs single members across seeds",
                f"{res['n_units']} seeds on outer fold 0; a seed re-draws the members' initialisation "
                "and, where groups' class mixes tie, the grouped split.")
    else:
        head = (f"# {expert}: ensemble vs single members across outer folds",
                f"{res['n_units']} outer folds at seed {res['per_seed'][0]['seed'] if res['per_seed'] else 42}; "
                "the folds' test sets share no group.")
    md = [head[0], "", head[1] + f" Test macro-F1. Deployed = {dep}. CV-selected = the combiner "
          "chosen on out-of-fold predictions. Best single is picked **on test** (favours the "
          "single side).", "",
          f"| {by if by == 'seed' else 'outer fold'} | n test | flow_gbdt | best single | deployed | CV-selected |",
          "| --- | --- | --- | --- | --- | --- |"]
    for s in res["per_seed"]:
        f = lambda v: "—" if v is None else f"{v:.3f}"
        md.append(f"| {s['unit']} | {s['n_test']} | {f(s['gbdt'])} | {f(s['best_single'])} "
                  f"({s['best_single_member']}) | {f(s['deployed'])} | {f(s['selected'])} "
                  f"({s['selected_combiner']}) |")
    if res["stats"]:
        md += ["", "| | mean | sd | min | max |", "| --- | --- | --- | --- | --- |"]
        for k, label in (("gbdt", "flow_gbdt"), ("best_single", "best single"),
                         ("deployed", "deployed"), ("selected", "CV-selected")):
            s = res["stats"][k]
            md.append(f"| {label} | {s['mean']:.3f} | {s['sd']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")
    md += ["", "| paired difference | mean | 95 % CI | wins / ties / losses | sign test p |",
           "| --- | --- | --- | --- | --- |"]
    for k, label in (("deployed_minus_gbdt", "deployed − flow_gbdt"),
                     ("deployed_minus_best_single", "deployed − best single"),
                     ("selected_minus_best_single", "CV-selected − best single")):
        p = res[k]
        ci = "—" if not p["ci95"] else f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
        md.append(f"| {label} | {p['mean']:+.3f} | {ci} | {p['wins']} / {p['ties']} / {p['losses']} "
                  f"| {p['sign_test_p']:.3f} |")
    return "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", default="E3", choices=("E1", "E2", "E3"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs/ensemble"))
    ap.add_argument("--out", default=str(ROOT / "results/ensemble_seeds"))
    ap.add_argument("--by", choices=("seed", "fold"), default="seed")
    ap.add_argument("--seed", type=int, default=42, help="the seed of the --by fold runs")
    args = ap.parse_args()

    from countdown.experts import get_expert

    deploy = list(get_expert(args.expert).deployed_members)
    res = summarise(collect(args.expert, Path(args.runs_dir), deploy, by=args.by, seed=args.seed))
    res.update(expert=args.expert, deploy=deploy, by=args.by)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.expert + ("-folds" if args.by == "fold" else "")
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2) + "\n")
    md = markdown(args.expert, deploy, res, by=args.by)
    (out / f"{stem}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
