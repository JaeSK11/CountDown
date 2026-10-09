"""Phase-5 ablation: single members vs soft vote vs stacking, each with and without calibration.

Everything is computed from one set of member predictions (out-of-fold over train, refit on
train for test), so every row of the table sees identical members and identical rows -- the
only thing that varies is how the probabilities are combined.  That is what makes "the
ensemble beats the best single member" a statement about combining, not about luck in
re-training.

Calibration is scored on test (ECE, NLL); it is fit on the OOF predictions only.

Choosing the winner by test score from ~30 variants would be selection on test.  So the
verdict uses two numbers that never look at test labels to choose:

``primary``   the configuration declared up front: every member, ``stack_logreg``,
              temperature calibration;
``selected``  the variant with the best **cross-validated OOF** macro-F1 -- the combiner is
              fit on K-1 OOF folds and scored on the held one (``fold`` ids from
              ``stacking.oof_predict``), so the choice is made inside train.

``best_ensemble`` (best by test) is still reported, labelled as the optimistic ceiling.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from countdown.ensemble.calibration import ProbaCalibrator, ece, nll
from countdown.ensemble.combiner import make_combiner
from countdown.eval.metrics import compute_metrics

COMBINERS = ("vote", "stack_logreg", "stack_lgbm")
CALIBRATIONS = ("none", "temperature")


def _score(proba: np.ndarray, y: np.ndarray, n: int) -> dict[str, float]:
    m = compute_metrics(y, proba.argmax(axis=1), n)
    return {"macro_f1": round(m["macro_f1"], 4), "accuracy": round(m["accuracy"], 4),
            "ece": round(ece(proba, y), 4), "nll": round(nll(proba, y), 4)}


def _calibrated(oof: dict[str, np.ndarray], test: dict[str, np.ndarray], y_oof: np.ndarray,
                method: str) -> tuple[dict, dict]:
    co, ct = {}, {}
    for m in oof:
        ok = ~np.isnan(oof[m]).any(axis=1)
        cal = ProbaCalibrator(method).fit(oof[m][ok], y_oof[ok])
        for src, dst in ((oof, co), (test, ct)):
            p = src[m]
            good = ~np.isnan(p).any(axis=1)
            q = np.full_like(p, np.nan)
            q[good] = cal.transform(p[good])
            dst[m] = q
    return co, ct


PRIMARY = ("stack_logreg", "temperature")


def _cv_score(comb: str, blocks: list[np.ndarray], y: np.ndarray, fold: np.ndarray,
              n_classes: int, seed: int) -> float:
    """Macro-F1 of ``comb`` fit on K-1 OOF folds, predicting the held fold, pooled."""
    pred = np.zeros((len(y), n_classes))
    for k in np.unique(fold):
        a, b = fold != k, fold == k
        model = make_combiner(comb, seed=seed).fit([x[a] for x in blocks], y[a])
        pred[b] = model.predict_proba([x[b] for x in blocks])
    return round(compute_metrics(y, pred.argmax(1), n_classes)["macro_f1"], 4)


def run_ablation(oof: dict[str, np.ndarray], test: dict[str, np.ndarray], y_oof: np.ndarray,
                 y_test: np.ndarray, n_classes: int, combiners: tuple[str, ...] = COMBINERS,
                 calibrations: tuple[str, ...] = CALIBRATIONS, subsets: bool = True,
                 seed: int = 42, fold: np.ndarray | None = None,
                 cv_combiners: tuple[str, ...] | None = None) -> dict[str, Any]:
    """``{rows, best_single, primary, selected, best_ensemble, ensemble_beats_best_single}``.

    ``fold`` (the OOF fold of each train row) enables ``cv_macro_f1`` and ``selected``;
    ``cv_combiners`` limits which combiners are cross-validated (default: all).
    """
    members = list(oof)
    cal = {c: _calibrated(oof, test, y_oof, c) for c in calibrations}
    cv_combiners = combiners if cv_combiners is None else cv_combiners
    rows: list[dict[str, Any]] = []
    for m in members:
        for c in calibrations:
            p = np.nan_to_num(cal[c][1][m], nan=1.0 / n_classes)
            po = np.nan_to_num(cal[c][0][m], nan=1.0 / n_classes)
            rows.append({"kind": "single", "members": [m], "combiner": "-",
                         "calibration": c, **_score(p, y_test, n_classes),
                         "cv_macro_f1": round(compute_metrics(y_oof, po.argmax(1),
                                                              n_classes)["macro_f1"], 4)})

    sets = [tuple(members)]
    if subsets and len(members) > 2:
        sets += [s for k in range(2, len(members)) for s in combinations(members, k)]
    if len(members) >= 2:
        for s in sets:
            for comb in combiners:
                for c in calibrations:
                    co, ct = cal[c]
                    model = make_combiner(comb, seed=seed).fit([co[m] for m in s], y_oof)
                    p = model.predict_proba([ct[m] for m in s])
                    cv = (_cv_score(comb, [co[m] for m in s], y_oof, fold, n_classes, seed)
                          if fold is not None and comb in cv_combiners else None)
                    rows.append({"kind": "ensemble", "members": list(s), "combiner": comb,
                                 "calibration": c, **_score(p, y_test, n_classes),
                                 "cv_macro_f1": cv})

    singles = [r for r in rows if r["kind"] == "single" and r["calibration"] == "none"]
    best_single = max(singles, key=lambda r: r["macro_f1"])
    ens = [r for r in rows if r["kind"] == "ensemble"]
    best_ens = max(ens, key=lambda r: r["macro_f1"]) if ens else None
    primary = next((r for r in ens if len(r["members"]) == len(members)
                    and (r["combiner"], r["calibration"]) == PRIMARY), None)
    scored = [r for r in rows if r.get("cv_macro_f1") is not None]
    selected = max(scored, key=lambda r: r["cv_macro_f1"]) if fold is not None and scored else None

    def beats(r: dict | None) -> bool | None:
        if r is None or r["kind"] != "ensemble":
            return None
        return bool(r["macro_f1"] >= best_single["macro_f1"])

    verdicts = [v for v in (beats(primary), beats(selected)) if v is not None]
    return {
        "rows": rows, "best_single": best_single, "primary": primary, "selected": selected,
        "best_ensemble": best_ens,
        "primary_beats_best_single": beats(primary),
        "selected_beats_best_single": beats(selected),
        # The exit criterion, judged only on configurations chosen without test labels.
        "ensemble_beats_best_single": any(verdicts) if verdicts else None,
    }


def to_markdown(result: dict[str, Any]) -> str:
    head = ("| kind | members | combiner | calibration | CV macro-F1 (OOF) | macro-F1 | accuracy "
            "| ECE | NLL |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    tag = {id(result.get("primary")): " **(primary)**", id(result.get("selected")): " **(selected)**"}
    lines = []
    for r in sorted(result["rows"], key=lambda r: (r["kind"] != "single", -r["macro_f1"])):
        cv = "—" if r.get("cv_macro_f1") is None else f"{r['cv_macro_f1']:.4f}"
        lines.append(f"| {r['kind']}{tag.get(id(r), '')} | {' + '.join(r['members'])} | "
                     f"{r['combiner']} | {r['calibration']} | {cv} | {r['macro_f1']:.4f} | "
                     f"{r['accuracy']:.4f} | {r['ece']:.4f} | {r['nll']:.4f} |")
    return head + "\n".join(lines) + "\n"
