"""Cross-dataset generalisation (Phase 6, task 6.6): train on one capture setting, test on another.

Every Phase-3/4/5 number is in-domain: train and test captures come from the same corpus,
recorded the same way.  A deployed sensor sees neither.  Two kinds of shift are measurable with
the shared ``traffic_type`` taxonomy:

``corpus``  ISCX-VPN <-> ISCX-Tor -- two capture campaigns, different hosts and sessions.
``tunnel``  plain -> tunnelled within one corpus (non-VPN -> VPN, non-Tor -> Tor) and back:
            does a model that learnt "file transfer" on the open wire recognise it inside a
            tunnel, which is the only form E1 sees in deployment for tunnelled traffic?

Model: ``flow_gbdt`` on ``flow_stats`` -- E1's deployed member -- 600 rounds without early
stopping (see ``_fit``), no class weights (D5), seed 42.  Scores are
macro-F1 over the classes **both** sides contain (a class the training side never saw cannot
be predicted; one the test side lacks cannot be scored), next to the in-domain grouped score
of the same training side on the same class set.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from countdown.eval.metrics import compute_metrics
from countdown.eval.splits import group_split
from countdown.schema import LabelSpace

TARGET = "traffic_type"


def _subset(name: str, tunnel: str | None) -> Callable[[], list]:
    def get() -> list:
        from countdown.data import load

        flows = load(name).flows
        if tunnel is None:
            return flows
        return [f for f in flows if f.label_fields.get("tunnel_type") == tunnel]
    return get


#: (experiment id, shift kind, train side, test side) -- sides are (label, loader).
PAIRS: list[tuple[str, str, tuple[str, Callable], tuple[str, Callable]]] = [
    ("vpn->tor", "corpus", ("ISCX-VPN", _subset("iscxvpn", None)), ("ISCX-Tor", _subset("iscxtor", None))),
    ("tor->vpn", "corpus", ("ISCX-Tor", _subset("iscxtor", None)), ("ISCX-VPN", _subset("iscxvpn", None))),
    ("nonvpn->vpn", "tunnel", ("ISCX-VPN plain", _subset("iscxvpn", "none")), ("ISCX-VPN tunnelled", _subset("iscxvpn", "vpn"))),
    ("vpn->nonvpn", "tunnel", ("ISCX-VPN tunnelled", _subset("iscxvpn", "vpn")), ("ISCX-VPN plain", _subset("iscxvpn", "none"))),
    ("nontor->tor", "tunnel", ("ISCX-Tor plain", _subset("iscxtor", "none")), ("ISCX-Tor tunnelled", _subset("iscxtor", "tor"))),
    ("tor->nontor", "tunnel", ("ISCX-Tor tunnelled", _subset("iscxtor", "tor")), ("ISCX-Tor plain", _subset("iscxtor", "none"))),
]


def _xy(flows: list, space: LabelSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from countdown.config import default_config
    from countdown.schema import FlowDataset

    ds = FlowDataset(flows, name="x", default_target=TARGET, config=default_config())
    return (ds.features("flow_stats"), ds.labels(TARGET, space=space),
            np.asarray(ds.groups("source_file"), dtype=object))


def _fit(X: np.ndarray, y: np.ndarray, g: np.ndarray, space: LabelSpace, seed: int):
    """Fixed 600 rounds, no early stopping -- the same rule on every side.

    Early stopping needs a grouped validation carve-out, and on the small sides (31 tunnelled
    VPN captures, 51 Tor ones) that is a few unrepresentative captures: on ISCX-VPN it stopped
    at round 1 and left a near-constant classifier (in-domain macro-F1 0.045 vs the harness's
    0.40).  A fixed budget keeps in-domain and cross comparable.
    """
    from countdown.models.base import ModelRegistry

    m = ModelRegistry.create("flow_gbdt", label_space=space, seed=seed)
    m.fit(X, y)
    return m


def _score(model, X: np.ndarray, y: np.ndarray, classes: np.ndarray, space: LabelSpace) -> dict[str, Any]:
    keep = np.isin(y, classes)
    pred = model.predict(X[keep])
    m = compute_metrics(y[keep], pred, len(space), space.names)
    per = {space.names[c]: round(m["per_class"][space.names[c]]["f1"], 4) for c in classes}
    # macro over the shared classes only (compute_metrics' macro_f1 is over classes present
    # in y_true, which is exactly `classes` here)
    return {"n": int(keep.sum()), "macro_f1": round(m["macro_f1"], 4),
            "accuracy": round(m["accuracy"], 4), "per_class_f1": per}


def run_pair(pid: str, kind: str, train: tuple, test: tuple, seed: int = 42) -> dict[str, Any]:
    space = LabelSpace.taxonomy(TARGET)
    Xa, ya, ga = _xy(train[1](), space)
    Xb, yb, gb = _xy(test[1](), space)
    shared = np.intersect1d(np.unique(ya), np.unique(yb))
    model = _fit(Xa, ya, ga, space, seed)
    cross = _score(model, Xb, yb, shared, space)

    # In-domain reference: the training side on its own grouped split, same class set.
    in_dom: dict[str, Any] | None
    try:
        tr, te = group_split(None, ya, ga, test_size=0.2, seed=seed)
        ref = _fit(Xa[tr], ya[tr], ga[tr], space, seed)
        in_dom = _score(ref, Xa[te], ya[te], np.intersect1d(shared, np.unique(ya[te])), space)
    except ValueError as exc:            # too few captures for a grouped split
        in_dom = {"error": str(exc)}
    drop = (round(in_dom["macro_f1"] - cross["macro_f1"], 4)
            if in_dom and "macro_f1" in in_dom else None)
    return {
        "id": pid, "shift": kind, "train": train[0], "test": test[0],
        "n_train": int(len(ya)), "train_captures": len(set(ga.tolist())),
        "n_test": int(len(yb)), "test_captures": len(set(gb.tolist())),
        "shared_classes": [space.names[c] for c in shared],
        "excluded_train_only": sorted(set(space.names[c] for c in np.unique(ya)) - set(space.names[c] for c in shared)),
        "excluded_test_only": sorted(set(space.names[c] for c in np.unique(yb)) - set(space.names[c] for c in shared)),
        "in_domain": in_dom, "cross": cross, "drop": drop,
    }


def run(seed: int = 42) -> dict[str, Any]:
    return {"model": "flow_gbdt on flow_stats (E1's deployed member)", "target": TARGET,
            "seed": seed, "pairs": [run_pair(*p, seed=seed) for p in PAIRS]}


def to_markdown(res: dict[str, Any]) -> str:
    lines = ["# Cross-dataset generalisation (traffic_type, flow_gbdt)", "",
             "Macro-F1 over the classes both sides contain; in-domain = the training side's own "
             "grouped 80/20 split on the same classes.", "",
             "| shift | train → test | shared classes | in-domain | cross | drop | test flows (captures) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for p in res["pairs"]:
        ind = p["in_domain"].get("macro_f1", "n/a") if p["in_domain"] else "n/a"
        lines.append(f"| {p['shift']} | {p['train']} → {p['test']} | {len(p['shared_classes'])} | "
                     f"{ind} | **{p['cross']['macro_f1']}** | {p['drop'] if p['drop'] is not None else '—'} | "
                     f"{p['cross']['n']:,} ({p['test_captures']}) |")
    lines += ["", "Per-class F1, cross:", "",
              "| train → test | " + " | ".join(LabelSpace.taxonomy(TARGET).names) + " |",
              "| --- |" + " --- |" * len(LabelSpace.taxonomy(TARGET).names)]
    for p in res["pairs"]:
        pc = p["cross"]["per_class_f1"]
        lines.append(f"| {p['id']} | " + " | ".join(
            f"{pc[c]:.2f}" if c in pc else "·" for c in LabelSpace.taxonomy(TARGET).names) + " |")
    return "\n".join(lines) + "\n"
