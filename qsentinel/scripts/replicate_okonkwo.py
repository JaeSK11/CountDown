#!/usr/bin/env python
"""Replicate Okonkwo et al., AISC 2022, and check the published numbers.

    python scripts/replicate_okonkwo.py --tasks all
    python scripts/replicate_okonkwo.py --tasks 7 --windows 60 --epochs 20   # quick look

What is reproduced
------------------
* the paper's **representation** -- a size-vs-time scatter image, sizes capped at the MTU
  with oversize packets dropped, cut into 15/30/60 s windows, **each window scaling the
  x-axis to its own length** per section 4.2 (``flow_image`` extractor,
  ``construction="scatter"``, ``window="auto"``),
* the paper's **windowing unit** -- one image per *capture*-window, not per flow.  Okonkwo
  never splits by 5-tuple; Table 2's 1x/2x/4x proportionality at 60/30/15 s only holds if
  fixed windows tile one pcap timeline (``data/windows.py``),
* the paper's **model** -- the eleven-layer CNN of Figure 2, whose 576-wide flatten this
  implementation reproduces exactly (``models/flow_image_cnn.py``),
* the paper's **training setup** -- batch 10, 60 epochs, Adam 1e-4, categorical CE,
* the paper's **seven identification tasks** and their exact class lists, read off the
  confusion matrices in Figure 5.

Protocol
--------
Every task runs under **both** protocols, for the same reason as
``replicate_drapergil.py``:

``paper``
    Random 80/10/10 over the pooled window images, which is what section 4.3 describes.
    A 60 s window contains the same packets as four 15 s windows, so near-duplicates land
    on both sides of the split and the score is partly memorisation.
``grouped``
    Split by ``source_file``, so every window cut from one capture stays on one side.

The gap between the two columns is the finding, not a bug in either.

Known fidelity gaps, reported rather than hidden
------------------------------------------------
* **Marker size.**  Okonkwo rendered matplotlib scatter plots to JPEG, so each packet was
  an anti-aliased blob a few pixels wide.  We rasterise directly with a 3x3 marker
  (``--marker``); 1 would be a materially sparser image than their model saw.
* **Window counts.**  Ours match Table 2 within 2-9% on ISCXVPN (2635/5101/9772 vs
  2693/5386/10772) but run ~1.5x high on ISCXTor (3920/7691/14852 vs 2482/4964/9928),
  which suggests the paper used a subset of the Tor captures it does not enumerate.
* **Class lists.**  The paper drops classes with <150 images after augmentation and
  excludes ``browsing`` from the non-VPN and VPN tasks as poorly defined.  We restrict to
  its stated classes instead of re-deriving that floor, and report what each split saw.
* **Augmentation.**  The paper augments with rotation and flipping.  On a size-vs-time
  plot that reverses causality and maps packet size onto the time axis, so it is *not*
  reproduced; ``--augment`` exists to measure it, and is off by default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qsentinel.config import default_config, get_logger
from qsentinel.data import load
from qsentinel.data.windows import expand_to_windows
from qsentinel.eval.metrics import compute_metrics
from qsentinel.eval.splits import group_split_three
from qsentinel.features import get_extractor
from qsentinel.models import ModelRegistry
from qsentinel.schema import LabelSpace

log = get_logger("replicate-okonkwo")

#: Table 4 of the paper: accuracy / precision / recall / F1 per task.
PUBLISHED: dict[str, dict[str, float]] = {
    "1-nonvpn-app":      {"accuracy": 0.93, "precision": 0.97, "recall": 0.96, "f1": 0.96},
    "2-nonvpn-traffic":  {"accuracy": 0.91, "precision": 0.91, "recall": 0.91, "f1": 0.91},
    "3-vpn-app":         {"accuracy": 0.96, "precision": 0.96, "recall": 0.96, "f1": 0.96},
    "4-vpn-traffic":     {"accuracy": 0.96, "precision": 0.96, "recall": 0.96, "f1": 0.96},
    "5-tor-app":         {"accuracy": 0.91, "precision": 0.92, "recall": 0.91, "f1": 0.91},
    "6-tor-traffic":     {"accuracy": 0.91, "precision": 0.93, "recall": 0.91, "f1": 0.91},
    "7-encryption":      {"accuracy": 0.99, "precision": 0.99, "recall": 0.99, "f1": 0.99},
}

#: The seven tasks.  ``classes`` are read off the Figure 5 confusion matrices; a task is
#: restricted to exactly those, so "how many of the paper's classes does our label
#: normalisation actually recover" is answered explicitly per run.
TASKS: dict[str, dict] = {
    "1-nonvpn-app": dict(
        datasets=["iscxvpn"], target="app_activity",
        where=lambda lf: not lf["is_vpn"],
        classes=["facebook_audio", "facebook_video", "hangouts_audio", "hangouts_video",
                 "netflix_video", "skype_audio", "skype_video", "vimeo_video",
                 "voipbuster_audio", "youtube_video"],
    ),
    "2-nonvpn-traffic": dict(
        datasets=["iscxvpn"], target="traffic_type",
        where=lambda lf: not lf["is_vpn"],
        classes=["chat", "file_transfer", "video_streaming", "voip"],
    ),
    "3-vpn-app": dict(
        datasets=["iscxvpn"], target="app_norm",
        where=lambda lf: lf["is_vpn"],
        classes=["email", "hangouts", "skype", "spotify", "voipbuster", "youtube"],
    ),
    "4-vpn-traffic": dict(
        datasets=["iscxvpn"], target="traffic_type",
        where=lambda lf: lf["is_vpn"],
        classes=["chat", "file_transfer", "video_streaming", "voip"],
    ),
    "5-tor-app": dict(
        datasets=["iscxtor"], target="app_norm",
        where=lambda lf: lf["is_tor"],
        classes=["facebook", "skype", "spotify", "youtube"],
    ),
    "6-tor-traffic": dict(
        datasets=["iscxtor"], target="traffic_type",
        where=lambda lf: lf["is_tor"],
        classes=["audio_streaming", "browsing", "chat", "file_transfer", "p2p",
                 "video_streaming", "voip"],
    ),
    "7-encryption": dict(
        datasets=["iscxvpn", "iscxtor"], target="tunnel_type",
        where=lambda lf: True,
        classes=["none", "vpn", "tor"],
    ),
}


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def build_task(name: str, spec: dict, windows, cache: dict, args) -> tuple:
    """``(X, y, groups, space, report)`` for one task."""
    flows = []
    for dsname in spec["datasets"]:
        ds = cache.get(dsname) or cache.setdefault(dsname, load(dsname))
        flows.extend(ds.flows)

    keep = [f for f in flows
            if spec["where"](f.label_fields)
            and str(f.label_fields.get(spec["target"])) in set(spec["classes"])]
    present = sorted({str(f.label_fields[spec["target"]]) for f in keep})
    missing = [c for c in spec["classes"] if c not in present]
    if not keep:
        raise ValueError(f"task {name}: no flows matched")

    win = []
    for w in windows:
        win.extend(expand_to_windows(keep, window=float(w)))

    extractor = get_extractor(
        "flow_image", size=args.size, construction=args.construction,
        window="auto", channels=args.channels, marker=args.marker,
    )
    t0 = time.time()
    X = extractor.transform(win)
    blank = extractor.blank_fraction(win)

    space = LabelSpace.from_names(spec["target"], spec["classes"])
    y = space.encode([f.label_fields[spec["target"]] for f in win])
    groups = np.asarray([f.meta["source_file"] for f in win], dtype=object)

    report = {
        "n_flows": len(keep), "n_images": len(win),
        "classes_expected": spec["classes"], "classes_missing": missing,
        "blank_image_fraction": round(blank, 4),
        "n_captures": len(set(groups.tolist())),
        "images_per_class": {c: int((y == space.index(c)).sum()) for c in present},
        "extract_seconds": round(time.time() - t0, 1),
    }
    log.info("[%s] %d flows -> %d images (%s), %d captures, %d/%d classes",
             name, len(keep), len(win), X.shape, report["n_captures"],
             len(present), len(spec["classes"]))
    return X, y, groups, space, report


def paper_split(y: np.ndarray, seed: int, test_size=0.1, val_size=0.1) -> dict:
    """The paper's random 80/10/10 over pooled window images -- leaky, on purpose."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    tr, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                               random_state=seed, stratify=y[rest])
    return {"train": np.sort(tr), "val": np.sort(val), "test": np.sort(test)}


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def run_one(name, X, y, groups, space, protocol, args) -> dict:
    seed = args.seed
    if protocol == "paper":
        split = paper_split(y, seed)
    else:
        split = group_split_three(None, y, groups, test_size=0.2, val_size=0.1, seed=seed)

    tr, va, te = split["train"], split["val"], split["test"]
    if len(te) == 0 or len(tr) == 0:
        return {"error": "empty split"}

    model = ModelRegistry.create(
        args.model, label_space=space,
        epochs=args.epochs, batch_size=args.batch_size, seed=seed,
        class_weight=args.class_weight, early_stop_patience=args.patience,
        gpu_resident_gb=args.gpu_resident_gb,
    )
    t0 = time.time()
    model.fit(X[tr], y[tr], val=(X[va], y[va]) if len(va) else None)
    y_pred = model.predict(X[te])
    m = compute_metrics(y[te], y_pred, n_classes=len(space), class_names=list(space.names))

    per = [v for v in m["per_class"].values() if v["support"] > 0]
    return {
        "protocol": protocol,
        "accuracy": m["accuracy"],
        # The paper reports macro precision/recall/F1 (Table 4).
        "precision": float(np.mean([v["precision"] for v in per])) if per else 0.0,
        "recall": float(np.mean([v["recall"] for v in per])) if per else 0.0,
        "macro_f1": m["macro_f1"],
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "classes_in_test": int(len(set(np.unique(y[te]).tolist()))),
        "train_seconds": round(time.time() - t0, 1),
        "confusion": m["confusion"],
        "per_class": m["per_class"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["all"],
                    help="task ids (1-7), their names, or 'all'")
    ap.add_argument("--protocols", nargs="+", default=["paper", "grouped"])
    ap.add_argument("--windows", nargs="+", type=float, default=[15, 30, 60])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--construction", default="scatter", choices=["scatter", "flowpic"])
    ap.add_argument("--channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--marker", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--class-weight", default=None)
    ap.add_argument("--model", default="flow_image_cnn_baseline",
                    help="registered image model; the variant sweep uses this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-resident-gb", type=float, default=12.0)
    ap.add_argument("--out", type=Path, default=Path("runs/okonkwo/replication.json"))
    args = ap.parse_args()

    wanted = list(TASKS) if "all" in args.tasks else [
        k for k in TASKS if k in args.tasks or k.split("-")[0] in args.tasks
    ]
    if not wanted:
        raise SystemExit(f"no task matched {args.tasks}; available: {list(TASKS)}")

    cache: dict = {}
    results: dict = {"config": vars(args) | {"out": str(args.out)}, "tasks": {}}

    for name in wanted:
        X, y, groups, space, report = build_task(name, TASKS[name], args.windows, cache, args)
        entry = {
            "report": report, "published": PUBLISHED[name], "runs": {},
            "variant": {"model": args.model, "construction": args.construction,
                        "channels": args.channels, "size": args.size},
        }
        for protocol in args.protocols:
            log.info("=== %s / %s ===", name, protocol)
            entry["runs"][protocol] = run_one(name, X, y, groups, space, protocol, args)
        results["tasks"][name] = entry
        del X
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str))

    print_table(results)
    log.info("wrote %s", args.out)


def print_table(results: dict) -> None:
    print(f"\n{'task':<20}{'published':>10}{'paper':>10}{'grouped':>10}{'gap':>8}"
          f"{'images':>9}{'classes':>9}")
    print("-" * 76)
    pub_acc, paper_acc, grp_acc = [], [], []
    for name, e in results["tasks"].items():
        p = e["published"]["accuracy"]
        a = e["runs"].get("paper", {}).get("accuracy")
        g = e["runs"].get("grouped", {}).get("accuracy")
        gap = (a - g) if (a is not None and g is not None) else None
        nc = f"{len(e['report']['images_per_class'])}/{len(e['report']['classes_expected'])}"
        print(f"{name:<20}{p:>10.2%}"
              f"{(f'{a:.2%}' if a is not None else '-'):>10}"
              f"{(f'{g:.2%}' if g is not None else '-'):>10}"
              f"{(f'{gap:+.1%}' if gap is not None else '-'):>8}"
              f"{e['report']['n_images']:>9}{nc:>9}")
        pub_acc.append(p)
        if a is not None: paper_acc.append(a)
        if g is not None: grp_acc.append(g)
    print("-" * 76)
    print(f"{'AVERAGE':<20}{np.mean(pub_acc):>10.2%}"
          f"{(f'{np.mean(paper_acc):.2%}' if paper_acc else '-'):>10}"
          f"{(f'{np.mean(grp_acc):.2%}' if grp_acc else '-'):>10}")
    print("\npublished average from Table 4: 93.86%")


if __name__ == "__main__":
    main()
