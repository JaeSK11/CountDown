#!/usr/bin/env python
"""Train the registered ``graph_gnn_baseline`` on a cached ISCX split through the Phase-3 harness.

    python scripts/train_graph_gnn.py --split tor                       # grouped fold 0
    python scripts/train_graph_gnn.py --split tor --fold 3              # grouped fold 3 of 5
    python scripts/train_graph_gnn.py --split tor --protocol sequential # the authors' split

``--fold k`` picks test fold ``k`` of the ``StratifiedGroupKFold`` that ``eval/splits.py``
draws fold 0 from.  On ISCX-Tor that matters: 51 captures of very unequal size mean fold 0
holds 46 % of the samples and no ``email`` or ``file_transfer`` capture at all, and the
greedy group assignment barely moves with the seed.  Running all five folds and averaging
is the honest grouped number there (the Draper-Gil replication does the same with 10).

This is ``scripts/train.py`` for a member whose input does not come from ``Flow`` objects
yet: the byte matrices live in the cache that ``build_tfegnn_cache.py`` writes, so this
script does steps 1-3 of the harness (load, bind the ``traffic_type`` LabelSpace, split
group-aware by ``source_file``) itself and then hands off to the same ``fit`` /
``evaluate`` / ``write_run`` path every other member uses.  Runs land in
``runs/graph-gnn/<id>/`` with the standard ``metrics.json`` / ``split.json`` / ``model.joblib``.

Differences from ``scripts/replicate_tfegnn.py``, on purpose: the model is the registered
member (so what is scored here is what the ensemble will get); the default protocol is the
grouped split with a val fold carved from train; the class set is the 8-way
``traffic_type`` taxonomy, so absent classes get a zero-support row rather than being
dropped; and macro-F1 comes from ``eval/metrics.py``.  ``--protocol sequential`` reproduces
the replication script's split for a like-for-like check.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from countdown.eval.metrics import evaluate, metrics_excluding_tail  # noqa: E402
from countdown.eval.report import summarize, write_run  # noqa: E402
from countdown.eval.splits import group_split, split_report  # noqa: E402
from countdown.models import ModelRegistry  # noqa: E402
from countdown.schema import LabelSpace  # noqa: E402

#: Per-dataset epoch counts from the authors' ``config.py`` (see replicate_tfegnn.py).
EPOCHS = {"vpn": 20, "nonvpn": 120, "tor": 100, "nontor": 120}
LR_MIN = {"vpn": 1e-4, "nonvpn": 1e-5, "tor": 1e-4, "nontor": 1e-4}
PUBLISHED_F1 = {"vpn": 0.9536, "nonvpn": 0.9240, "tor": 0.9855, "nontor": 0.8507}


def load_cache(root: Path, split: str):
    files = sorted(glob.glob(str(root / split / "*.npz")))
    if not files:
        raise SystemExit(f"no shards in {root / split} -- run build_tfegnn_cache.py first")
    H, P, L, S = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            H.append(d["header"]); P.append(d["payload"])
            L.append(d["label"]); S.append(d["source_file"])
    X = np.concatenate([np.concatenate(H), np.concatenate(P)], axis=2).astype(np.int16)
    return X, np.concatenate(L), np.concatenate(S), int(H[0].shape[2])


def grouped_fold(y: np.ndarray, groups: np.ndarray, fold: int, n_folds: int, seed: int):
    """Test fold ``fold`` of a shuffled ``StratifiedGroupKFold`` -- fold 0 is what
    ``eval.splits.group_split`` returns for ``test_size = 1 / n_folds``."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, groups=groups))
    tr, te = folds[fold]
    return np.sort(tr), np.sort(te)


def sequential_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The authors' split: per class, the first 10 % in capture order is the test set."""
    tr, te = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        cut = int(len(idx) / 10) + 1
        te.append(idx[:cut]); tr.append(idx[cut:])
    return np.concatenate(tr), np.concatenate(te)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(EPOCHS), required=True)
    ap.add_argument("--cache", default="cache/tfegnn")
    ap.add_argument("--protocol", choices=("grouped", "sequential"), default="grouped")
    ap.add_argument("--n-folds", type=int, default=5, help="grouped: K of the StratifiedGroupKFold")
    ap.add_argument("--fold", type=int, default=0, help="grouped: which fold is the test fold")
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=0, help="0 = the authors' per-dataset value")
    ap.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    ap.add_argument("--patience", type=int, default=0, help="early stop on val macro-F1; 0 = off")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs-dir", default="runs/graph-gnn")
    args = ap.parse_args()

    X, names, sources, header_len = load_cache(Path(args.cache), args.split)
    taxonomy = LabelSpace.taxonomy("traffic_type")
    if set(names.tolist()) <= set(taxonomy.names):
        space = taxonomy
    else:
        # The VPN / non-VPN caches carry the paper's six-class set (audio + video merged into
        # "streaming"), which is what Table 2 is scored on; the taxonomy cannot express it.
        space = LabelSpace.from_names("traffic_type_tfegnn", sorted(set(names.tolist())))
    y = space.encode(names)
    print(f"{args.split}: X={X.shape} from {len(set(sources.tolist()))} captures, "
          f"{len(np.unique(y))}/{len(space)} classes realised")

    if args.protocol == "grouped":
        tr, te = grouped_fold(y, sources, args.fold, args.n_folds, args.seed)
        va = np.array([], dtype=np.int64)
        if args.val_size > 0:
            inner = args.val_size / (1.0 - 1.0 / args.n_folds)
            try:
                sub_tr, sub_va = group_split(None, y[tr], sources[tr], test_size=inner, seed=args.seed + 1)
                tr, va = tr[sub_tr], tr[sub_va]
            except ValueError as exc:
                print(f"no val fold ({exc}); training without one")
    else:
        tr, te = sequential_split(y)
        va = np.array([], dtype=np.int64)
    audit = split_report(y, sources, tr, te, len(space), space.names)
    audit["group_key"] = "source_file" if args.protocol == "grouped" else "(sequential, authors')"
    audit["fold"] = f"{args.fold}/{args.n_folds}" if args.protocol == "grouped" else None
    audit["n_val"] = int(len(va))

    params = dict(
        header_len=header_len, epochs=args.epochs or EPOCHS[args.split],
        lr_min=LR_MIN[args.split], class_weight=None if args.class_weight == "none" else "balanced",
        early_stop_patience=args.patience, workers=args.workers, seed=args.seed,
    )
    model = ModelRegistry.create("graph_gnn_baseline", label_space=space, **params)
    t0 = time.time()
    model.fit(X[tr], y[tr], val=(X[va], y[va]) if len(va) else None)
    train_seconds = time.time() - t0

    metrics = evaluate(model, X[te], y[te], space=space, y_train=y[tr], seed=args.seed)
    metrics["tail_excluded"] = metrics_excluding_tail(metrics, 50)
    metrics["published_macro_f1"] = PUBLISHED_F1[args.split]
    metrics["n_params"] = model.n_params()
    metrics["train_seconds"] = train_seconds
    metrics["history"] = model.history_

    config = {
        "model": "graph_gnn_baseline", "dataset": f"iscx-{args.split} (tfegnn cache)",
        "target": space.target, "features": "byte_matrix", "protocol": args.protocol,
        "cache": args.cache, "fold": args.fold, "n_folds": args.n_folds, "val_size": args.val_size,
        "seed": args.seed, "params": params, "n_samples": int(X.shape[0]),
        "feature_shape": [int(d) for d in X.shape[1:]], "label_space": space.to_dict(),
    }
    run_dir = write_run(metrics, config, model=model, split=audit,
                        runs_dir=Path(args.runs_dir), extra={"split": audit})
    tag = f"{args.protocol} fold {args.fold}/{args.n_folds}" if args.protocol == "grouped" else args.protocol
    print(f"\ngraph_gnn_baseline / iscx-{args.split} / traffic_type ({tag})")
    print(summarize(metrics, audit))
    print(f"  published     {PUBLISHED_F1[args.split]:.4f} macro-F1 (authors' protocol)")
    print(f"  train         {train_seconds / 60:.1f} min, {model.n_params() / 1e6:.1f}M params")
    print(f"  run           {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
