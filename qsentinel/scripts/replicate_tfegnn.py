#!/usr/bin/env python
"""Train TFE-GNN on a cached ISCX split and compare against the published numbers.

    python scripts/replicate_tfegnn.py --split vpn
    python scripts/replicate_tfegnn.py --split tor --runs 3

Reads the shards written by ``build_tfegnn_cache.py``, builds byte-level graphs on the
fly (~11k graphs/s per core, so a few seconds per epoch with workers), trains, and reports
accuracy / precision / recall / macro-F1 next to Zhang et al. Table 2.

Hyperparameters follow the authors' ``config.py`` where it and the paper disagree, since
the code is what produced the published numbers: batch 102 with 5-step gradient
accumulation (~512 effective), 20 epochs, Adam, lr 1e-2 decayed to 1e-4 with 10% warmup.
The split is stratified 9:1 (Sec. 4.1.2), with no separate validation set -- the paper
tunes nothing per-run, so carving a third split out would not match what they did.
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from qsentinel.features.traffic_graph import build_byte_graph  # noqa: E402
from qsentinel.models.tfe_gnn import TFEGNNNet  # noqa: E402

#: Per-dataset training settings from the authors' ``config.py``.  The base ``Config``
#: says 20 epochs and the paper says 120, but **each dataset subclass overrides it** --
#: and the override is what ran.  Training ISCX-Tor for 20 epochs instead of 100 leaves
#: the loss still descending; this table is not a detail.
TRAIN_CFG = {
    "vpn": {"epochs": 20, "lr_min": 1e-4},
    "nonvpn": {"epochs": 120, "lr_min": 1e-5},
    "tor": {"epochs": 100, "lr_min": 1e-4},
    "nontor": {"epochs": 120, "lr_min": 1e-4},
}

#: Zhang et al., WWW'23, Table 2 -- what we are trying to reproduce.
PUBLISHED = {
    "vpn": {"AC": 0.9591, "PR": 0.9526, "RC": 0.9593, "F1": 0.9536, "name": "ISCX-VPN"},
    "nonvpn": {"AC": 0.9040, "PR": 0.9316, "RC": 0.9190, "F1": 0.9240, "name": "ISCX-nonVPN"},
    "tor": {"AC": 0.9886, "PR": 0.9792, "RC": 0.9939, "F1": 0.9855, "name": "ISCX-Tor"},
    "nontor": {"AC": 0.9390, "PR": 0.8742, "RC": 0.8335, "F1": 0.8507, "name": "ISCX-nonTor"},
}


class ByteGraphDataset(Dataset):
    """Cached byte matrices -> per-packet header/payload graphs, built on access.

    Graphs are built in ``__getitem__`` rather than precomputed: at ~88 us per graph a
    worker pool keeps up with the GPU easily, and materialising 1.8M graph objects for
    even the small splits would cost far more memory than it saves time.
    """

    def __init__(self, header: np.ndarray, payload: np.ndarray, labels: np.ndarray) -> None:
        self.header = header
        self.payload = payload
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        h_graphs, p_graphs = [], []
        for rows, acc in ((self.header[idx], h_graphs), (self.payload[idx], p_graphs)):
            for row in rows:
                g = build_byte_graph(row)
                acc.append(
                    Data(
                        x=torch.from_numpy(g.node_values.astype(np.int64)),
                        edge_index=torch.from_numpy(g.edge_index),
                        num_nodes=g.n_nodes,
                    )
                )
        return h_graphs, p_graphs, int(self.labels[idx])


def collate(items):
    h = [g for it in items for g in it[0]]
    p = [g for it in items for g in it[1]]
    y = torch.tensor([it[2] for it in items], dtype=torch.long)
    return Batch.from_data_list(h), Batch.from_data_list(p), y


def load_split(cache_root: Path, split: str):
    files = sorted(glob.glob(str(cache_root / split / "*.npz")))
    if not files:
        raise SystemExit(f"no shards in {cache_root / split} -- run build_tfegnn_cache.py first")
    H, P, L = [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            H.append(d["header"])
            P.append(d["payload"])
            L.append(d["label"])
    header = np.concatenate(H)
    payload = np.concatenate(P)
    labels = np.concatenate(L)
    classes = sorted(set(labels.tolist()))
    y = np.array([classes.index(v) for v in labels], dtype=np.int64)
    return header, payload, y, classes


def make_split(y: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices.

    ``sequential`` reproduces the authors' ``construct_dataset_from_bytes_ISCX``: within
    each class, the *first* 10% of samples become the test set and the rest train.  Since
    samples are accumulated capture by capture, that makes the split largely
    capture-disjoint.

    ``random`` is a stratified shuffle.  It is the weaker evaluation of the two: samples
    from one flow's 60-second blocks are near-duplicates, so a random split scatters them
    across train and test and leaks.  It is offered only to quantify that gap, never as
    the headline.
    """
    if mode == "random":
        return train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)

    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)  # already in capture order
        cut = int(len(idx) / 10) + 1
        test_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def evaluate(net, loader, device, n_packets: int) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    preds, trues = [], []
    with torch.no_grad():
        for hb, pb, y in loader:
            hb, pb = hb.to(device), pb.to(device)
            b = y.shape[0]
            z = net.encode_packets(
                hb.x, hb.edge_index, hb.batch, pb.x, pb.edge_index, pb.batch,
                n_packets=b * n_packets,
            )
            logits = net(z.view(b, n_packets, -1))
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_once(args, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    header, payload, y, classes = load_split(Path(args.cache), args.split)
    n_packets = header.shape[1]

    idx_tr, idx_te = make_split(y, mode=args.split_mode, seed=seed)
    train_ds = ByteGraphDataset(header[idx_tr], payload[idx_tr], y[idx_tr])
    test_ds = ByteGraphDataset(header[idx_te], payload[idx_te], y[idx_te])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.workers, persistent_workers=args.workers > 0,
    )

    net = TFEGNNNet(n_classes=len(classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = max(1, (len(train_loader) // args.accum) * args.epochs)
    warmup = max(1, int(steps * 0.1))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return max(args.lr_min / args.lr, 1.0 - prog * (1.0 - args.lr_min / args.lr))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    lossf = nn.CrossEntropyLoss()

    print(
        f"  {len(idx_tr)} train / {len(idx_te)} test, {len(classes)} classes, "
        f"{n_packets} packets/sample, device={device}"
    )

    for epoch in range(args.epochs):
        net.train()
        t0, total, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (hb, pb, yb) in enumerate(train_loader):
            hb, pb, yb = hb.to(device), pb.to(device), yb.to(device)
            b = yb.shape[0]
            z = net.encode_packets(
                hb.x, hb.edge_index, hb.batch, pb.x, pb.edge_index, pb.batch,
                n_packets=b * n_packets,
            )
            loss = lossf(net(z.view(b, n_packets, -1)), yb) / args.accum
            loss.backward()
            total += float(loss) * args.accum
            nb += 1
            if (i + 1) % args.accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        print(
            f"  epoch {epoch + 1:2d}/{args.epochs}  loss {total / max(nb, 1):.4f}  "
            f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:5.1f}s",
            flush=True,
        )

    true, pred = evaluate(net, test_loader, device, n_packets)
    return {
        "seed": seed,
        "AC": float(accuracy_score(true, pred)),
        "PR": float(precision_score(true, pred, average="macro", zero_division=0)),
        "RC": float(recall_score(true, pred, average="macro", zero_division=0)),
        "F1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "classes": classes,
        "n_test": int(len(true)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=sorted(PUBLISHED), required=True)
    ap.add_argument("--cache", default="cache/tfegnn")
    ap.add_argument("--epochs", type=int, default=0, help="0 = per-dataset default")
    ap.add_argument("--batch-size", type=int, default=102)
    ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lr-min", type=float, default=0.0, help="0 = per-dataset default")
    ap.add_argument("--split-mode", choices=("sequential", "random"), default="sequential")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="runs/tfegnn")
    args = ap.parse_args()

    if not args.epochs:
        args.epochs = TRAIN_CFG[args.split]["epochs"]
    if not args.lr_min:
        args.lr_min = TRAIN_CFG[args.split]["lr_min"]

    ref = PUBLISHED[args.split]
    print(
        f"=== TFE-GNN replication: {ref['name']} ({args.split}) ===\n"
        f"    {args.epochs} epochs, lr {args.lr:.0e}->{args.lr_min:.0e}, "
        f"{args.split_mode} split, batch {args.batch_size}x{args.accum}"
    )

    results = [run_once(args, seed=32 + i) for i in range(args.runs)]

    print(f"\n{'metric':<8} {'ours':>10} {'published':>10} {'delta':>10}")
    summary = {}
    for m in ("AC", "PR", "RC", "F1"):
        vals = [r[m] for r in results]
        mean = float(np.mean(vals))
        summary[m] = {"mean": mean, "std": float(np.std(vals)), "published": ref[m]}
        print(f"{m:<8} {mean:>10.4f} {ref[m]:>10.4f} {mean - ref[m]:>+10.4f}")
    if args.runs > 1:
        print("\n(std over runs: " + ", ".join(f"{m} {summary[m]['std']:.4f}" for m in summary) + ")")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = {
        "split": args.split,
        "published": ref,
        "summary": summary,
        "runs": results,
        "config": vars(args),
    }
    (out_dir / f"{args.split}.json").write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {out_dir / f'{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
