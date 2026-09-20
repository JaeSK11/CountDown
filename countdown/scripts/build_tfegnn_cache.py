#!/usr/bin/env python
"""Preprocess ISCX pcaps into cached TFE-GNN samples.

    python scripts/build_tfegnn_cache.py --split vpn      --workers 8
    python scripts/build_tfegnn_cache.py --split tor      --workers 8

One shard per capture is written under ``cache/tfegnn/<split>/``, so a re-run skips what
is already there and a crash costs one file rather than the whole corpus.  Graph
construction is the acknowledged bottleneck for this member; this pass gets the corpus
down from 48 GB of pcap to fixed-size byte matrices, which is what makes repeated
training runs affordable.

The four splits are the four columns of the paper's Table 2.  They are separate datasets,
not one pooled one: ISCX-VPN is the tunnelled captures only, ISCX-nonVPN the rest, and
likewise for Tor.  Pooling them would not be comparable to any published number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from countdown.data.base import get_loader  # noqa: E402
from countdown.features.byte_prep import extract_byte_samples  # noqa: E402

#: ISCX-Tor is evaluated over eight user-behaviour categories, which is exactly the
#: project's unified ``traffic_type`` taxonomy.
TOR_CLASSES = (
    "browsing",
    "email",
    "chat",
    "audio_streaming",
    "video_streaming",
    "file_transfer",
    "voip",
    "p2p",
)

#: ISCX-VPN is evaluated over six.  The published six merge audio and video into one
#: "streaming" class and carry no browsing category, so we fold the taxonomy to match
#: rather than reporting a 7- or 8-way number against a 6-way baseline.
VPN_CLASS_MAP = {
    "chat": "chat",
    "email": "email",
    "file_transfer": "file_transfer",
    "audio_streaming": "streaming",
    "video_streaming": "streaming",
    "voip": "voip",
    "p2p": "p2p",
    # "browsing" deliberately absent -> such captures are dropped for this split.
}
VPN_CLASSES = ("chat", "email", "file_transfer", "streaming", "voip", "p2p")

SPLITS = {
    # split      loader      tunnelled?  segment seconds  classes
    "vpn": ("iscxvpn", True, None, VPN_CLASS_MAP),
    "nonvpn": ("iscxvpn", False, None, VPN_CLASS_MAP),
    "tor": ("iscxtor", True, 60.0, {c: c for c in TOR_CLASSES}),
    "nontor": ("iscxtor", False, 60.0, {c: c for c in TOR_CLASSES}),
}


def _shard_is_readable(path: Path) -> bool:
    """A shard counts as cached only if it actually loads."""
    try:
        with np.load(path, allow_pickle=True) as d:
            _ = d["header"].shape
        return True
    except Exception:
        print(f"  .. rebuilding unreadable shard {path.name}")
        path.unlink(missing_ok=True)
        return False


def _tunnel_flag(split: str, fields: dict) -> bool:
    return bool(fields.get("is_vpn" if split.endswith("vpn") else "is_tor", False))


def _process_one(args) -> dict:
    path, label_fields, meta, segment_seconds, class_map, out_path, strip_addr = args
    from countdown.features.byte_prep import PrepStats

    stats = PrepStats()
    label = class_map.get(label_fields.get("traffic_type"))
    if label is None:
        return {"path": str(path), "skipped": "class-not-in-split", "samples": 0}

    try:
        samples = extract_byte_samples(
            path,
            label_fields=label_fields,
            meta=meta,
            segment_seconds=segment_seconds,
            stats=stats,
            strip_addressing=strip_addr,
        )
    except Exception as exc:  # one bad capture must not sink the corpus
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}", "samples": 0}

    if not samples:
        return {"path": str(path), "skipped": "no-samples", "samples": 0, **stats.as_dict()}

    # Write to a temp path and rename: np.savez_compressed is not atomic, and a build
    # killed mid-write leaves a truncated shard that only fails later, at training time.
    tmp_path = Path(str(out_path) + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        header=np.stack([s.header for s in samples]).astype(np.int16),
        payload=np.stack([s.payload for s in samples]).astype(np.int16),
        n_packets=np.array([s.n_packets for s in samples], dtype=np.int32),
        label=np.array([label] * len(samples)),
        source_file=np.array([str(path)] * len(samples)),
    )
    tmp_path.replace(out_path)
    return {"path": str(path), "samples": len(samples), **stats.as_dict()}


def _process_app(args) -> dict:
    """CSTNET: one shard per app.  A pcap there is one TLS session, so per-capture shards
    would mean 46k tiny files; the label is ``app`` and every sample keeps its own
    ``source_file`` so the trainer can group by capture day."""
    app, paths, out_path, strip_addr = args
    from countdown.features.byte_prep import PrepStats

    stats = PrepStats()
    H, P, N, S = [], [], [], []
    errors = 0
    for path in paths:
        try:
            samples = extract_byte_samples(path, label_fields={"app": app}, meta={"source_file": str(path)},
                                           segment_seconds=None, stats=stats, strip_addressing=strip_addr)
        except Exception:
            errors += 1
            continue
        for smp in samples:
            H.append(smp.header); P.append(smp.payload); N.append(smp.n_packets); S.append(str(path))
    if not H:
        return {"path": app, "skipped": "no-samples", "samples": 0, "errors": errors, **stats.as_dict()}
    tmp_path = Path(str(out_path) + ".tmp.npz")
    np.savez_compressed(
        tmp_path, header=np.stack(H).astype(np.int16), payload=np.stack(P).astype(np.int16),
        n_packets=np.asarray(N, dtype=np.int32), label=np.array([app] * len(H)), source_file=np.array(S),
    )
    tmp_path.replace(out_path)
    return {"path": app, "samples": len(H), "errors": errors, **stats.as_dict()}


def _build_cstnet(args) -> int:
    out_dir = Path(args.out) / "cstnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_app: dict[str, list] = {}
    for item in get_loader("cstnet").discover():
        by_app.setdefault(item.label_fields["app"], []).append(item.path)
    apps = sorted(by_app)[: args.limit] if args.limit else sorted(by_app)
    jobs = []
    for app in apps:
        shard = out_dir / f"{app}.npz"
        if shard.exists() and not args.force and _shard_is_readable(shard):
            continue
        jobs.append((app, by_app[app], shard, not args.keep_addressing))
    print(f"[cstnet] {len(apps)} apps, {sum(len(by_app[a]) for a in apps)} pcaps, "
          f"{len(jobs)} shards to build, {args.workers} workers")
    t0, results = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(_process_app, j) for j in jobs]), 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} shards, {sum(r['samples'] for r in results)} samples, "
                      f"{time.time() - t0:.0f}s", flush=True)
    summary = {"split": "cstnet", "shards": len(results), "samples": sum(r["samples"] for r in results),
               "errors": sum(r.get("errors", 0) for r in results), "seconds": round(time.time() - t0, 1)}
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=sorted(SPLITS) + ["cstnet"], required=True)
    ap.add_argument("--out", default="cache/tfegnn", help="cache root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="process at most N captures")
    ap.add_argument("--force", action="store_true", help="rebuild shards that already exist")
    ap.add_argument(
        "--keep-addressing", action="store_true",
        help="reproduce the reference byte layout, which leaves IPs/ports in the header "
             "(diagnostic only -- see byte_prep.DEVIATIONS)",
    )
    args = ap.parse_args()
    if args.split == "cstnet":
        return _build_cstnet(args)

    loader_name, want_tunnel, segment_seconds, class_map = SPLITS[args.split]
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    items = get_loader(loader_name).discover()
    items = [i for i in items if _tunnel_flag(args.split, i.label_fields) == want_tunnel]
    items = [i for i in items if i.label_fields.get("traffic_type") in class_map]
    if args.limit:
        items = items[: args.limit]

    jobs = []
    for item in items:
        shard = out_dir / (Path(item.path).stem + ".npz")
        if shard.exists() and not args.force and _shard_is_readable(shard):
            continue
        jobs.append(
            (item.path, dict(item.label_fields), dict(item.meta), segment_seconds, class_map,
             shard, not args.keep_addressing)
        )

    print(
        f"[{args.split}] {len(items)} captures in split, {len(jobs)} to build "
        f"({len(items) - len(jobs)} cached), {args.workers} workers"
    )
    if not jobs:
        return 0

    t0 = time.time()
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, j): j[0] for j in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if res.get("error"):
                print(f"  !! {Path(res['path']).name}: {res['error']}")
            if done % 10 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                print(
                    f"  {done}/{len(jobs)} captures  {elapsed:6.1f}s  "
                    f"{sum(r['samples'] for r in results)} samples",
                    flush=True,
                )

    total = sum(r["samples"] for r in results)
    errors = [r for r in results if r.get("error")]
    skipped = [r for r in results if r.get("skipped")]
    summary = {
        "split": args.split,
        "captures": len(jobs),
        "samples": total,
        "errors": len(errors),
        "skipped": len(skipped),
        "elapsed_s": round(time.time() - t0, 1),
        "packets_kept": sum(r.get("packets_kept", 0) for r in results),
        "packets_no_payload": sum(r.get("packets_no_payload", 0) for r in results),
        "flows_overlong": sum(r.get("flows_overlong", 0) for r in results),
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if errors:
        print(f"\n{len(errors)} capture(s) failed -- see messages above; shards not written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
