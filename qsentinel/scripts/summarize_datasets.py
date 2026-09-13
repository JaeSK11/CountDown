#!/usr/bin/env python
"""Print a per-dataset summary: flow counts, class distribution and feature dimensions.

    python scripts/summarize_datasets.py                    # all five datasets
    python scripts/summarize_datasets.py iscxvpn iscxtor    # a subset
    python scripts/summarize_datasets.py --refresh          # ignore caches and re-parse
    python scripts/summarize_datasets.py cstnet --top-k-apps 20 --max-per-app 50

The first run parses (and caches) the corpus; later runs read the parquet cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qsentinel.config import Config, setup_logging
from qsentinel.data import available_datasets, get_loader

#: Which label field to report per dataset, beyond its default target.
EXTRA_TARGETS = {
    "iscxvpn": ["tunnel_type"],
    "iscxtor": ["tunnel_type"],
    "mobileapp": ["app"],
}


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def summarize(name: str, args: argparse.Namespace, cfg: Config) -> dict | None:
    loader = get_loader(name, cfg)
    t0 = time.time()
    try:
        ds = loader.load(refresh=args.refresh, workers=args.workers, limit=args.limit)
    except FileNotFoundError as exc:
        print(f"\n{name}: SKIPPED -- {exc}")
        return None
    elapsed = time.time() - t0

    X = ds.features("flow_stats")
    seq = ds.features("packet_seq")
    y = ds.labels()
    st = loader.stats

    print(f"\n{'=' * 78}\n{name}  ({elapsed:.1f}s)\n{'=' * 78}")
    print(f"  flows           : {len(ds):,}")
    print(f"  packets read    : {st.packets_read:,}  "
          f"(used {st.packets_used:,}, non-IP {st.non_ip:,}, "
          f"non-TCP/UDP {st.non_tcp_udp:,}, malformed {st.malformed:,})")
    print(f"  short flows drop: {st.flows_dropped_short:,} "
          f"(< min_packets={cfg.flow.get('min_packets')})")
    if st.truncated_at_max_flows:
        print("  WARNING: at least one capture hit max_flows_per_pcap and was truncated")
    print(f"  packets/flow    : min {min((f.n_packets for f in ds), default=0)}  "
          f"median {int(np.median([f.n_packets for f in ds])) if len(ds) else 0}  "
          f"max {max((f.n_packets for f in ds), default=0)}")
    print(f"  flow_stats X    : {X.shape}  {X.dtype}  finite={bool(np.isfinite(X).all())}")
    print(f"  packet_seq      : {seq.shape}  {seq.dtype}")
    print(f"  target          : {ds.default_target}  ({len(ds.label_names())} classes)")

    for target in [ds.default_target] + EXTRA_TARGETS.get(name, []):
        counts = ds.class_counts(target)
        shown = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        head = shown if len(shown) <= args.max_classes else shown[: args.max_classes]
        print(f"\n  class counts [{target}] ({len(counts)} classes):")
        width = max(len(k) for k, _ in head)
        for label, count in head:
            bar = "#" * max(1, int(40 * count / shown[0][1]))
            print(f"    {label:<{width}}  {count:>7,}  {bar}")
        if len(shown) > len(head):
            rest = sum(c for _, c in shown[len(head):])
            print(f"    {'... ' + str(len(shown) - len(head)) + ' more':<{width}}  {rest:>7,}")

    return {"name": name, "flows": len(ds), "features": X.shape[1],
            "classes": len(ds.label_names()), "seconds": elapsed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", default=None,
                        help=f"datasets to summarize (default: all of {available_datasets()})")
    parser.add_argument("--refresh", action="store_true", help="ignore caches and re-parse")
    parser.add_argument("--workers", type=int, default=None, help="parse worker processes")
    parser.add_argument("--limit", type=int, default=None,
                        help="only read the first N source files (skips the cache)")
    parser.add_argument("--max-classes", type=int, default=25,
                        help="rows to print per class table")
    parser.add_argument("--top-k-apps", type=int, default=None,
                        help="cstnet: keep only the K largest app classes")
    parser.add_argument("--max-per-app", type=int, default=None,
                        help="cstnet: keep at most N pcaps per app")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = Config.load()
    if args.top_k_apps is not None:
        cfg.datasets["cstnet"].setdefault("filters", {})["top_k_apps"] = args.top_k_apps
    if args.max_per_app is not None:
        cfg.datasets["cstnet"].setdefault("filters", {})["max_pcaps_per_app"] = args.max_per_app

    names = args.datasets or available_datasets()
    rows = [r for r in (summarize(n, args, cfg) for n in names) if r]

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'dataset':<16}{'flows':>10}{'features':>10}{'classes':>9}{'load (s)':>10}")
    for r in rows:
        print(f"{r['name']:<16}{r['flows']:>10,}{r['features']:>10}"
              f"{r['classes']:>9}{r['seconds']:>10.1f}")
    print(f"{'TOTAL':<16}{sum(r['flows'] for r in rows):>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
