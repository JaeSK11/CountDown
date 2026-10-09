#!/usr/bin/env python
"""Re-parse a dataset with the Phase-1 reassembler and check what changed against a backup.

    python scripts/reparse_check.py cstnet --backup cache/cstnet/phase0-backup/flows_f91697df9e38.parquet

The Phase-0 caches of CSTNET and MobileApp predate Phase 1, so their flows carry no
``l7_hint`` and the Phase-5 router cannot be tested on them.  A re-parse fixes that, but it
rewrites the cache every scored run read -- so this reports, per source file, whether the
flows (count, packet arrays, flow_stats features) are the same as before.  If they are, the
Phase-4/5 scores stand for the new cache and only the hint/handshake fields were added.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from countdown.data.base import get_loader  # noqa: E402
from countdown.ensemble.router import Router  # noqa: E402
from countdown.schema import FlowDataset  # noqa: E402


def key(f) -> str:
    return "/".join(Path(str(f.meta.get("source_file"))).parts[-2:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--backup", required=True, help="the pre-reparse parquet cache")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loader = get_loader(args.dataset)
    old = loader._read_cache(Path(args.backup))
    t0 = time.time()
    new_ds = loader.load(refresh=True, workers=args.workers)
    secs = time.time() - t0
    new = new_ds.flows

    def stats(flows):
        ds = FlowDataset(flows, name=args.dataset, default_target=new_ds.default_target,
                         config=new_ds.config)
        return ds.features("flow_stats")

    by_old: dict[str, list] = {}
    for f in old:
        by_old.setdefault(key(f), []).append(f)
    by_new: dict[str, list] = {}
    for f in new:
        by_new.setdefault(key(f), []).append(f)

    same_count = [k for k in by_old if len(by_old[k]) == len(by_new.get(k, []))]
    pkt_same = 0
    for k in same_count:
        a = sorted(by_old[k], key=lambda f: (float(f.timestamps[0]) if len(f.timestamps) else 0.0))
        b = sorted(by_new[k], key=lambda f: (float(f.timestamps[0]) if len(f.timestamps) else 0.0))
        pkt_same += all(np.array_equal(x.sizes, y.sizes) and np.array_equal(x.directions, y.directions)
                        for x, y in zip(a, b))
    feat_equal = None
    if len(old) == len(new) and [key(f) for f in old] == [key(f) for f in new]:
        feat_equal = bool(np.allclose(stats(old), stats(new), equal_nan=True))

    from collections import Counter

    report = {
        "dataset": args.dataset, "parse_seconds": round(secs, 1),
        "flows_old": len(old), "flows_new": len(new),
        "files_old": len(by_old), "files_new": len(by_new),
        "files_same_flow_count": len(same_count),
        "files_same_packets": int(pkt_same),
        "flow_stats_identical_same_order": feat_equal,
        "l7_hint_new": dict(Counter(f.l7_hint for f in new).most_common()),
        "routing": Router().report(new),
    }
    out = Path(args.out or f"runs/reparse/{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
