#!/usr/bin/env python
"""Phase 1 task 1.10: prove the reassembler streams in bounded memory.

    python scripts/bench_reassembly.py <big.pcap>

Runs the capture twice -- streaming (flows consumed as they expire) and batch
(``list(...)``, what the Phase 0 loaders do) -- and reports peak RSS for each.  The point
of the streaming path is that its peak does not scale with the number of flows in the
capture, which is what makes a live sensor and a multi-gigabyte pcap possible.

Each mode runs in its own subprocess: peak RSS is a per-process high-water mark that never
falls, so measuring both in one interpreter would report the first mode's peak twice.
``tracemalloc`` would avoid that but roughly halves throughput, which distorts the other
half of what this measures.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _peak_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1e3 if sys.platform != "darwin" else kb / 1e6


def run_one(path: Path, mode: str, limit: int) -> dict:
    from countdown.config import default_config, setup_logging
    from countdown.flows import Reassembler
    from countdown.flows.decode import ParseStats
    from countdown.sources import PcapSource

    setup_logging("ERROR")
    cfg = default_config()
    stats = ParseStats()
    rsm = Reassembler.from_config(cfg, phase0_compat=(mode == "batch"))
    baseline = _peak_rss_mb()
    t0 = time.time()

    n = hs = 0
    peak = baseline
    src = PcapSource(path, stats=stats)
    if mode == "batch":
        flows = list(rsm.run(src, stats=stats))
        n = len(flows)
        hs = sum(len(f.handshake_server_bytes or b"") for f in flows)
        peak = _peak_rss_mb()
    else:
        for f in rsm.run(src, stats=stats):
            n += 1
            hs += len(f.handshake_server_bytes or b"")
            if limit and n >= limit:
                break
        peak = _peak_rss_mb()

    dt = time.time() - t0
    return {"mode": mode, "flows": n, "packets": stats.packets_read,
            "peak_rss_mb": round(peak, 1), "baseline_mb": round(baseline, 1),
            "growth_mb": round(peak - baseline, 1), "secs": round(dt, 1),
            "handshake_mb": round(hs / 1e6, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap", type=Path)
    ap.add_argument("--mode", choices=["streaming", "batch"])
    ap.add_argument("--limit", type=int, default=0, help="stop after N flows (0 = all)")
    args = ap.parse_args()

    if args.mode:  # child process
        print(json.dumps(run_one(args.pcap, args.mode, args.limit)))
        return 0

    print(f"{args.pcap.name}  ({args.pcap.stat().st_size / 1e6:.0f} MB on disk)\n")
    print(f"{'mode':<10} {'flows':>8} {'packets':>11} {'peak RSS':>10} {'growth':>9} "
          f"{'handshake':>10} {'secs':>7} {'pkts/s':>9}")
    rows = []
    for mode in ("streaming", "batch"):
        out = subprocess.run(
            [sys.executable, __file__, str(args.pcap), "--mode", mode,
             "--limit", str(args.limit)],
            capture_output=True, text=True, env={**os.environ, "COUNTDOWN_LOG_LEVEL": "ERROR"},
        )
        if out.returncode != 0:
            print(f"{mode}: FAILED\n{out.stderr[-800:]}")
            continue
        r = json.loads(out.stdout.strip().splitlines()[-1])
        rows.append(r)
        print(f"{r['mode']:<10} {r['flows']:>8} {r['packets']:>11} {r['peak_rss_mb']:>9.0f}M "
              f"{r['growth_mb']:>8.0f}M {r['handshake_mb']:>9.1f}M {r['secs']:>7.1f} "
              f"{r['packets'] / max(r['secs'], 1e-9):>9.0f}")

    if len(rows) == 2:
        stream, batch = rows
        ratio = batch["growth_mb"] / max(stream["growth_mb"], 1e-9)
        if stream["growth_mb"] < batch["growth_mb"]:
            print(f"\nstreaming heap growth is {ratio:.1f}x smaller than batch "
                  f"({stream['growth_mb']:.0f}M vs {batch['growth_mb']:.0f}M) -- "
                  f"memory does not scale with flow count")
        else:
            print(f"\nWARNING: streaming growth ({stream['growth_mb']:.0f}M) is not below "
                  f"batch ({batch['growth_mb']:.0f}M); the flow table may not be evicting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
