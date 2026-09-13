#!/usr/bin/env python
"""Print the per-flow table for any capture: what Phase 1 hands to Phase 2.

    python scripts/inspect_flows.py <pcap> [more.pcap ...]
    python scripts/inspect_flows.py --dir data/CSTNET-TLS1.3/extracted --limit 20
    python scripts/inspect_flows.py capture.pcap --hex           # handshake hexdump
    python scripts/inspect_flows.py capture.pcap --only tls      # filter by l7_hint

Columns
-------
``5-tuple``   client -> server, canonicalised (client is the SYN sender when there is one)
``npkts``     packets in the flow            ``bytes``  total on-wire bytes
``hint``      coarse L7 guess (advisory -- Phase 2 is authoritative)
``hs c/s``    reassembled handshake bytes captured per direction; ``!`` marks a window
              that cannot be trusted as the stream start (mid-stream capture or a gap)
``rec``       first TLS record as ``type/len``, so you can see a whole ClientHello or
              ServerHello landed before Phase 2 tries to parse it
``expiry``    why the reassembler closed the flow
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from countdown.config import default_config, setup_logging
from countdown.flows import Reassembler
from countdown.flows.decode import ParseStats
from countdown.flows.handshake import describe_records, records_complete
from countdown.sources import DirectorySource, PcapSource

_TLS_TYPES = {0x14: "ccs", 0x15: "alert", 0x16: "hs", 0x17: "app"}


def _record_summary(buf: bytes | None) -> str:
    if not buf:
        return "-"
    recs = describe_records(buf, limit=1)
    if not recs:
        return f"raw:{buf[:2].hex()}"
    ctype, _version, length = recs[0]
    whole = "" if records_complete(buf) else "+"
    return f"{_TLS_TYPES.get(ctype, hex(ctype))}/{length}{whole}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcaps", nargs="*", type=Path, help="capture files")
    ap.add_argument("--dir", type=Path, help="walk a directory of captures instead")
    ap.add_argument("--limit", type=int, default=50, help="max flows to print (0 = all)")
    ap.add_argument("--only", help="only show flows with this l7_hint")
    ap.add_argument("--hex", action="store_true", help="hexdump the handshake windows")
    ap.add_argument("--min-packets", type=int, default=None)
    ap.add_argument("--compat", action="store_true",
                    help="run in Phase-0 batch mode (no FIN/RST teardown)")
    args = ap.parse_args()

    if not args.pcaps and not args.dir:
        ap.error("give at least one pcap, or --dir")

    setup_logging("WARNING")
    cfg = default_config()
    stats = ParseStats()
    overrides = {"phase0_compat": True} if args.compat else {}
    if args.min_packets is not None:
        overrides["min_packets"] = args.min_packets
    rsm = Reassembler.from_config(cfg, **overrides)

    if args.dir:
        source = DirectorySource(args.dir, stats=stats)
        n_files = len(source.files())
        print(f"# {args.dir}  ({n_files} captures)")
    else:
        missing = [p for p in args.pcaps if not p.exists()]
        if missing:
            print(f"no such file: {missing[0]}", file=sys.stderr)
            return 2
        source = (PcapSource(args.pcaps[0], stats=stats) if len(args.pcaps) == 1
                  else _Chain(args.pcaps, stats))
        print(f"# {', '.join(str(p) for p in args.pcaps)}")

    header = (f"{'5-tuple':<52} {'npkts':>6} {'bytes':>10} {'hint':<16} "
              f"{'hs c':>6} {'hs s':>6} {'rec':<10} {'expiry':<14} {'dur':>8}")
    print(header)
    print("-" * len(header))

    hints: Counter[str] = Counter()
    expiries: Counter[str] = Counter()
    shown = n = 0
    incomplete = 0

    for flow in rsm.run(source, stats=stats):
        n += 1
        hints[flow.l7_hint or "?"] += 1
        expiries[flow.expiry_reason or "?"] += 1
        incomplete += bool(flow.handshake_incomplete)
        if args.only and flow.l7_hint != args.only:
            continue
        if args.limit and shown >= args.limit:
            continue
        shown += 1

        cb = flow.handshake_client_bytes or b""
        sb = flow.handshake_server_bytes or b""
        mark = "!" if flow.handshake_incomplete else " "
        five = (f"{flow.client_ip}:{flow.client_port} -> "
                f"{flow.server_ip}:{flow.server_port}/{flow.l4_proto}")
        print(f"{five:<52} {flow.n_packets:>6} {int(flow.sizes.sum()):>10} "
              f"{(flow.l7_hint or '?'):<16} {len(cb):>5}{mark} {len(sb):>6} "
              f"{_record_summary(sb or cb):<10} {(flow.expiry_reason or '?'):<14} "
              f"{flow.duration:>8.2f}")
        if args.hex:
            for name, buf in (("client", cb), ("server", sb)):
                if buf:
                    print(f"      {name} head ({len(buf)} B): {buf[:48].hex(' ')}"
                          f"{' ...' if len(buf) > 48 else ''}")

    if args.limit and n > shown:
        print(f"... {n - shown} more flows not shown (--limit 0 for all)")

    print(f"\nflows: {n}   handshake_incomplete: {incomplete}")
    print("hints:   " + "  ".join(f"{k}={v}" for k, v in hints.most_common()))
    print("expiry:  " + "  ".join(f"{k}={v}" for k, v in expiries.most_common()))
    d = stats.as_dict()
    print(f"packets: read={d['packets_read']} used={d['packets_used']} "
          f"non_ip={d['non_ip']} non_tcp_udp={d['non_tcp_udp']} malformed={d['malformed']} "
          f"dropped_short={d['flows_dropped_short']}")
    return 0


class _Chain:
    """Several explicitly-named captures as one multi-capture source."""

    multi_capture = True

    def __init__(self, paths, stats):
        self.paths, self.stats = paths, stats

    def __iter__(self):
        for p in self.paths:
            yield from PcapSource(p, stats=self.stats)

    def close(self):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
