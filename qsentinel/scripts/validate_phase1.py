#!/usr/bin/env python
"""Measure the Phase 1 Definition-of-Done claims on the real datasets.

    python scripts/validate_phase1.py                 # sampled, ~1 min
    python scripts/validate_phase1.py --sample 2000   # deeper

Checks, per dataset:

* **DoD 1.7** -- on the direct-TLS corpora (CSTNET, PostQuantumTLS), >90% of TCP :443
  flows hint ``tls`` and carry a captured ServerHello window.
* **DoD 5.4** -- on the tunneled corpora (ISCXVPN, ISCXTor), flows are produced without
  errors; the hint may legitimately be tunnel/opaque and no PQC verdict is expected.
* Handshake trustworthiness: how many windows are marked ``handshake_incomplete``, which
  is what tells Phase 2 to report "not observable" rather than "not PQC".
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsentinel.config import default_config, setup_logging
from qsentinel.flows import Reassembler
from qsentinel.flows.decode import ParseStats
from qsentinel.flows.handshake import TLS_HANDSHAKE, records_complete
from qsentinel.sources import PcapSource

TARGETS = [
    ("cstnet", "CSTNET-TLS1.3/extracted", "**/*.pcap", "tls", None),
    ("postquantumtls", "PostQuantumTLS", "**/*.pcap", "tls", None),
    ("iscxtor", "ISCXTor2016/Tor", "**/*.pcap", "tunneled", "tor"),
    ("iscxvpn", "ISCXVPN2016", "**/vpn_*.pcap", "tunneled", "vpn"),
]


def has_server_hello(buf: bytes | None) -> bool:
    """A TLS handshake record whose first message is ServerHello (type 0x02)."""
    return bool(buf) and len(buf) >= 6 and buf[0] == TLS_HANDSHAKE and buf[5] == 0x02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=250, help="captures per dataset")
    ap.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging("ERROR")
    cfg = default_config()
    rng = random.Random(args.seed)
    failures: list[str] = []

    for name, subdir, pattern, kind, context in TARGETS:
        root = args.data_root / subdir
        if not root.exists():
            print(f"\n=== {name}: SKIPPED (no {root}) ===")
            continue

        files = sorted(root.glob(pattern))
        files = [f for f in files if f.stat().st_size < 400_000_000]
        if not files:
            print(f"\n=== {name}: SKIPPED (no captures matched) ===")
            continue
        chosen = rng.sample(files, min(args.sample, len(files)))

        stats = ParseStats()
        hints: Counter[str] = Counter()
        n_flows = n443 = n443_tls = n443_sh = n_incomplete = n_whole_record = 0
        n_midsession = n_mid_flagged = n_false_alarm = 0
        errors: list[str] = []
        t0 = time.time()

        for path in chosen:
            rsm = Reassembler.from_config(cfg, context_hint=context)
            try:
                for f in rsm.run(PcapSource(path, stats=stats), dataset=name):
                    n_flows += 1
                    hints[f.l7_hint or "?"] += 1
                    n_incomplete += bool(f.handshake_incomplete)
                    if f.l4_proto == "tcp" and f.server_port == 443:
                        n443 += 1
                        if f.l7_hint == "tls":
                            n443_tls += 1
                            sb = f.handshake_server_bytes or b""
                            if has_server_hello(sb):
                                # Handshake IS in the capture: did reassembly recover a
                                # whole record?  This is the metric Phase 1 controls.
                                n443_sh += 1
                                n_whole_record += bool(records_complete(sb))
                                # False-alarm direction: a recoverable handshake must
                                # not be written off as unobservable.
                                n_false_alarm += bool(f.handshake_incomplete
                                                      and records_complete(sb))
                            else:
                                # Connection predates the capture -- no handshake exists
                                # in the file.  It must be flagged, not silently trusted.
                                n_midsession += 1
                                n_mid_flagged += bool(f.handshake_incomplete)
            except Exception:
                errors.append(f"{path.name}: {traceback.format_exc(limit=1).strip()}")

        dt = time.time() - t0
        pct = lambda a, b: (100.0 * a / b) if b else 0.0
        print(f"\n=== {name}  ({len(chosen)} captures, {dt:.1f}s) ===")
        print(f"  flows={n_flows}  packets_read={stats.packets_read}  "
              f"dropped_short={stats.flows_dropped_short}  malformed={stats.malformed}")
        print(f"  hints: " + "  ".join(f"{k}={v}" for k, v in hints.most_common(8)))
        print(f"  handshake_incomplete: {n_incomplete} ({pct(n_incomplete, n_flows):.1f}%)")
        if n443:
            print(f"  TCP :443 flows={n443}  hinted tls={n443_tls} ({pct(n443_tls, n443):.1f}%)")
            print(f"    handshake present in capture: {n443_sh} ({pct(n443_sh, n443_tls):.1f}% of tls)"
                  f"  ->  whole ServerHello record reassembled: {n_whole_record} "
                  f"({pct(n_whole_record, n443_sh):.1f}%)")
            print(f"    recoverable but wrongly flagged incomplete: {n_false_alarm} "
                  f"({pct(n_false_alarm, n443_sh):.1f}%)")
            print(f"    no server handshake in capture: {n_midsession} "
                  f"({pct(n_midsession, n443_tls):.1f}% of tls)  ->  flagged incomplete: "
                  f"{n_mid_flagged} ({pct(n_mid_flagged, n_midsession):.1f}%)")

        if errors:
            failures.append(f"{name}: {len(errors)} captures raised")
            for e in errors[:3]:
                print(f"  ERROR {e}")
        if kind == "tls":
            # DoD 1.7, split into the part Phase 1 controls and the part it cannot.
            if pct(n443_tls, n443) < 90.0:
                failures.append(f"{name}: only {pct(n443_tls,n443):.1f}% of :443 flows hint tls (DoD: >90%)")
            if pct(n_whole_record, n443_sh) < 99.0:
                failures.append(f"{name}: only {pct(n_whole_record,n443_sh):.1f}% of present "
                                f"handshakes reassembled into a whole record (target: >99%)")
            if n_midsession and pct(n_mid_flagged, n_midsession) < 100.0:
                failures.append(f"{name}: {n_midsession - n_mid_flagged} flows with no server "
                                f"handshake NOT flagged incomplete -- Phase 2 would trust a "
                                f"window it cannot reach a verdict from")
            if pct(n_false_alarm, n443_sh) > 1.0:
                failures.append(f"{name}: {n_false_alarm} flows with a whole ServerHello were "
                                f"flagged incomplete ({pct(n_false_alarm,n443_sh):.1f}%) -- "
                                f"Phase 2 would discard usable handshakes")
        elif kind == "tunneled" and n_flows == 0:
            failures.append(f"{name}: produced no flows (DoD: flows produced without errors)")

    print("\n" + "=" * 72)
    if failures:
        print("PHASE 1 VALIDATION FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PHASE 1 VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
