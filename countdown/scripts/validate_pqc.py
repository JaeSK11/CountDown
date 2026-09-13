#!/usr/bin/env python3
"""Validate countdown's crypto parsing against tshark, and report PQC metrics (task 2.9).

Phase 2 is deterministic, which means it can be *checked* rather than merely evaluated: an
independent implementation of the same spec must extract the same codepoint from the same
bytes.  Wireshark's dissector is that independent implementation.  Any disagreement is a
parser bug in one of the two, and on a corpus this size the disagreements are findable.

Three things are measured, and they answer different questions:

1. **Oracle agreement** -- of the flows where *both* tools recovered a negotiated group,
   how often do the codepoints match?  Target >=99%.  This validates the parser.
2. **Coverage** -- how many flows yielded a group at all, and why the rest did not.  A
   parser that agrees with tshark on the three flows it can parse is not a good parser, so
   agreement without coverage is meaningless and both are always printed together.
3. **PQC detection precision / recall / F1** -- with tshark's group as ground truth, treat
   "quantum resistant" (hybrid or pure PQC) as the positive class.

Note on the positive class: real-world PQC positives are scarce (PostQuantumTLS was
captured in 2023, when adoption was ~0), so recall is computed over whatever positives the
corpus contains *plus* the guaranteed ones from ``gen_pqc_pcaps.sh``.  The mix is printed
explicitly rather than being folded into a single headline number.

Usage:
    python scripts/validate_pqc.py --pcaps 'data/_pqc_synth/*.pcap'
    python scripts/validate_pqc.py --pcaps 'data/CSTNET-TLS1.3/**/*.pcap' --limit 300
    python scripts/validate_pqc.py --pcaps '...' --json report.json
"""

from __future__ import annotations

import argparse
import collections
import glob as globlib
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from countdown.config import get_logger, setup_logging  # noqa: E402
from countdown.crypto import kem_registry, parse_handshake, pqc_verdict  # noqa: E402
from countdown.flows import Reassembler  # noqa: E402
from countdown.sources import PcapSource  # noqa: E402

log = get_logger("validate_pqc")

TSHARK_TIMEOUT = 120


# -- oracle -------------------------------------------------------------------------------

def tshark_available() -> bool:
    return shutil.which("tshark") is not None


def tshark_server_groups(path: Path) -> dict[tuple[str, str, str, str], list[int]]:
    """``{(server_ip, server_port, client_ip, client_port): [group_codepoint, ...]}``.

    Filtered to ``tls.handshake.type == 2`` so only the *server's selection* is collected --
    a ClientHello lists every group it offers, and mixing the two would compare our
    negotiated field against the client's wish list.

    A **list** per key, in frame order, because one 5-tuple can legitimately carry several
    ServerHellos: a port is reused, or an idle timeout splits one connection into two
    flows.  Keying a single value per tuple made the second flow -- which correctly has no
    handshake -- look like a parser miss against the first flow's ServerHello.
    """
    cmd = [
        "tshark", "-r", str(path),
        "-Y", "tls.handshake.type == 2",
        "-T", "fields",
        "-e", "ip.src", "-e", "ip.dst",
        "-e", "ipv6.src", "-e", "ipv6.dst",
        "-e", "tcp.srcport", "-e", "tcp.dstport",
        "-e", "udp.srcport", "-e", "udp.dstport",
        "-e", "tls.handshake.extensions_key_share_group",
        "-E", "separator=|",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=TSHARK_TIMEOUT).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("tshark failed on %s: %s", path, exc)
        return {}

    result: dict[tuple[str, str, str, str], list[int]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 9:
            continue
        v4src, v4dst, v6src, v6dst, tsp, tdp, usp, udp_p, group = parts[:9]
        # IPv6 is not a corner case here: a third of this corpus's TLS flows are v6, and
        # asking tshark only for ip.src silently dropped every one of them, which showed up
        # as "we found a group and tshark did not" rather than as a missing field.
        src, dst = (v4src or v6src), (v4dst or v6dst)
        sport, dport = (tsp or usp), (tdp or udp_p)
        if not (src and dst and sport and dport and group):
            continue
        # A frame can carry several handshake messages; the ServerHello's key_share is the
        # first group field in a server-flight frame.
        first = group.split(",")[0].strip()
        try:
            codepoint = int(first, 16) if first.lower().startswith("0x") else int(first)
        except ValueError:
            continue
        result.setdefault((src, sport, dst, dport), []).append(codepoint)
    return result


# -- comparison ---------------------------------------------------------------------------

@dataclass
class Row:
    pcap: str
    flow_id: str
    five_tuple: str
    ours: int | None
    theirs: int | None
    label: str
    basis: str
    parse_ok: bool
    l7_hint: str | None
    transport: str
    reason: str = ""


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    files: int = 0
    flows: int = 0
    oracle_ran: bool = False
    #: ServerHellos tshark found that no flow of ours claimed -- the true coverage gap.
    unclaimed: list[tuple[str, str, int]] = field(default_factory=list)

    # -- coverage ------------------------------------------------------------------------
    @property
    def ours_have_group(self) -> list[Row]:
        return [r for r in self.rows if r.ours is not None]

    @property
    def comparable(self) -> list[Row]:
        """Flows where both tools produced a group -- the only fair agreement denominator."""
        return [r for r in self.rows if r.ours is not None and r.theirs is not None]

    @property
    def agreements(self) -> list[Row]:
        return [r for r in self.comparable if r.ours == r.theirs]

    @property
    def disagreements(self) -> list[Row]:
        return [r for r in self.comparable if r.ours != r.theirs]

    @property
    def missed(self) -> list[Row]:
        """A flow that read a ServerHello but got no group where tshark did."""
        return [r for r in self.rows if r.theirs is not None and r.ours is None]

    @property
    def extra(self) -> list[Row]:
        """We found a group and tshark did not (e.g. it lacks a QUIC key or a codepoint)."""
        return [r for r in self.rows if r.ours is not None and r.theirs is None]


def _is_qr(codepoint: int | None) -> bool:
    return codepoint is not None and kem_registry.is_quantum_resistant(codepoint)


def collect(pcaps: list[Path], use_oracle: bool, context_hint: str | None) -> Report:
    rep = Report()
    rep.oracle_ran = use_oracle
    for path in pcaps:
        rep.files += 1
        try:
            kw = {"context_hint": context_hint} if context_hint else {}
            flows = list(Reassembler(**kw).run(PcapSource(str(path))))
        except Exception as exc:
            log.warning("reassembly failed on %s: %s", path, exc)
            continue

        oracle = tshark_server_groups(path) if use_oracle else {}

        for f in flows:
            rep.flows += 1
            hs = parse_handshake(f)
            v = pqc_verdict(hs)
            ft = f.five_tuple
            # The oracle keys on the ServerHello's own direction: server -> client.
            key = (ft.dst_ip, str(ft.dst_port), ft.src_ip, str(ft.src_port))

            # Claim an oracle ServerHello only for a flow that actually saw one.  Flows are
            # emitted in start order and tshark rows are in frame order, so the first flow
            # on a reused tuple takes the first ServerHello.  Anything left over at the end
            # is a genuine miss rather than an artefact of two flows sharing a 5-tuple.
            # Try the reversed orientation too.  A flow captured without a SYN on a
            # non-standard port (7824, 18443, ... -- common in CSTNET) can be oriented
            # server-first by the reassembler, which does not affect parsing, since both
            # directions are read, but does flip this lookup.
            rkey = (ft.src_ip, str(ft.src_port), ft.dst_ip, str(ft.dst_port))

            theirs = None
            for candidate in (key, rkey):
                pending = oracle.get(candidate)
                if pending and hs.saw_server_hello:
                    theirs = pending.pop(0)
                    break

            rep.rows.append(Row(
                pcap=path.name, flow_id=f.flow_id, five_tuple=str(ft),
                ours=hs.negotiated_group, theirs=theirs,
                label=v.label, basis=v.basis, parse_ok=hs.parse_ok,
                l7_hint=f.l7_hint, transport=hs.transport, reason=v.reason,
            ))

        for key, leftover in oracle.items():
            for codepoint in leftover:
                rep.unclaimed.append((path.name, f"{key[0]}:{key[1]}", codepoint))
    return rep


# -- reporting ----------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:6.2f}%" if den else "   n/a"


def print_report(rep: Report, show: int) -> bool:
    print()
    print("=" * 78)
    print("  PHASE 2 VALIDATION -- crypto parsing vs tshark, and PQC detection metrics")
    print("=" * 78)
    print(f"  kem_groups.yaml version : {kem_registry.table_version()}")
    print(f"  pcap files              : {rep.files}")
    print(f"  flows                   : {rep.flows}")

    # -- 1. coverage -----------------------------------------------------------------------
    parsed = [r for r in rep.rows if r.parse_ok]
    with_group = rep.ours_have_group
    print()
    print("-- 1. parser coverage " + "-" * 56)
    print(f"  handshake parsed        : {len(parsed):6d} / {rep.flows}  {_pct(len(parsed), rep.flows)}")
    print(f"  negotiated group found  : {len(with_group):6d} / {rep.flows}  {_pct(len(with_group), rep.flows)}")

    by_label = collections.Counter(r.label for r in rep.rows)
    by_basis = collections.Counter(r.basis for r in rep.rows)
    print(f"  verdict labels          : {dict(by_label)}")
    print(f"  verdict basis           : {dict(by_basis)}")

    no_group = [r for r in rep.rows if r.ours is None]
    if no_group:
        reasons = collections.Counter(r.reason[:64] for r in no_group)
        print(f"  flows with no group ({len(no_group)}), by reason:")
        for reason, count in reasons.most_common(6):
            print(f"      {count:5d}  {reason}")

    # -- 2. oracle agreement ----------------------------------------------------------------
    ok = True
    print()
    print("-- 2. oracle agreement (tshark) " + "-" * 46)
    if not rep.oracle_ran:
        print("  SKIPPED -- tshark not available or --no-oracle given.")
        print("  Agreement is the only external check on the parser; install tshark to run it.")
    else:
        comp, agree = rep.comparable, rep.agreements
        rate = len(agree) / len(comp) if comp else 0.0
        print(f"  both produced a group   : {len(comp):6d}")
        print(f"  codepoints agree        : {len(agree):6d}          {_pct(len(agree), len(comp))}")
        print(f"  disagree                : {len(rep.disagreements):6d}")
        print(f"  parsed SH but no group  : {len(rep.missed):6d}")
        print(f"  we found, tshark missed : {len(rep.extra):6d}")
        print(f"  tshark SH we never read : {len(rep.unclaimed):6d}"
              f"   <- true coverage gap")
        if comp:
            verdict = "PASS" if rate >= 0.99 else "FAIL"
            if rate < 0.99:
                ok = False
            print(f"  >= 99% target           : {verdict}  ({rate * 100:.2f}%)")
        else:
            print("  no comparable flows -- agreement undefined")

        for r in rep.disagreements[:show]:
            print(f"      MISMATCH {r.pcap} {r.five_tuple}")
            print(f"               ours={kem_registry.name_of(r.ours)} (0x{r.ours:04X})  "
                  f"tshark=0x{r.theirs:04X}")
        for r in rep.missed[:show]:
            print(f"      NO GROUP {r.pcap} {r.five_tuple} tshark=0x{r.theirs:04X} "
                  f"hint={r.l7_hint} reason={r.reason[:44]}")
        for pcap, server, codepoint in rep.unclaimed[:show]:
            print(f"      UNCLAIMED {pcap} server={server} tshark=0x{codepoint:04X}")

    # -- 3. per-family counts ----------------------------------------------------------------
    print()
    print("-- 3. negotiated groups by family " + "-" * 44)
    fams = collections.Counter(kem_registry.classify(r.ours) for r in with_group)
    names = collections.Counter(kem_registry.name_of(r.ours) for r in with_group)
    for fam, count in fams.most_common():
        print(f"  {fam:12s} {count:6d}  {_pct(count, len(with_group))}")
    print("  groups seen:")
    for name, count in names.most_common(12):
        print(f"      {count:6d}  {name}")

    unknown = [r for r in with_group if kem_registry.classify(r.ours) == kem_registry.UNKNOWN]
    grease = [r for r in with_group if kem_registry.is_grease(r.ours)]
    print(f"  unknown codepoints      : {len(unknown):6d}  {_pct(len(unknown), len(with_group))}")
    print(f"  GREASE leaked to verdict: {len(grease):6d}   (must be 0)")
    if grease:
        ok = False

    # -- 4. PQC detection metrics --------------------------------------------------------------
    print()
    print("-- 4. PQC detection (positive class = hybrid or pure PQC) " + "-" * 20)
    if not rep.oracle_ran or not rep.comparable:
        # Without an oracle, score against our own registry over the flows we did parse.
        pos = [r for r in with_group if _is_qr(r.ours)]
        print("  no oracle -- reporting observed positives only (not precision/recall)")
        print(f"  quantum-resistant flows : {len(pos):6d} / {len(with_group)}  "
              f"{_pct(len(pos), len(with_group))}")
    else:
        tp = sum(1 for r in rep.comparable if _is_qr(r.ours) and _is_qr(r.theirs))
        fp = sum(1 for r in rep.comparable if _is_qr(r.ours) and not _is_qr(r.theirs))
        fn = sum(1 for r in rep.comparable if not _is_qr(r.ours) and _is_qr(r.theirs))
        tn = sum(1 for r in rep.comparable if not _is_qr(r.ours) and not _is_qr(r.theirs))
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (tp and (prec + rec)) else float("nan")
        print(f"  TP {tp:5d}   FP {fp:5d}   FN {fn:5d}   TN {tn:5d}")
        print(f"  precision {prec:.4f}   recall {rec:.4f}   F1 {f1:.4f}")
        if tp + fn == 0:
            print("  NOTE: no ground-truth positives in this corpus -- recall is undefined.")
            print("        Run scripts/gen_pqc_pcaps.sh and include data/_pqc_synth/*.pcap.")
        if fp:
            ok = False

    print()
    print("=" * 78)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcaps", required=True, help="glob of pcap files (quote it)")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of pcap files")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--no-oracle", action="store_true", help="skip the tshark cross-check")
    ap.add_argument("--context-hint", default=None, choices=["tor", "vpn"],
                    help="dataset-level tunnel knowledge, as Phase 1 uses it")
    ap.add_argument("--show", type=int, default=10, help="how many mismatches to print")
    ap.add_argument("--json", type=Path, default=None, help="also write the raw rows here")
    args = ap.parse_args()
    setup_logging()

    paths = [Path(p) for p in sorted(globlib.glob(args.pcaps, recursive=True))]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        log.error("no pcap files matched %s", args.pcaps)
        return 2
    if args.limit and len(paths) > args.limit:
        rng = random.Random(args.sample_seed)
        paths = sorted(rng.sample(paths, args.limit))
        log.info("sampled %d of the matching files (seed=%d)", len(paths), args.sample_seed)

    use_oracle = not args.no_oracle and tshark_available()
    if not args.no_oracle and not use_oracle:
        log.warning("tshark not found on PATH -- the oracle cross-check will be skipped")

    log.info("validating %d pcap file(s), oracle=%s", len(paths), use_oracle)
    rep = collect(paths, use_oracle, args.context_hint)
    ok = print_report(rep, args.show)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({
                "kem_table_version": kem_registry.table_version(),
                "files": rep.files, "flows": rep.flows, "oracle_ran": rep.oracle_ran,
                "unclaimed": rep.unclaimed,
                "rows": [vars(r) for r in rep.rows],
            }, fh, indent=2)
        log.info("wrote %s", args.json)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
