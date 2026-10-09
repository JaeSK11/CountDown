"""``countdown`` command line (Phase 6, task 6.2).

    countdown analyze capture.pcap                      # JSON lines to stdout
    countdown analyze capture.pcap --out verdicts.csv   # format from the suffix
    countdown analyze capture.pcap --mode fast --context mobile --redact

(``python -m countdown`` is the same thing.)

A per-run summary (verdict counts, routing, top classes) goes to stderr, so stdout stays
machine-readable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

FIELDS = ["flow_id", "five_tuple", "proto", "start", "packets", "l7", "tls_version", "cipher",
          "kem_group", "sni", "pqc_verdict", "pqc_basis", "pqc_confidence", "quantum_safe",
          "at_risk", "expert", "route_reason", "target", "traffic_class", "confidence",
          "fallback_from"]


def _write(records: list[dict[str, Any]], out: str | None, fmt: str | None) -> None:
    fmt = fmt or (Path(out).suffix.lstrip(".") if out else "jsonl")
    stream = open(out, "w", newline="") if out else sys.stdout
    try:
        if fmt == "csv":
            w = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                            for k, v in r.items()})
        elif fmt == "json":
            json.dump(records, stream, indent=2, default=str)
            stream.write("\n")
        elif fmt == "jsonl":
            for r in records:
                stream.write(json.dumps(r, default=str) + "\n")
        else:
            raise SystemExit(f"unknown output format {fmt!r} (csv | json | jsonl)")
    finally:
        if out:
            stream.close()


def _summary(records: list[dict[str, Any]]) -> str:
    n = len(records)
    lines = [f"{n} flows"]
    for key in ("pqc_verdict", "route_reason", "expert"):
        c = Counter(str(r.get(key)) for r in records)
        lines.append(f"  {key:13s} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    c = Counter(f"{r['target']}={r['traffic_class']}" for r in records if r.get("traffic_class"))
    if c:
        lines.append("  top classes   " + ", ".join(f"{k} {v}" for k, v in c.most_common(6)))
    return "\n".join(lines)


def cmd_analyze(args: argparse.Namespace) -> int:
    from countdown.pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(mode=args.mode, context=args.context, fallback=not args.no_fallback,
                         fallback_threshold=args.fallback_threshold,
                         classify_quantum_safe=args.classify_quantum_safe,
                         include_members=args.members, redact=args.redact, store=args.store)
    pipe = Pipeline(cfg)
    records = []
    for p in args.pcap:
        recs = pipe.analyze_pcap(p)
        for r in recs:
            r["source"] = None if args.redact else str(p)
        records.extend(recs)
    _write(records, args.out, args.format)
    print(_summary(records), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="countdown", description="Quantum-readiness triage of network traffic.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="pcap(s) -> per-flow verdict records")
    a.add_argument("pcap", nargs="+")
    a.add_argument("--out", help="output file; format from its suffix (.csv/.json/.jsonl)")
    a.add_argument("--format", choices=("csv", "json", "jsonl"))
    a.add_argument("--mode", choices=("full", "fast"), default="full",
                   help="fast = each expert's GBDT only (no GPU, no byte graphs)")
    a.add_argument("--context", help="capture-level routing override, e.g. mobile")
    a.add_argument("--no-fallback", action="store_true", help="disable the E2 -> E1 confidence fallback")
    a.add_argument("--fallback-threshold", type=float, help="override E2's exported threshold")
    a.add_argument("--classify-quantum-safe", action="store_true",
                   help="also classify hybrid/PQC flows (skipped by default)")
    a.add_argument("--members", action="store_true", help="per-member breakdown in each record")
    a.add_argument("--redact", action="store_true", help="drop IPs, SNI and file paths")
    a.add_argument("--store", help="deployed-experts directory (default countdown/models_store)")
    a.set_defaults(func=cmd_analyze)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
