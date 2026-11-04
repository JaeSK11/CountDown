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

FIELDS = ["flow_id", "five_tuple", "proto", "start", "packets", "bytes", "l7", "tls_version", "cipher",
          "kem_group", "sni", "pqc_verdict", "pqc_basis", "pqc_confidence", "quantum_safe", "pqc_listed",
          "at_risk", "auth_verdict", "cert_key", "expert", "route_reason", "target", "traffic_class",
          "confidence", "hndl_score", "hndl_level", "mitigations", "fallback_from"]


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
                         include_members=args.members, redact=args.redact, store=args.store,
                         allow_unverified=args.allow_unverified,
                         decision_bias=not args.no_decision_bias)
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


def cmd_triage(args: argparse.Namespace) -> int:
    """analyze + the PQC-triage summary, from the pipeline's predicted classes, and the
    harvest-now-decrypt-later risk levels with what to change first (Phase 7)."""
    from countdown.eval.tiers import load_tiers
    from countdown.eval.triage import classes_markdown, to_markdown, triage_records
    from countdown.pipeline import Pipeline, PipelineConfig

    pipe = Pipeline(PipelineConfig(mode=args.mode, context=args.context, store=args.store))
    records = [r for p in args.pcap for r in pipe.analyze_pcap(p)]
    from countdown.risk import summarise, summary_markdown

    t = triage_records(records, tiers=load_tiers())
    t["hndl"] = summarise(records)
    md = [to_markdown(f"{len(args.pcap)} capture(s)", t), summary_markdown(t["hndl"])]
    for target, sub in t["at_risk_classes_by_target"].items():
        md.append(classes_markdown(f"Exposed flows classified as {target} (predicted)", sub))
    md.append("_Class mixes are model predictions. Measured error of the mix on held-out data "
              "(total variation, by flows / by bytes): traffic_type 0.120 / 0.091, app 0.028 / 0.056 "
              "-- see PHASE-7.md section 5. HNDL scores take their sensitivity from these predicted "
              "classes._")
    text = "\n".join(md)
    if args.out:
        Path(args.out).write_text(json.dumps(t, indent=2, default=str) if args.out.endswith(".json") else text)
    print(text)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Capture on an interface and stream verdict records as JSON lines (task 6.4)."""
    from countdown.pipeline import Pipeline, PipelineConfig
    from countdown.sources.live import LiveSource

    pipe = Pipeline(PipelineConfig(mode=args.mode, context=args.context, redact=args.redact,
                                   store=args.store))
    src = LiveSource(args.iface, ports=args.port or None, promiscuous=args.promiscuous,
                     duration=args.duration, max_packets=args.count)
    out = open(args.out, "a") if args.out else sys.stdout
    seen: list[dict[str, Any]] = []

    def emit(records: list[dict[str, Any]]) -> None:
        for r in records:
            out.write(json.dumps(r, default=str) + "\n")
        out.flush()
        seen.extend(records)

    print(f"capturing on {args.iface} ({args.mode} mode); Ctrl-C to stop", file=sys.stderr)
    try:
        pipe.analyze_stream(src, emit, batch_seconds=args.batch_seconds)
    except KeyboardInterrupt:
        pass
    except PermissionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if args.out:
            out.close()
    print(_summary(seen), file=sys.stderr)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from countdown.pipeline import zoo

    if args.action == "fetch":
        status = zoo.fetch(args.tag)
    elif args.action == "manifest":
        print(f"wrote {zoo.write_manifest(tag=args.tag)}", file=sys.stderr)
        status = zoo.verify()
    else:
        status = zoo.verify()
    for name, st in status.items():
        print(f"{name:12s} {st}")
    return 0 if status and all(v == "ok" for v in status.values()) else 1


def cmd_experiments(args: argparse.Namespace) -> int:
    from countdown.eval import experiments

    if args.action == "list":
        for n in experiments.EXPERIMENTS:
            print(n)
        return 0
    names = None if args.all or not args.names else args.names
    for n, path in experiments.run(names).items():
        print(f"{n:14s} -> {path}", file=sys.stderr)
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
    a.add_argument("--no-decision-bias", action="store_true",
                   help="plain argmax: ignore configs/decision_bias.yaml (E1's per-class biases)")
    a.add_argument("--redact", action="store_true", help="drop IPs, SNI and file paths")
    a.add_argument("--store", help="deployed-experts directory (default countdown/models_store)")
    a.add_argument("--allow-unverified", action="store_true",
                   help="load artifacts not listed in models_store.sha256 (your own export)")
    a.set_defaults(func=cmd_analyze)
    t = sub.add_parser("triage", help="pcap(s) -> PQC-triage summary (exposure + what is exposed)")
    t.add_argument("pcap", nargs="+")
    t.add_argument("--out", help="write the summary (.json for the raw numbers, else markdown)")
    t.add_argument("--mode", choices=("full", "fast"), default="full")
    t.add_argument("--context")
    t.add_argument("--store")
    t.set_defaults(func=cmd_triage)
    lv = sub.add_parser("live", help="capture on an interface, stream verdicts (needs CAP_NET_RAW)")
    lv.add_argument("iface")
    lv.add_argument("--mode", choices=("full", "fast"), default="fast",
                    help="fast (default) = GBDTs only, the real-time path")
    lv.add_argument("--port", type=int, action="append", help="keep only these ports (repeatable)")
    lv.add_argument("--duration", type=float, help="stop after N seconds")
    lv.add_argument("--count", type=int, help="stop after N captured frames")
    lv.add_argument("--batch-seconds", type=float, default=2.0)
    lv.add_argument("--promiscuous", action="store_true")
    lv.add_argument("--context")
    lv.add_argument("--redact", action="store_true")
    lv.add_argument("--out", help="append JSON lines here instead of stdout")
    lv.add_argument("--store")
    lv.set_defaults(func=cmd_live)
    mo = sub.add_parser("models", help="fetch / verify the deployed experts (model zoo)")
    mo.add_argument("action", choices=("fetch", "verify", "manifest"))
    mo.add_argument("--tag", default="models-v1")
    mo.set_defaults(func=cmd_models)
    e = sub.add_parser("experiments", help="regenerate the results tables under results/experiments/")
    e.add_argument("action", choices=("run", "list"))
    e.add_argument("names", nargs="*", help="experiments to run (default: all)")
    e.add_argument("--all", action="store_true")
    e.set_defaults(func=cmd_experiments)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
