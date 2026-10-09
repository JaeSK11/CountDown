"""Phase 6.8: the shipped example captures give the documented output.

The crypto verdicts need no model and are always checked; the routing and labels need the
published model zoo and are checked only when ``models_store/`` verifies against the manifest
(a fresh clone after ``countdown models fetch``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
EXPECTED = [json.loads(line) for line in (EX / "expected_fast.jsonl").read_text().splitlines()]
CRYPTO = ("tls_version", "kem_group", "pqc_verdict", "pqc_basis", "quantum_safe", "at_risk", "pqc_listed")
MODEL = ("expert", "route_reason", "target", "traffic_class", "confidence")


def _by_source(records):
    return {Path(r["source"] or "").name or i: r for i, r in enumerate(records)}


def test_examples_crypto_verdicts_without_models():
    from countdown.crypto import enrich_flow
    from countdown.flows.reassembly import Reassembler
    from countdown.sources.pcap import PcapSource

    got = []
    for p in sorted(EX.glob("*.pcap")):
        with PcapSource(p) as src:
            for f in Reassembler().run(src):
                v = enrich_flow(f)
                got.append((p.name, f.kem_group, v.label, v.is_quantum_resistant))
    assert got == [("classical_x25519.pcap", "x25519", "classical", False),
                   ("downgrade_offered_pqc.pcap", "x25519", "classical", False),
                   ("hybrid_x25519mlkem768.pcap", "X25519MLKEM768", "hybrid", True)]
    assert [tuple(r[k] for k in ("kem_group", "pqc_verdict", "quantum_safe")) for r in EXPECTED] == \
        [(g, lab, safe) for _, g, lab, safe in got]


def test_examples_full_output_with_the_published_zoo():
    from countdown.pipeline import zoo

    status = zoo.verify()
    if not status or any(v != "ok" for v in status.values()):
        pytest.skip(f"model zoo not present / not verified: {status}")
    from countdown.pipeline import Pipeline, PipelineConfig

    pipe = Pipeline(PipelineConfig(mode="fast", redact=True))
    got = [r for p in sorted(EX.glob("*.pcap")) for r in pipe.analyze_pcap(p)]
    for g, e in zip(got, EXPECTED):
        assert {k: g[k] for k in CRYPTO + MODEL} == {k: e[k] for k in CRYPTO + MODEL}
