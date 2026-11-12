"""Phase 10 section 2b: a TLS handshake with nothing after it is ``handshake_only``, not a
traffic class.  Golden capture (2026-11-12, this machine): five ``openssl s_client`` probes
(3 x TLS 1.3, 2 x TLS 1.2 -- what the census sends) and one real HTTPS GET."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from countdown.crypto import enrich_flow
from countdown.crypto.handshake_only import is_handshake_only, tls_records
from countdown.flows import Reassembler
from countdown.sources import PcapSource

GOLDEN = Path(__file__).parent / "golden" / "handshake_only" / "sclient_and_sessions.pcap"


@pytest.fixture(scope="module")
def flows():
    out = list(Reassembler().run(PcapSource(GOLDEN)))
    for f in out:
        enrich_flow(f)
    return out


def test_probes_are_handshake_only_and_the_session_is_not(flows):
    by = {(f.meta["crypto"]["sni"], f.tls_version, is_handshake_only(f)) for f in flows}
    assert by == {("example.com", "1.3", True), ("www.cloudflare.com", "1.3", True),
                  ("www.google.com", "1.3", True), ("example.org", "1.2", True),
                  ("www.wikipedia.org", "1.2", True), ("example.com", "1.3", False)}


def test_a_client_stream_beyond_the_window_is_never_handshake_only(flows):
    f = copy.deepcopy(next(f for f in flows if is_handshake_only(f)))
    f.meta["handshake_bytes_seen"] = [len(f.handshake_client_bytes) + 1, 0]
    assert not is_handshake_only(f)


def test_record_walk():
    rec = lambda t, n: bytes([t, 3, 3]) + n.to_bytes(2, "big") + b"\x00" * n
    assert tls_records(rec(22, 4) + rec(23, 19)) == [(22, 4), (23, 19)]
    assert tls_records(rec(22, 4)[:-1]) is None           # truncated record
    assert tls_records(b"GET / HTTP/1.1\r\n") is None     # not TLS


def test_pipeline_keeps_the_verdict_and_skips_the_experts(flows):
    from countdown.pipeline import Pipeline

    recs = Pipeline(experts={}).analyze_flows(flows)
    hs = [r for r in recs if r["handshake_only"]]
    assert len(hs) == 5
    assert all(r["traffic_class"] == "handshake_only" and r["expert"] is None
               and r["route_reason"] == "handshake_only" and r["pqc_verdict"] for r in hs)
    assert {r["pqc_verdict"] for r in hs} == {"hybrid", "classical"}
