"""Phase 6.3/6.4: live capture pieces that run without privileges -- heartbeat expiry, the
streaming byte collector, streaming == file analysis, the port filter, the privilege error."""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from countdown.features.byte_prep import ByteCollector, extract_byte_samples
from countdown.flows.reassembly import Reassembler
from countdown.sources.base import RawPacket
from countdown.sources.live import LiveSource, _ports
from countdown.sources.pcap import PcapSource


def _packets(path):
    with PcapSource(path) as src:
        return list(src)


def test_heartbeat_expires_idle_flows_before_the_source_ends(two_flow_pcap):
    pkts = _packets(two_flow_pcap)
    last = pkts[-1].ts
    emitted_before_end = []

    def source():
        yield from pkts
        # one heartbeat to arm the sweep, then one far past the idle timeout
        yield RawPacket(ts=last + 1, linktype=pkts[0].linktype, data=b"")
        yield RawPacket(ts=last + 10_000, linktype=pkts[0].linktype, data=b"")
        emitted_before_end.append(len(flows))     # reached only after the heartbeats
    flows = []
    for f in Reassembler(min_packets=1, sweep_interval=0.5).run(source()):
        flows.append(f)
    assert emitted_before_end and emitted_before_end[0] >= 1
    assert {f.expiry_reason for f in flows[:emitted_before_end[0]]} <= {"idle_timeout", "fin", "rst"}


def test_byte_collector_matches_the_file_extractor(two_flow_pcap):
    col = ByteCollector()
    for p in _packets(two_flow_pcap):
        col.add(p.linktype, p.data)
    for s in extract_byte_samples(two_flow_pcap):
        m = col.take(s.meta["five_tuple"])
        np.testing.assert_array_equal(m, np.concatenate([s.header, s.payload], axis=1))
    assert len(col) == 0 and col.take(("x", 1, "y", 2, "tcp")) is None


def test_streaming_gives_the_same_records_as_file_analysis(two_flow_pcap, monkeypatch):
    from countdown.crypto.pqc_verdict import Verdict
    from countdown.pipeline import Pipeline, PipelineConfig
    from tests.test_pipeline import _expert

    monkeypatch.setattr("countdown.crypto.enrich_flow",
                        lambda f: Verdict(label="classical", basis="negotiated", confidence=1.0))
    experts = {"E1": _expert("E1", "traffic_type", ["chat", "file_transfer"]),
               "E2": _expert("E2", "app", ["small.example", "big.example"])}
    pipe = Pipeline(PipelineConfig(mode="fast", fallback=False), experts=experts)
    offline = pipe.analyze_pcap(two_flow_pcap)
    got = []
    pipe.analyze_stream(PcapSource(two_flow_pcap), got.extend, batch_seconds=0.0)
    key = lambda r: (r["five_tuple"], r["start"])                     # noqa: E731
    strip = lambda r: {k: v for k, v in r.items() if k != "flow_id"}   # noqa: E731
    assert [strip(r) for r in sorted(got, key=key)] == [strip(r) for r in sorted(offline, key=key)]


def _eth_ipv4_tcp(sport: int, dport: int) -> bytes:
    eth = b"\x00" * 12 + struct.pack("!H", 0x0800)
    ip = bytes([0x45, 0]) + b"\x00" * 7 + bytes([6]) + b"\x00" * 10
    return eth + ip + struct.pack("!HH", sport, dport) + b"\x00" * 16


def test_port_filter_parser():
    assert _ports(_eth_ipv4_tcp(51000, 443)) == (51000, 443)
    assert _ports(b"\x00" * 10) is None


def test_bpf_is_refused_and_missing_privilege_is_explained():
    with pytest.raises(ValueError, match="ports="):
        LiveSource("lo", bpf="tcp port 443")
    if os.geteuid() == 0:
        pytest.skip("running as root: the raw socket would open")
    try:
        next(iter(LiveSource("lo", duration=0.1)))
    except PermissionError as exc:
        assert "CAP_NET_RAW" in str(exc)
    # (with the capability -- e.g. the container's python-netraw -- it simply captures)


def test_stop_event_ends_the_capture(monkeypatch):
    import socket
    import threading

    class _Quiet:
        def recvfrom(self, n):
            raise socket.timeout

        def close(self):
            pass

    stop = threading.Event()
    src = LiveSource("lo", timeout_ms=1, stop=stop)
    monkeypatch.setattr(src, "_open", lambda: _Quiet())
    beats = 0
    for raw in src:
        beats += 1
        if beats == 3:
            stop.set()
    assert beats == 3
