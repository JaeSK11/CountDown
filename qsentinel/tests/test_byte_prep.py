"""Byte-retaining preprocessing for the TFE-GNN baseline.

These tests build synthetic captures rather than leaning on the ISCX corpus, so they run
without the datasets present and so each preprocessing rule can be checked in isolation.
"""

from __future__ import annotations

import socket
import struct
from pathlib import Path

import numpy as np
import pytest

dpkt = pytest.importorskip("dpkt")

from qsentinel.features.byte_prep import (  # noqa: E402
    PrepStats,
    _canonical_key,
    _pack_sample,
    _segment_by_time,
    _split_packet,
    extract_byte_samples,
)
from qsentinel.features.traffic_graph import (  # noqa: E402
    BYTE_PAD_TRUNC_LENGTH,
    FLOW_PAD_TRUNC_LENGTH,
    HEADER_BYTE_PAD_TRUNC_LENGTH,
    PAD_TRUNC_DIGIT,
)

SRC_IP, DST_IP = "10.1.2.3", "93.184.216.34"
SRC_PORT, DST_PORT = 44444, 443


def _frame(payload: bytes, sport: int = SRC_PORT, dport: int = DST_PORT, proto: str = "tcp") -> bytes:
    """One Ethernet/IPv4/TCP-or-UDP frame carrying ``payload``."""
    if proto == "tcp":
        transport = dpkt.tcp.TCP(sport=sport, dport=dport, seq=1, ack=1, off=5, flags=dpkt.tcp.TH_PUSH)
    else:
        transport = dpkt.udp.UDP(sport=sport, dport=dport)
    transport.data = payload
    if proto == "udp":
        transport.ulen = 8 + len(payload)

    ip = dpkt.ip.IP(
        src=socket.inet_aton(SRC_IP),
        dst=socket.inet_aton(DST_IP),
        p=dpkt.ip.IP_PROTO_TCP if proto == "tcp" else dpkt.ip.IP_PROTO_UDP,
    )
    ip.data = transport
    ip.len = 20 + len(bytes(transport))
    eth = dpkt.ethernet.Ethernet(
        src=b"\xaa" * 6, dst=b"\xbb" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip
    )
    return bytes(eth)


def _write_pcap(path: Path, frames: list[tuple[float, bytes]]) -> Path:
    with open(path, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        for ts, buf in frames:
            writer.writepkt(buf, ts=ts)
    return path


def test_split_packet_removes_addressing() -> None:
    payload = bytes(range(60))
    header, got_payload = _split_packet(_frame(payload), dpkt.pcap.DLT_EN10MB)

    assert got_payload == payload
    # IPv4 header minus its 8 address bytes (12..19), then the TCP header minus its 4
    # port bytes -> 12 + 0 + (20 - 4) = 28 bytes for a no-options IPv4/TCP packet.
    assert len(header) == 28
    assert header[0] == 0x45  # version/IHL survives
    assert header[9] == dpkt.ip.IP_PROTO_TCP  # protocol survives

    # Neither address nor either port may appear anywhere in the header bytes.
    assert socket.inet_aton(SRC_IP) not in header
    assert socket.inet_aton(DST_IP) not in header
    assert struct.pack("!H", SRC_PORT) not in header[:4]
    assert struct.pack("!HH", SRC_PORT, DST_PORT) not in header


def test_split_packet_udp_and_non_transport() -> None:
    header, payload = _split_packet(_frame(b"\x01\x02\x03", proto="udp"), dpkt.pcap.DLT_EN10MB)
    assert payload == b"\x01\x02\x03"
    assert len(header) == 12 + (8 - 4)  # IPv4-minus-addresses + UDP len/checksum

    # ICMP has no ports; it must be rejected rather than mis-sliced.
    ip = dpkt.ip.IP(
        src=socket.inet_aton(SRC_IP), dst=socket.inet_aton(DST_IP), p=dpkt.ip.IP_PROTO_ICMP
    )
    ip.data = dpkt.icmp.ICMP(type=8)
    eth = dpkt.ethernet.Ethernet(
        src=b"\xaa" * 6, dst=b"\xbb" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip
    )
    assert _split_packet(bytes(eth), dpkt.pcap.DLT_EN10MB) is None


def test_canonical_key_is_direction_independent() -> None:
    fwd = dpkt.ethernet.Ethernet(_frame(b"x")).data
    rev_raw = _frame(b"y", sport=DST_PORT, dport=SRC_PORT)
    rev = dpkt.ethernet.Ethernet(rev_raw).data
    rev.src, rev.dst = rev.dst, rev.src
    assert _canonical_key(fwd, fwd.data) == _canonical_key(rev, rev.data)


def test_payloadless_packets_drop_from_both_streams(tmp_path: Path) -> None:
    """Header and payload must stay index-aligned -- the desync the reference risks."""
    frames = [
        (1.0, _frame(b"")),  # pure ACK, no payload -> dropped
        (1.1, _frame(b"A" * 20)),
        (1.2, _frame(b"")),  # dropped
        (1.3, _frame(b"B" * 30)),
    ]
    stats = PrepStats()
    samples = extract_byte_samples(_write_pcap(tmp_path / "c.pcap", frames), stats=stats)

    assert stats.packets_no_payload == 2
    assert stats.packets_kept == 2
    assert len(samples) == 1
    s = samples[0]
    assert s.n_packets == 2
    assert s.header.shape[0] == s.payload.shape[0] == FLOW_PAD_TRUNC_LENGTH
    # Row 0 carries the 'A' packet and row 1 the 'B' packet, in both streams.
    assert s.payload[0][0] == ord("A")
    assert s.payload[1][0] == ord("B")


def test_sample_geometry_and_padding(tmp_path: Path) -> None:
    frames = [(1.0 + i * 0.01, _frame(bytes([i + 1]) * 5)) for i in range(3)]
    samples = extract_byte_samples(_write_pcap(tmp_path / "c.pcap", frames))
    s = samples[0]

    assert s.header.shape == (FLOW_PAD_TRUNC_LENGTH, HEADER_BYTE_PAD_TRUNC_LENGTH)
    assert s.payload.shape == (FLOW_PAD_TRUNC_LENGTH, BYTE_PAD_TRUNC_LENGTH)
    assert s.header.dtype == np.int16 and s.payload.dtype == np.int16
    assert s.n_packets == 3

    # Real payload bytes, then PAD for the rest of the row and all trailing rows.
    assert s.payload[0][:5].tolist() == [1] * 5
    assert s.payload[0][5] == PAD_TRUNC_DIGIT
    assert (s.payload[3:] == PAD_TRUNC_DIGIT).all()
    assert (s.header[3:] == PAD_TRUNC_DIGIT).all()


def test_truncates_to_fifty_packets(tmp_path: Path) -> None:
    frames = [(1.0 + i * 0.001, _frame(b"z" * 10)) for i in range(FLOW_PAD_TRUNC_LENGTH + 25)]
    s = extract_byte_samples(_write_pcap(tmp_path / "c.pcap", frames))[0]
    assert s.n_packets == FLOW_PAD_TRUNC_LENGTH
    assert s.header.shape[0] == FLOW_PAD_TRUNC_LENGTH
    assert not (s.payload == PAD_TRUNC_DIGIT).all(axis=1).any(), "no all-PAD rows when full"


def test_separate_flows_become_separate_samples(tmp_path: Path) -> None:
    frames = [
        (1.0, _frame(b"a" * 10, sport=1111)),
        (1.1, _frame(b"b" * 10, sport=2222)),
        (1.2, _frame(b"c" * 10, sport=1111)),
    ]
    samples = extract_byte_samples(_write_pcap(tmp_path / "c.pcap", frames))
    assert len(samples) == 2
    assert sorted(s.n_packets for s in samples) == [1, 2]


def test_tor_segmentation_cuts_at_sixty_seconds() -> None:
    packets = [(b"h", b"p")] * 5
    times = [0.0, 10.0, 59.9, 60.1, 130.0]
    blocks = list(_segment_by_time(packets, times))
    assert [len(b) for b in blocks] == [3, 1, 1], "0-60, 60-120, 120-180"

    # A flow shorter than one window yields exactly one block, not zero.
    assert [len(b) for b in _segment_by_time(packets[:2], [0.0, 5.0])] == [2]


def test_segmentation_applied_end_to_end(tmp_path: Path) -> None:
    frames = [(t, _frame(b"q" * 8)) for t in (0.0, 1.0, 61.0, 125.0)]
    path = _write_pcap(tmp_path / "c.pcap", frames)
    assert len(extract_byte_samples(path)) == 1  # one flow, unsegmented
    assert len(extract_byte_samples(path, segment_seconds=60.0)) == 3


def test_pack_sample_handles_empty_packet_list() -> None:
    s = _pack_sample([], {}, {})
    assert s.n_packets == 0
    assert s.header.shape == (FLOW_PAD_TRUNC_LENGTH, HEADER_BYTE_PAD_TRUNC_LENGTH)
    assert (s.header == PAD_TRUNC_DIGIT).all()


def test_stats_account_for_every_packet(tmp_path: Path) -> None:
    frames = [(1.0, _frame(b"")), (1.1, _frame(b"x" * 4)), (1.2, _frame(b"y" * 4, proto="udp"))]
    stats = PrepStats()
    extract_byte_samples(_write_pcap(tmp_path / "c.pcap", frames), stats=stats)
    assert stats.packets_read == 3
    accounted = (
        stats.packets_kept
        + stats.packets_no_payload
        + stats.packets_no_ip
        + stats.packets_not_tcp_udp
    )
    assert accounted == stats.packets_read, "no packet may vanish unexplained"


def test_overlong_check_applies_per_segment_not_per_flow(tmp_path: Path, monkeypatch) -> None:
    """Tor's one-giant-flow shape must survive segmentation.

    All Tor traffic rides a few long-lived connections, so a capture is often a single
    5-tuple flow far above the packet threshold.  Rejecting on the parent flow discards
    the whole capture; the threshold belongs on each emitted 60-second block.
    """
    from qsentinel.features import byte_prep

    monkeypatch.setattr(byte_prep, "ANOMALOUS_FLOW_THRESHOLD", 6)

    # One flow of 9 packets spread over three 60s blocks: 4 + 3 + 2.
    times = [0.0, 1.0, 2.0, 3.0, 61.0, 62.0, 63.0, 121.0, 122.0]
    frames = [(t, _frame(b"t" * 6)) for t in times]
    path = _write_pcap(tmp_path / "tor.pcap", frames)

    # Unsegmented: 9 > 6, so the flow is rejected outright.
    stats_flat = PrepStats()
    assert byte_prep.extract_byte_samples(path, stats=stats_flat) == []
    assert stats_flat.flows_overlong == 1

    # Segmented: every block is under the threshold, so all three survive.
    stats_seg = PrepStats()
    samples = byte_prep.extract_byte_samples(path, segment_seconds=60.0, stats=stats_seg)
    assert [s.n_packets for s in samples] == [4, 3, 2]
    assert stats_seg.flows_overlong == 0
