"""Phase 1 task 1.4: link/IP/L4 decoding, including the frames Phase 0 only counted."""

from __future__ import annotations

import dpkt
import pytest

from qsentinel.flows.decode import (
    DLT_EN10MB,
    DecodedPacket,
    ParseStats,
    decode,
    iter_ip_packets,
    open_capture,
)
from qsentinel.sources import PcapSource
from tests.conftest import CLIENT_IP, CLIENT_PORT, SERVER_IP, SERVER_PORT, payload_frame


def _decode_one(pcap, index: int = 0) -> DecodedPacket:
    raw = list(PcapSource(pcap))[index]
    pkt = decode(raw.ts, raw.data, raw.linktype)
    assert pkt is not None
    return pkt


def test_decodes_ethernet_ipv4_tcp(simple_pcap):
    pkt = _decode_one(simple_pcap)
    assert (pkt.src_ip, pkt.src_port) == (CLIENT_IP, CLIENT_PORT)
    assert (pkt.dst_ip, pkt.dst_port) == (SERVER_IP, SERVER_PORT)
    assert pkt.proto == "tcp"
    assert pkt.is_pure_syn


def test_decode_returns_payload_and_sequence_numbers(tls_handshake_pcap):
    """Phase 0 discarded both; head reassembly is impossible without them."""
    pkt = _decode_one(tls_handshake_pcap, 3)
    assert pkt.payload[:1] == b"\x16"
    assert pkt.seq == 1001
    assert len(pkt.payload) == 405


def test_decodes_vlan_tagged_frames(vlan_pcap):
    """dpkt unwraps 802.1Q, and the wire size must account for the 4 extra header bytes."""
    pkt = _decode_one(vlan_pcap)
    assert pkt.proto == "tcp"
    assert pkt.src_ip == CLIENT_IP
    plain = decode(0.0, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                                      b"\x00" * 100), DLT_EN10MB)
    assert pkt.size == plain.size + 4


def test_decodes_ipv6(ipv6_pcap):
    pkt = _decode_one(ipv6_pcap)
    assert pkt.src_ip == "2001:db8::1"
    assert pkt.proto == "tcp"
    # IPv6 has no total-length field: size is L2 + 40 byte header + payload length.
    assert pkt.size == 14 + 40 + 20 + 100


def test_decodes_udp_with_zero_tcp_fields(quic_pcap):
    pkt = _decode_one(quic_pcap)
    assert pkt.proto == "udp"
    assert (pkt.flags, pkt.seq, pkt.ack) == (0, 0, 0)
    assert pkt.payload[:1] == b"\xc3"


def test_garbage_is_counted_not_crashed(garbage_pcap):
    """DoD 1.4: a skip counter increments and decoding continues."""
    stats = ParseStats()
    decoded = 0
    for raw in PcapSource(garbage_pcap, stats=stats):
        if decode(raw.ts, raw.data, raw.linktype, stats) is not None:
            decoded += 1
    assert decoded == 6
    assert stats.packets_read == 9
    assert stats.non_ip + stats.malformed + stats.non_tcp_udp == 3


def test_every_frame_is_accounted_for(garbage_pcap):
    stats = ParseStats()
    used = sum(1 for raw in PcapSource(garbage_pcap, stats=stats)
               if decode(raw.ts, raw.data, raw.linktype, stats) is not None)
    assert used + stats.non_ip + stats.non_tcp_udp + stats.malformed == stats.packets_read


def test_tcp_flag_helpers():
    fin = decode(0.0, payload_frame(CLIENT_IP, SERVER_IP, 1, 2, b"", flags=0x11), DLT_EN10MB)
    rst = decode(0.0, payload_frame(CLIENT_IP, SERVER_IP, 1, 2, b"", flags=0x04), DLT_EN10MB)
    synack = decode(0.0, payload_frame(CLIENT_IP, SERVER_IP, 1, 2, b"", flags=0x12), DLT_EN10MB)
    assert fin.is_fin and not fin.is_rst
    assert rst.is_rst
    assert synack.is_syn and not synack.is_pure_syn


def test_open_capture_sniffs_pcapng_named_pcap(tmp_path, simple_pcap):
    """The PostQuantumTLS dataset is pcapng named *.pcap; suffix trust loses it entirely."""
    packets = [(ts, bytes(buf)) for ts, buf in dpkt.pcap.Reader(open(simple_pcap, "rb"))]
    disguised = tmp_path / "actually_pcapng.pcap"
    with open(disguised, "wb") as fh:
        writer = dpkt.pcapng.Writer(fh, linktype=DLT_EN10MB)
        for ts, buf in packets:
            writer.writepkt(buf, ts=ts)
    reader, fh = open_capture(disguised)
    try:
        assert isinstance(reader, dpkt.pcapng.Reader)
        assert sum(1 for _ in reader) == len(packets)
    finally:
        fh.close()


def test_iter_ip_packets_compat_shim_matches_decode(simple_pcap):
    """Phase 0 imports this by name; it must keep yielding the original 8-tuple."""
    rows = list(iter_ip_packets(simple_pcap))
    assert len(rows) == 11
    ts, sip, sport, dip, dport, proto, size, flags = rows[0]
    assert (sip, sport, dip, dport, proto) == (CLIENT_IP, CLIENT_PORT, SERVER_IP, SERVER_PORT, "tcp")
    assert flags == 0x02
