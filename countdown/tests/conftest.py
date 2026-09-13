"""Shared fixtures: synthetic captures so the unit tests never touch the 90 GB corpus."""

from __future__ import annotations

import socket
from pathlib import Path

import dpkt
import numpy as np
import pytest

CLIENT_IP = "10.0.0.5"
SERVER_IP = "93.184.216.34"
CLIENT_PORT = 51000
SERVER_PORT = 443

TCP_SYN = 0x02
TCP_ACK = 0x10


def _frame(src_ip: str, dst_ip: str, sport: int, dport: int,
           payload_len: int, flags: int = TCP_ACK) -> bytes:
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=b"\x00" * payload_len)
    ip = dpkt.ip.IP(
        src=socket.inet_aton(src_ip),
        dst=socket.inet_aton(dst_ip),
        p=dpkt.ip.IP_PROTO_TCP,
        data=tcp,
    )
    ip.len = len(bytes(ip))
    eth = dpkt.ethernet.Ethernet(
        src=b"\xaa" * 6, dst=b"\xbb" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip
    )
    return bytes(eth)


def write_pcap(path: Path, packets: list[tuple[float, bytes]]) -> Path:
    with open(path, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        for ts, buf in packets:
            writer.writepkt(buf, ts=ts)
    return path


@pytest.fixture
def simple_pcap(tmp_path: Path) -> Path:
    """One TCP flow: client SYN then alternating 100 B up / 500 B down."""
    packets = [(0.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, 0, TCP_SYN))]
    t = 0.0
    for i in range(5):
        t += 0.1
        packets.append((t, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, 100)))
        t += 0.1
        packets.append((t, _frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, 500)))
    return write_pcap(tmp_path / "simple.pcap", packets)


@pytest.fixture
def two_flow_pcap(tmp_path: Path) -> Path:
    """Two interleaved TCP flows on different client ports."""
    packets = []
    t = 0.0
    for i in range(6):
        for port in (51000, 51001):
            packets.append((t, _frame(CLIENT_IP, SERVER_IP, port, SERVER_PORT, 100 + port % 7)))
            t += 0.01
            packets.append((t, _frame(SERVER_IP, CLIENT_IP, SERVER_PORT, port, 300)))
            t += 0.01
    return write_pcap(tmp_path / "two_flows.pcap", packets)


@pytest.fixture
def timeout_pcap(tmp_path: Path) -> Path:
    """One 5-tuple, but a 200 s idle gap splits it into two flows."""
    packets = []
    for base in (0.0, 200.0):
        t = base
        for i in range(6):
            packets.append((t, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, 100)))
            t += 0.1
            packets.append((t, _frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, 200)))
            t += 0.1
    return write_pcap(tmp_path / "timeout.pcap", packets)


@pytest.fixture
def server_first_pcap(tmp_path: Path) -> Path:
    """Capture that starts mid-session with a server packet and carries no SYN."""
    packets = []
    t = 0.0
    for i in range(6):
        packets.append((t, _frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, 500)))
        t += 0.1
        packets.append((t, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, 100)))
        t += 0.1
    return write_pcap(tmp_path / "server_first.pcap", packets)


@pytest.fixture
def synthetic_flows():
    """Hand-built flows for feature tests (no pcap parsing involved)."""
    from countdown.schema import FiveTuple, Flow

    def mk(flow_id: str, n: int, size: int = 100) -> Flow:
        return Flow(
            flow_id=flow_id,
            five_tuple=FiveTuple(CLIENT_IP, CLIENT_PORT, SERVER_IP, SERVER_PORT, "tcp"),
            timestamps=np.arange(n, dtype=np.float64) * 0.5,
            sizes=np.full(n, size, dtype=np.int32),
            directions=np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8),
            dataset="synthetic",
            label="chat",
            label_fields={"traffic_type": "chat", "is_vpn": False, "app": "test"},
        )

    return [mk("a", 10), mk("b", 3), mk("c", 100, size=9000), mk("d", 1)]


# --------------------------------------------------------------------------------------
# Phase 1 fixtures: payload-carrying frames, so head reassembly can be exercised
# --------------------------------------------------------------------------------------

TCP_FIN = 0x01
TCP_RST = 0x04
TCP_PSH = 0x08


def payload_frame(
    src_ip: str, dst_ip: str, sport: int, dport: int, payload: bytes = b"",
    flags: int = TCP_ACK, seq: int = 1, ack: int = 1, v6: bool = False,
) -> bytes:
    """An Ethernet/IP/TCP frame carrying real payload bytes at a chosen sequence number."""
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, seq=seq, ack=ack, data=payload)
    if v6:
        ip = dpkt.ip6.IP6(src=socket.inet_pton(socket.AF_INET6, src_ip),
                          dst=socket.inet_pton(socket.AF_INET6, dst_ip),
                          nxt=dpkt.ip.IP_PROTO_TCP, data=tcp)
        ip.plen = len(bytes(tcp))
        ethertype = dpkt.ethernet.ETH_TYPE_IP6
    else:
        ip = dpkt.ip.IP(src=socket.inet_aton(src_ip), dst=socket.inet_aton(dst_ip),
                        p=dpkt.ip.IP_PROTO_TCP, data=tcp)
        ip.len = len(bytes(ip))
        ethertype = dpkt.ethernet.ETH_TYPE_IP
    return bytes(dpkt.ethernet.Ethernet(src=b"\xaa" * 6, dst=b"\xbb" * 6,
                                        type=ethertype, data=ip))


def udp_frame(src_ip: str, dst_ip: str, sport: int, dport: int, payload: bytes = b"") -> bytes:
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    udp.ulen = len(bytes(udp))
    ip = dpkt.ip.IP(src=socket.inet_aton(src_ip), dst=socket.inet_aton(dst_ip),
                    p=dpkt.ip.IP_PROTO_UDP, data=udp)
    ip.len = len(bytes(ip))
    return bytes(dpkt.ethernet.Ethernet(src=b"\xaa" * 6, dst=b"\xbb" * 6,
                                        type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def tls_record(content_type: int, body: bytes, version: bytes = b"\x03\x03") -> bytes:
    return bytes([content_type]) + version + len(body).to_bytes(2, "big") + body


def client_hello(n: int = 300) -> bytes:
    """A ClientHello-shaped handshake record (type 0x01), body padded to ``n``."""
    return tls_record(0x16, b"\x01" + b"\x00" * (n - 1), version=b"\x03\x01")


def server_hello(n: int = 700) -> bytes:
    """A ServerHello-shaped handshake record (type 0x02), body padded to ``n``."""
    return tls_record(0x16, b"\x02" + b"\x00" * (n - 1))


def quic_initial(version: int = 1, n: int = 200) -> bytes:
    """A QUIC long-header Initial packet: flags, version, then length-prefixed CIDs."""
    return (bytes([0xC3]) + version.to_bytes(4, "big")
            + b"\x08" + b"\x11" * 8 + b"\x04" + b"\x22" * 4 + b"\x00" * n)


@pytest.fixture
def tls_handshake_pcap(tmp_path: Path) -> Path:
    """A TLS flow whose ServerHello arrives in three out-of-order segments, one duplicated.

    This is the case Phase 0's splitter could not handle and Phase 2 cannot parse without:
    the ServerHello record spans segment boundaries.
    """
    sh = server_hello(900)
    ch = client_hello(400)
    pk = [
        (0.00, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"",
                             flags=TCP_SYN, seq=1000, ack=0)),
        (0.01, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, b"",
                             flags=TCP_SYN | TCP_ACK, seq=5000, ack=1001)),
        (0.02, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"", seq=1001, ack=5001)),
        (0.03, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, ch,
                             flags=TCP_PSH | TCP_ACK, seq=1001, ack=5001)),
        # middle, then head, then a duplicate of the middle, then the tail
        (0.05, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, sh[400:700], seq=5401)),
        (0.06, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, sh[0:400], seq=5001)),
        (0.07, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, sh[400:700], seq=5401)),
        (0.08, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, sh[700:],
                             flags=TCP_PSH | TCP_ACK, seq=5701)),
        (0.09, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                             tls_record(0x17, b"\x00" * 60), seq=1001 + len(ch))),
    ]
    return write_pcap(tmp_path / "tls_handshake.pcap", pk)


@pytest.fixture
def teardown_pcap(tmp_path: Path) -> Path:
    """A flow that is torn down with FIN/FIN/ACK, then sees a late straggler."""
    pk, t, seq = [], 0.0, 1
    for _ in range(4):
        t += 0.1
        pk.append((t, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                                    b"\x00" * 100, seq=seq)))
        seq += 100
        t += 0.1
        pk.append((t, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, b"\x00" * 200)))
    t += 0.1
    pk.append((t, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"",
                                flags=TCP_FIN | TCP_ACK, seq=seq)))
    t += 0.1
    pk.append((t, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, b"",
                                flags=TCP_FIN | TCP_ACK)))
    t += 0.1
    pk.append((t, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"", seq=seq + 1)))
    # long after the fin_grace window: a new flow in streaming mode
    t += 30.0
    for _ in range(4):
        t += 0.1
        pk.append((t, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"\x00" * 50)))
    return write_pcap(tmp_path / "teardown.pcap", pk)


@pytest.fixture
def quic_pcap(tmp_path: Path) -> Path:
    """A UDP flow carrying QUIC long-header Initial packets."""
    pk, t = [], 0.0
    for _ in range(5):
        pk.append((t, udp_frame(CLIENT_IP, SERVER_IP, 55000, 443, quic_initial())))
        t += 0.05
        pk.append((t, udp_frame(SERVER_IP, CLIENT_IP, 443, 55000, quic_initial())))
        t += 0.05
    return write_pcap(tmp_path / "quic.pcap", pk)


@pytest.fixture
def vlan_pcap(tmp_path: Path) -> Path:
    """802.1Q-tagged frames -- the l2 header is 18 bytes, not 14."""
    pk = []
    for i in range(6):
        inner = payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"\x00" * 100)
        tagged = inner[:12] + b"\x81\x00\x00\x64" + inner[12:]
        pk.append((i * 0.1, tagged))
    return write_pcap(tmp_path / "vlan.pcap", pk)


@pytest.fixture
def ipv6_pcap(tmp_path: Path) -> Path:
    pk = []
    for i in range(6):
        pk.append((i * 0.1, payload_frame("2001:db8::1", "2001:db8::2", CLIENT_PORT,
                                          SERVER_PORT, b"\x00" * 100, v6=True)))
        pk.append((i * 0.1 + 0.05, payload_frame("2001:db8::2", "2001:db8::1", SERVER_PORT,
                                                 CLIENT_PORT, b"\x00" * 300, v6=True)))
    return write_pcap(tmp_path / "ipv6.pcap", pk)


@pytest.fixture
def garbage_pcap(tmp_path: Path) -> Path:
    """Non-IP and truncated frames mixed into a real flow."""
    pk = [
        (0.0, b"\xaa" * 6 + b"\xbb" * 6 + b"\x08\x06" + b"\x00" * 20),  # ARP
        (0.01, b"\x00" * 8),                                            # truncated junk
        (0.02, b"\xaa" * 6 + b"\xbb" * 6 + b"\x08\x00" + b"\x45" + b"\xff" * 3),  # bad IPv4
    ]
    for i in range(6):
        pk.append((0.1 + i * 0.1, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT,
                                                SERVER_PORT, b"\x00" * 100)))
    return write_pcap(tmp_path / "garbage.pcap", pk)
