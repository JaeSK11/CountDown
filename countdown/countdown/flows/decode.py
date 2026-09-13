"""Link / IP / L4 decoding: one raw frame -> a ``DecodedPacket``.

Split out of Phase 0's ``flows/extract.py`` so that the streaming reassembler and the
batch extractor share one decode path (Phase 1, task 1.4).  Two things are new relative
to Phase 0:

* the **L4 payload** is returned, so the reassembler can rebuild the handshake window;
* **TCP seq/ack** are returned, so segments can be ordered and retransmits dropped.

The link-layer handling is carried over unchanged -- deliberately, byte for byte -- because
Phase 0's ``(X, y)`` regression depends on the exact ``l2_len`` values it produces
(``tests/test_phase0_regression.py`` pins this).

Sizes are reconstructed as ``L2 header length + IP total length`` rather than ``len(buf)``:
several of these datasets are snaplen-truncated, and the IP header's own length field is
the only surviving record of the real frame size.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import dpkt

# libpcap DLT values we know how to decode.
DLT_NULL = 0
DLT_EN10MB = 1
DLT_RAW_BSD = 12
DLT_RAW_ALT = 14
DLT_RAW = 101
DLT_IEEE802_11 = 105
DLT_LINUX_SLL = 113
DLT_LINUX_SLL2 = 276

PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1",  # LE, microsecond
    b"\xa1\xb2\xc3\xd4",  # BE, microsecond
    b"\x4d\x3c\xb2\xa1",  # LE, nanosecond
    b"\xa1\xb2\x3c\x4d",  # BE, nanosecond
}

# TCP flag bits.
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10


@dataclass
class ParseStats:
    """Per-capture accounting so that nothing is dropped invisibly."""

    packets_read: int = 0
    packets_used: int = 0
    non_ip: int = 0
    non_tcp_udp: int = 0
    malformed: int = 0
    flows_built: int = 0
    flows_dropped_short: int = 0
    truncated_at_max_flows: bool = False

    def merge(self, other: "ParseStats") -> None:
        self.packets_read += other.packets_read
        self.packets_used += other.packets_used
        self.non_ip += other.non_ip
        self.non_tcp_udp += other.non_tcp_udp
        self.malformed += other.malformed
        self.flows_built += other.flows_built
        self.flows_dropped_short += other.flows_dropped_short
        self.truncated_at_max_flows |= other.truncated_at_max_flows

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(slots=True)
class DecodedPacket:
    """One decoded IP/TCP-UDP packet.

    ``size`` is the reconstructed on-wire frame length (not ``len(payload)``).
    ``flags``/``seq``/``ack`` are 0 for UDP.
    """

    ts: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str  # "tcp" | "udp"
    size: int
    flags: int
    seq: int
    ack: int
    payload: bytes

    @property
    def src(self) -> tuple[str, int]:
        return (self.src_ip, self.src_port)

    @property
    def dst(self) -> tuple[str, int]:
        return (self.dst_ip, self.dst_port)

    @property
    def is_syn(self) -> bool:
        return bool(self.flags & TCP_SYN)

    @property
    def is_pure_syn(self) -> bool:
        """SYN without ACK -- the definitive marker of the connection initiator."""
        return bool(self.flags & TCP_SYN) and not (self.flags & TCP_ACK)

    @property
    def is_fin(self) -> bool:
        return bool(self.flags & TCP_FIN)

    @property
    def is_rst(self) -> bool:
        return bool(self.flags & TCP_RST)


def open_capture(path: str | Path):
    """Open a capture as pcap or pcapng, deciding from the file's magic bytes.

    The PostQuantumTLS captures are pcapng files named ``*.pcap``; trusting the suffix
    silently loses that whole dataset.
    """
    path = Path(path)
    fh = open(path, "rb")
    magic = fh.read(4)
    fh.seek(0)
    try:
        if magic == PCAPNG_MAGIC:
            return dpkt.pcapng.Reader(fh), fh
        if magic in PCAP_MAGICS:
            return dpkt.pcap.Reader(fh), fh
        # Unknown magic: try both before giving up.
        try:
            return dpkt.pcap.Reader(fh), fh
        except Exception:
            fh.seek(0)
            return dpkt.pcapng.Reader(fh), fh
    except Exception:
        fh.close()
        raise


# --------------------------------------------------------------------------------------
# Link-layer decoding
# --------------------------------------------------------------------------------------


def _l3_from_raw(buf: bytes):
    """Decode a buffer that starts directly at the IP header."""
    if not buf:
        return None, 0
    version = buf[0] >> 4
    if version == 4:
        return dpkt.ip.IP(buf), 0
    if version == 6:
        return dpkt.ip6.IP6(buf), 0
    return None, 0


def decode_l3(buf: bytes, datalink: int):
    """Return ``(ip_object_or_None, l2_header_len)`` for one raw frame.

    dpkt unwraps 802.1Q / QinQ inside ``Ethernet`` itself, so the VLAN case falls out of
    the Ethernet branch with an ``l2_len`` of 18/22 rather than 14.
    """
    if datalink == DLT_EN10MB:
        eth = dpkt.ethernet.Ethernet(buf)  # dpkt unwraps 802.1Q / QinQ itself
        ip = eth.data
        if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return ip, len(buf) - len(bytes(ip)) if bytes(ip) else 14
        return None, 14
    if datalink == DLT_LINUX_SLL:
        sll = dpkt.sll.SLL(buf)
        ip = sll.data
        return (ip, 16) if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) else (None, 16)
    if datalink == DLT_LINUX_SLL2:
        sll2 = dpkt.sll2.SLL2(buf)
        ip = sll2.data
        return (ip, 20) if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) else (None, 20)
    if datalink == DLT_NULL:
        lo = dpkt.loopback.Loopback(buf)
        ip = lo.data
        return (ip, 4) if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) else (None, 4)
    if datalink in (DLT_RAW, DLT_RAW_BSD, DLT_RAW_ALT):
        return _l3_from_raw(buf)
    if datalink == DLT_IEEE802_11:
        wifi = dpkt.ieee80211.IEEE80211(buf)
        ip = getattr(wifi, "data", None)
        while ip is not None and not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
            ip = getattr(ip, "data", None)
            if isinstance(ip, (bytes, bytearray)):
                ip = None
        return (ip, 24) if ip is not None else (None, 24)
    # Unknown link type: last-resort attempt at Ethernet, then raw IP.
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data, 14
    except Exception:
        pass
    return _l3_from_raw(buf)


# Backwards-compatible private alias (Phase 0 name).
_decode_l3 = decode_l3


def ip_str(raw: bytes) -> str:
    return socket.inet_ntop(socket.AF_INET6 if len(raw) == 16 else socket.AF_INET, raw)


def wire_size(ip, l2_len: int) -> int:
    """Reconstruct on-wire frame length from the IP header's own length field."""
    if isinstance(ip, dpkt.ip6.IP6):
        return l2_len + 40 + int(ip.plen)
    return l2_len + int(ip.len)


_ip_str = ip_str
_wire_size = wire_size


# --------------------------------------------------------------------------------------
# Frame -> DecodedPacket
# --------------------------------------------------------------------------------------


def decode(ts: float, buf: bytes, datalink: int,
           stats: ParseStats | None = None) -> DecodedPacket | None:
    """Decode one frame.  Returns ``None`` for anything not IP/TCP-UDP.

    Every rejection is counted in ``stats`` -- malformed, non-IP or non-TCP-UDP -- so a
    capture never loses packets without leaving a trace.
    """
    try:
        ip, l2_len = decode_l3(buf, datalink)
    except Exception:
        if stats is not None:
            stats.malformed += 1
        return None
    if ip is None:
        if stats is not None:
            stats.non_ip += 1
        return None

    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        proto, flags, seq, ack = "tcp", int(l4.flags), int(l4.seq), int(l4.ack)
    elif isinstance(l4, dpkt.udp.UDP):
        proto, flags, seq, ack = "udp", 0, 0, 0
    else:
        if stats is not None:
            stats.non_tcp_udp += 1
        return None

    payload = l4.data if isinstance(l4.data, (bytes, bytearray)) else bytes(l4.data)

    return DecodedPacket(
        ts=float(ts),
        src_ip=ip_str(ip.src),
        src_port=int(l4.sport),
        dst_ip=ip_str(ip.dst),
        dst_port=int(l4.dport),
        proto=proto,
        size=wire_size(ip, l2_len),
        flags=flags,
        seq=seq,
        ack=ack,
        payload=bytes(payload),
    )


def iter_ip_packets(
    path: str | Path,
    stats: "ParseStats | None" = None,
) -> Iterator[tuple[float, str, int, str, int, str, int, int]]:
    """Phase 0 compatibility shim: yield the original 8-tuple, payload discarded.

    Kept because Phase 0 code and tests import it by name.  New code should go through
    ``PcapSource`` + ``decode``.
    """
    stats = stats if stats is not None else ParseStats()
    reader, fh = open_capture(path)
    try:
        datalink = reader.datalink()
        for ts, buf in reader:
            stats.packets_read += 1
            pkt = decode(ts, bytes(buf), datalink, stats)
            if pkt is None:
                continue
            yield (pkt.ts, pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port,
                   pkt.proto, pkt.size, pkt.flags)
    finally:
        fh.close()
