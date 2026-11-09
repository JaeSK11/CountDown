"""Live capture from a network interface (Phase 6, task 6.3).

The reassembler, handshake capture and crypto parsers are written against ``Source`` and never
learn whether packets came off a disk or a wire.  Two things are different live, and both are
handled here or in the reassembler:

* **A live source never ends**, so flows close only by the expiry rules (idle / FIN-RST /
  max-duration).  Those run on the packet clock; on an idle wire there are no packets, so the
  source yields a **heartbeat** -- a ``RawPacket`` with empty ``data`` at the current time --
  every ``timeout_ms`` without traffic, and the reassembler sweeps on it.
* **Privilege.** A raw ``AF_PACKET`` socket needs ``CAP_NET_RAW`` (root, or the capability).
  The supported way to grant it is the container, not the host Python: the image's
  ``python-netraw`` interpreter carries ``cap_net_raw`` as a file capability (the container runs
  as uid 1000, so ``--cap-add`` alone does not reach it).  See ``docker/README.md``
  (``docker compose run --rm job countdown live eth0``).

Backed by a Linux ``AF_PACKET`` socket -- no libpcap, no new dependency.  Frames arrive as
Ethernet (DLT 1), loopback included.  Filtering is by port in Python (``ports``), not kernel
BPF: compiling a BPF expression needs libpcap or tcpdump, neither of which is in the image.
"""

from __future__ import annotations

import socket
import struct
import time
from typing import Iterator, Sequence

from countdown.sources.base import RawPacket, Source

ETH_P_ALL = 0x0003
DLT_EN10MB = 1
PACKET_OUTGOING = 4
SOL_PACKET = 263
PACKET_ADD_MEMBERSHIP = 1
PACKET_MR_PROMISC = 1
PACKET_STATISTICS = 6


def _ports(frame: bytes) -> tuple[int, int] | None:
    """(sport, dport) of an Ethernet/IPv4|IPv6 TCP/UDP frame, else None.  Enough for a filter."""
    if len(frame) < 14:
        return None
    etype = struct.unpack_from("!H", frame, 12)[0]
    off = 14
    if etype == 0x8100 and len(frame) >= 18:          # one VLAN tag
        etype = struct.unpack_from("!H", frame, 16)[0]
        off = 18
    if etype == 0x0800 and len(frame) >= off + 20:
        ihl = (frame[off] & 0x0F) * 4
        proto, l4 = frame[off + 9], off + ihl
    elif etype == 0x86DD and len(frame) >= off + 40:
        proto, l4 = frame[off + 6], off + 40
    else:
        return None
    if proto not in (6, 17) or len(frame) < l4 + 4:
        return None
    return struct.unpack_from("!HH", frame, l4)


class LiveSource(Source):
    """Capture from a network interface.

    Parameters
    ----------
    iface:
        Interface name, e.g. ``"eth0"`` or ``"lo"``.
    ports:
        Keep only TCP/UDP frames with one of these ports on either side (e.g. ``[443]``).
        ``None`` keeps everything.
    snaplen:
        Bytes read per frame -- generous, so a full ClientHello/ServerHello record fits.
    promiscuous:
        Put the interface into promiscuous mode (also needs ``CAP_NET_ADMIN``).
    timeout_ms:
        Idle time after which a heartbeat is yielded so flow expiry keeps running.
    duration / max_packets:
        Stop after this many seconds / captured frames (None = run until interrupted).
    stop:
        An object with ``is_set()`` (e.g. ``threading.Event``): stop cleanly once it is set,
        checked on every frame and heartbeat.  For a capture driven by the same process
        (``scripts/pqc_census.py``), where the end is "when the probes are done".
    """

    multi_capture = False

    def __init__(
        self,
        iface: str,
        ports: Sequence[int] | None = None,
        snaplen: int = 65535,
        promiscuous: bool = False,
        timeout_ms: int = 500,
        duration: float | None = None,
        max_packets: int | None = None,
        bpf: str | None = None,
        stop: object | None = None,
    ) -> None:
        if bpf:
            raise ValueError("kernel BPF filters are not supported (no libpcap in the image); "
                             "use ports=[...] for a port filter")
        self.iface = iface
        self.ports = None if ports is None else {int(p) for p in ports}
        self.snaplen = int(snaplen)
        self.promiscuous = bool(promiscuous)
        self.timeout_ms = int(timeout_ms)
        self.duration = duration
        self.max_packets = max_packets
        self.stop = stop
        self.packets = 0
        #: Frames the kernel dropped because this socket's receive buffer was full (read
        #: from ``PACKET_STATISTICS`` at close).  A non-zero count means the capture missed
        #: traffic -- report it next to any result built from the capture.
        self.kernel_drops = 0
        self.kernel_seen = 0
        self._sock: socket.socket | None = None

    def _open(self) -> socket.socket:
        try:
            s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(ETH_P_ALL))
        except PermissionError as exc:
            raise PermissionError(
                f"live capture on {self.iface!r} needs CAP_NET_RAW. Run it in the container: "
                f"`docker compose -f docker/compose.yaml run --rm job "
                f"countdown live {self.iface}` (see docker/README.md)") from exc
        s.bind((self.iface, 0))
        if self.promiscuous:
            idx = socket.if_nametoindex(self.iface)
            mreq = struct.pack("iHH8s", idx, PACKET_MR_PROMISC, 0, b"")
            s.setsockopt(SOL_PACKET, PACKET_ADD_MEMBERSHIP, mreq)
        s.settimeout(self.timeout_ms / 1000.0)
        return s

    def __iter__(self) -> Iterator[RawPacket]:
        self._sock = self._open()
        source = f"live:{self.iface}"
        loopback = self.iface == "lo"
        t_end = None if self.duration is None else time.time() + float(self.duration)
        try:
            while True:
                if t_end is not None and time.time() >= t_end:
                    return
                if self.max_packets is not None and self.packets >= self.max_packets:
                    return
                if self.stop is not None and self.stop.is_set():
                    return
                try:
                    data, addr = self._sock.recvfrom(self.snaplen)
                except socket.timeout:
                    yield RawPacket(ts=time.time(), linktype=DLT_EN10MB, data=b"", source=source)
                    continue
                # On loopback every frame is seen twice (outgoing, then incoming): keep one.
                if loopback and len(addr) > 2 and addr[2] == PACKET_OUTGOING:
                    continue
                if self.ports is not None:
                    pp = _ports(data)
                    if pp is None or not (pp[0] in self.ports or pp[1] in self.ports):
                        continue
                self.packets += 1
                yield RawPacket(ts=time.time(), linktype=DLT_EN10MB, data=data, source=source)
        finally:
            self.close()

    def _read_stats(self) -> None:
        """Accumulate the kernel's (packets, drops) counters; reading them resets them."""
        try:
            raw = self._sock.getsockopt(SOL_PACKET, PACKET_STATISTICS, 8)
        except (OSError, AttributeError):        # not an AF_PACKET socket (tests)
            return
        seen, drops = struct.unpack("II", raw)
        self.kernel_seen += seen
        self.kernel_drops += drops

    def close(self) -> None:
        if self._sock is not None:
            self._read_stats()
            self._sock.close()
            self._sock = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LiveSource(iface={self.iface!r}, ports={sorted(self.ports) if self.ports else None})"
