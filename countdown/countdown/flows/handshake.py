"""Handshake-window capture and the coarse L7 transport hint (Phase 1, tasks 1.6-1.7).

Two jobs, both deliberately shallow:

``HeadBuffer``
    Rebuilds the **first N bytes of one direction of a TCP stream** from segments that
    may arrive out of order, duplicated, overlapping, or with the SYN missing entirely.
    This exists for exactly one reason: a TLS ClientHello/ServerHello routinely spans
    several segments, and Phase 2 cannot parse a record that arrives in pieces.  We only
    ever need the head of the stream, so the buffer stops at ``window`` bytes and stops
    at the first unfilled gap -- full bidirectional stream reassembly is out of scope.

``l7_hint``
    A cheap, **transport-level** guess at what protocol a flow carries: does the TCP
    stream open with a TLS record, is this a QUIC long header, does it look like an
    OpenVPN/IKE/WireGuard tunnel.  No field parsing, no cipher extraction, no verdict --
    all of that is Phase 2's authoritative pass.  The hint is advisory: it tells Phase 2
    which parser to try and lets tunneled flows be excluded from PQC verdicts up front.

Memory note: buffers hold at most ``window`` bytes per direction and are trimmed to the
bytes actually captured when the flow is emitted, so a mostly-idle 200k-flow capture does
not pay 2 x 8 KB per flow.
"""

from __future__ import annotations

from typing import Iterable

# ---- protocol constants ---------------------------------------------------------------

TLS_HANDSHAKE = 0x16
TLS_CHANGE_CIPHER_SPEC = 0x14
TLS_ALERT = 0x15
TLS_APPLICATION_DATA = 0x17

#: Legal TLS/SSL record-layer versions: 0x0300 (SSLv3) .. 0x0304 (TLS 1.3).
_TLS_MINORS = frozenset({0x00, 0x01, 0x02, 0x03, 0x04})

#: QUIC versions we recognise.  Exact values plus the IETF-draft and gQUIC families.
_QUIC_EXACT = frozenset({
    0x00000000,  # version negotiation
    0x00000001,  # RFC 9000 (QUIC v1)
    0x6B3343CF,  # draft-29 (Facebook/mvfst deployment value)
    0x709A50C4,  # RFC 9369 (QUIC v2)
})

_IKE_PORTS = frozenset({500, 4500})
_OPENVPN_PORTS = frozenset({1194, 1195})
_WIREGUARD_PORTS = frozenset({51820})
#: Tor relay ORPort/DirPort.  Coarse by design -- a real guard list is out of scope.
_TOR_PORTS = frozenset({9001, 9030, 9051, 9101, 9150})

DEFAULT_WINDOW = 8192

#: Ceilings on the out-of-order holding area, so a pathological stream cannot grow it.
_MAX_PENDING_SEGMENTS = 32

_U32 = 1 << 32


def _seq_delta(seq: int, base: int) -> int:
    """Signed distance ``seq - base`` on the 32-bit TCP sequence circle."""
    d = (seq - base) % _U32
    return d - _U32 if d >= (_U32 >> 1) else d


class HeadBuffer:
    """The reassembled opening bytes of one direction of one flow.

    TCP segments are placed by sequence number; UDP payloads are appended in arrival
    order (datagrams are already whole, and QUIC's first Initial is what matters).

    The buffer is *append-only up to the first gap*: a segment landing beyond the
    contiguous prefix is held in ``_pending`` and drains when the hole fills.  If the hole
    never fills, ``has_gap`` stays True and the flow is marked ``handshake_incomplete`` --
    a truthful "we could not recover this" rather than a silently truncated record.
    """

    __slots__ = ("window", "is_tcp", "base", "isn_known", "_buf", "_pending",
                 "_pending_bytes", "has_gap", "bytes_seen", "segments_seen",
                 "overflowed", "first_payload")

    def __init__(self, window: int = DEFAULT_WINDOW, is_tcp: bool = True) -> None:
        self.window = int(window)
        self.is_tcp = bool(is_tcp)
        self.base: int | None = None
        self.isn_known = False
        self._buf = bytearray()
        self._pending: dict[int, bytes] = {}
        self._pending_bytes = 0
        self.has_gap = False
        self.bytes_seen = 0
        self.segments_seen = 0
        self.overflowed = False
        self.first_payload: bytes = b""

    # -- state ------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buf)

    @property
    def full(self) -> bool:
        return len(self._buf) >= self.window

    @property
    def empty(self) -> bool:
        return not self._buf

    @property
    def trusted_start(self) -> bool:
        """Whether offset 0 of this buffer really is the start of the stream.

        A SYN proves it.  So does a well-formed protocol header sitting at offset 0 --
        and that second route is not a nicety: every one of CSTNET-TLS1.3's 46k captures
        begins mid-connection with the SYN stripped, yet starts exactly on a TLS record
        boundary.  Requiring a SYN would mark that entire dataset unparseable.
        """
        if self.isn_known:
            return True
        return starts_at_stream_start(bytes(self._buf[:8]), self.is_tcp)

    @property
    def complete(self) -> bool:
        """True when the captured prefix is contiguous from the start of the stream."""
        return bool(self._buf) and not self.has_gap and self.trusted_start

    def data(self) -> bytes:
        """The reassembled prefix, trimmed to what was actually captured."""
        return bytes(self._buf)

    # -- filling ----------------------------------------------------------------------
    def note_syn(self, seq: int) -> None:
        """Record the initial sequence number from a SYN, so offset 0 is the true start.

        Without a SYN (mid-stream capture) the first data segment defines the origin and
        ``isn_known`` stays False -- the bytes may not be the start of the stream.
        """
        if self.base is None:
            self.base = (seq + 1) % _U32
            self.isn_known = True

    def add(self, payload: bytes, seq: int = 0) -> None:
        """Place one segment/datagram payload into the head window."""
        if not payload or self.full:
            return
        self.segments_seen += 1
        self.bytes_seen += len(payload)
        if not self.first_payload:
            self.first_payload = bytes(payload[: self.window])

        if not self.is_tcp:
            self._append_at_end(payload)
            return

        if self.base is None:
            # Mid-stream capture: this segment defines the origin, but we cannot claim it
            # is the start of the stream.
            self.base = seq % _U32
            self.isn_known = False

        off = _seq_delta(seq, self.base)
        if off < 0:
            # Starts before our origin: keep only the part we have not already passed.
            skip = -off
            if skip >= len(payload):
                return  # pure retransmit of already-consumed bytes
            payload = payload[skip:]
            off = 0
        if off >= self.window:
            self.has_gap = True
            self.overflowed = True
            return

        payload = payload[: self.window - off]
        self._place(off, payload)

    # -- internals --------------------------------------------------------------------
    def _append_at_end(self, payload: bytes) -> None:
        room = self.window - len(self._buf)
        if room <= 0:
            return
        self._buf += payload[:room]

    def _place(self, off: int, payload: bytes) -> None:
        end = len(self._buf)
        if off <= end:
            # Contiguous or overlapping: append only the genuinely new tail.
            new = payload[end - off:]
            if new:
                self._buf += new
                self._drain()
            return
        # Beyond the contiguous prefix: hold it until the hole fills.
        if len(self._pending) >= _MAX_PENDING_SEGMENTS or self._pending_bytes >= self.window:
            self.has_gap = True
            return
        prev = self._pending.get(off)
        if prev is None or len(payload) > len(prev):
            self._pending_bytes += len(payload) - (len(prev) if prev else 0)
            self._pending[off] = bytes(payload)
        self.has_gap = True

    def _drain(self) -> None:
        """Absorb held segments that are now contiguous with the prefix."""
        if not self._pending:
            return
        progressed = True
        while progressed and self._pending:
            progressed = False
            for off in sorted(self._pending):
                if off > len(self._buf):
                    break
                payload = self._pending.pop(off)
                self._pending_bytes -= len(payload)
                new = payload[len(self._buf) - off:]
                if new:
                    self._buf += new
                progressed = True
                break
        if not self._pending:
            self.has_gap = self.overflowed


# --------------------------------------------------------------------------------------
# Coarse L7 hint
# --------------------------------------------------------------------------------------


def looks_like_tls(buf: bytes) -> bool:
    """True when a TCP stream opens with a plausible TLS/SSL **handshake** record."""
    if len(buf) < 5 or buf[0] != TLS_HANDSHAKE:
        return False
    if buf[1] != 0x03 or buf[2] not in _TLS_MINORS:
        return False
    length = int.from_bytes(buf[3:5], "big")
    return 0 < length <= 0x4000


def looks_like_tls_record(buf: bytes) -> bool:
    """True for any TLS record type -- catches streams captured mid-session."""
    if len(buf) < 5:
        return False
    if buf[0] not in (TLS_HANDSHAKE, TLS_CHANGE_CIPHER_SPEC, TLS_ALERT, TLS_APPLICATION_DATA):
        return False
    if buf[1] != 0x03 or buf[2] not in _TLS_MINORS:
        return False
    return 0 < int.from_bytes(buf[3:5], "big") <= 0x4000


def is_known_quic_version(version: int) -> bool:
    if version in _QUIC_EXACT:
        return True
    if (version & 0xFFFFFF00) == 0xFF000000:  # IETF drafts ff0000xx
        return True
    if (version & 0xFFFF0000) == 0x51300000:  # gQUIC "Q0xx" (e.g. Q043 = 0x51303433)
        return True
    if (version & 0xFFFFFFF0) == 0xFACEB000:  # mvfst experimental
        return True
    return (version & 0x0F0F0F0F) == 0x0A0A0A0A  # GREASE / forced version negotiation


def quic_long_header(buf: bytes) -> tuple[bool, int]:
    """``(is_quic_long_header, version)`` for a UDP payload.

    A QUIC long header is ``0b11xxxxxx`` (header form + fixed bit), then a 4-byte version,
    then length-prefixed connection IDs -- both capped at 20 bytes, which is the cheapest
    structural check that rejects most random UDP.
    """
    if len(buf) < 7:
        return False, 0
    if (buf[0] & 0xC0) != 0xC0:
        return False, 0
    version = int.from_bytes(buf[1:5], "big")
    dcid_len = buf[5]
    if dcid_len > 20 or len(buf) < 6 + dcid_len + 1:
        return False, 0
    if buf[6 + dcid_len] > 20:
        return False, 0
    return is_known_quic_version(version), version


def _looks_like_openvpn(buf: bytes) -> bool:
    """OpenVPN's first byte is ``opcode<<3 | key_id``; opcodes 1-10 are defined."""
    if not buf:
        return False
    opcode = buf[0] >> 3
    return 1 <= opcode <= 10


def _looks_like_isakmp(buf: bytes) -> bool:
    """ISAKMP: 8-byte initiator cookie, 8-byte zero responder cookie on the first message."""
    if len(buf) < 28:
        return False
    if buf[:4] == b"\x00\x00\x00\x00":  # NAT-T non-ESP marker
        buf = buf[4:]
        if len(buf) < 28:
            return False
    version, exch = buf[17], buf[18]
    return version in (0x10, 0x20) and 0 <= exch <= 43


def _looks_like_wireguard(buf: bytes) -> bool:
    """WireGuard message type 1-4 in a little-endian u32, i.e. three zero bytes after it."""
    return len(buf) >= 4 and 1 <= buf[0] <= 4 and buf[1:4] == b"\x00\x00\x00"


def l7_hint(
    proto: str,
    client_head: bytes,
    server_head: bytes,
    client_port: int,
    server_port: int,
    context_hint: str | None = None,
) -> str:
    """Coarse transport-level protocol hint.  Advisory only -- Phase 2 is authoritative.

    Returns one of: ``tls``, ``quic``, ``tunnel_openvpn``, ``tunnel_ike``,
    ``tunnel_wireguard``, ``tor``, ``tcp``, ``udp``, ``opaque``.

    ``context_hint`` carries dataset-level knowledge (e.g. an ISCXTor capture is known to
    be Tor) and is applied only when the wire evidence is unspecific -- a flow that
    genuinely looks like a tunnel handshake keeps its wire-derived hint.
    """
    ports = {int(client_port), int(server_port)}
    head = client_head or server_head

    if proto == "udp":
        if ports & _IKE_PORTS or _looks_like_isakmp(head):
            return "tunnel_ike"
        if ports & _WIREGUARD_PORTS or _looks_like_wireguard(head):
            return "tunnel_wireguard"
        is_quic, _version = quic_long_header(client_head)
        if not is_quic:
            is_quic, _version = quic_long_header(server_head)
        if is_quic:
            return "quic"
        if ports & _OPENVPN_PORTS and _looks_like_openvpn(head):
            return "tunnel_openvpn"
        if context_hint in ("tor", "vpn"):
            return context_hint
        return "udp" if head else "opaque"

    # TCP
    if looks_like_tls(client_head) or looks_like_tls(server_head):
        if context_hint == "tor" or (ports & _TOR_PORTS):
            return "tor"
        return "tls"
    if ports & _OPENVPN_PORTS and _looks_like_openvpn(head):
        return "tunnel_openvpn"
    if looks_like_tls_record(client_head) or looks_like_tls_record(server_head):
        # Mid-session capture: TLS records but no handshake in the window.
        return "tor" if (context_hint == "tor" or ports & _TOR_PORTS) else "tls"
    if context_hint in ("tor", "vpn"):
        return context_hint
    return "tcp" if head else "opaque"


#: Handshake message types that can legitimately *open* a TLS connection.
HS_CLIENT_HELLO = 0x01
HS_SERVER_HELLO = 0x02


def opens_handshake(buf: bytes) -> bool:
    """True when the window opens on the **start** of a TLS handshake.

    Distinguishes a window that begins at a ClientHello/ServerHello from one that merely
    begins on some handshake record from later in the session -- a NewSessionTicket
    (0x04) or KeyUpdate, which is the signature of a capture that started after the
    connection was already established.  Both look like ``0x16 0x03 ..`` on the wire, and
    only the first is parseable into a cipher suite and KEM group.
    """
    return looks_like_tls(buf) and len(buf) >= 6 and buf[5] in (HS_CLIENT_HELLO, HS_SERVER_HELLO)


def starts_at_stream_start(buf: bytes, is_tcp: bool = True) -> bool:
    """True when ``buf`` opens on a recognisable protocol header.

    Used to trust a handshake window captured without a SYN: a TLS record header or a
    QUIC long header at offset 0 is proof that nothing was missed before it.
    """
    if not buf:
        return False
    if is_tcp:
        return looks_like_tls_record(buf)
    return quic_long_header(buf)[0]


def describe_records(buf: bytes, limit: int = 8) -> list[tuple[int, int, int]]:
    """Walk the TLS record layer: ``[(content_type, version, length), ...]``.

    A debugging aid for ``inspect_flows.py`` -- it shows whether a captured window really
    holds a whole ClientHello/ServerHello before Phase 2 tries to parse it.
    """
    out: list[tuple[int, int, int]] = []
    off = 0
    while off + 5 <= len(buf) and len(out) < limit:
        ctype = buf[off]
        version = int.from_bytes(buf[off + 1:off + 3], "big")
        length = int.from_bytes(buf[off + 3:off + 5], "big")
        if ctype not in (TLS_HANDSHAKE, TLS_CHANGE_CIPHER_SPEC, TLS_ALERT,
                         TLS_APPLICATION_DATA) or length == 0 or length > 0x4000:
            break
        out.append((ctype, version, length))
        off += 5 + length
    return out


def records_complete(buf: bytes) -> bool:
    """True when the window holds at least one *whole* TLS record."""
    recs = describe_records(buf, limit=1)
    return bool(recs) and len(buf) >= 5 + recs[0][2]
