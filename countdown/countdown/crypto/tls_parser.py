"""TLS record + handshake parser (Phase 2, tasks 2.2-2.3).

Turns the opening bytes Phase 1 reassembled (``Flow.handshake_client_bytes`` /
``handshake_server_bytes``) into the cryptographic parameters the verdict engine needs:
TLS version, cipher suite, offered and **negotiated** key-exchange groups, signature
algorithms, plus context (SNI, ALPN, ECH).

Why a hand-rolled parser rather than a TLS library: every TLS library is built to
*complete* a handshake it is a party to.  We are a passive observer with a truncated,
possibly one-sided byte window, and we must extract what is there and truthfully report
what is not.  A library would reject these inputs outright; a parser that reports
``parse_ok=False`` with the fields it did recover is exactly what Stage 3 needs.

Three structural facts drive the design:

* **Only ``0x16`` records are plaintext handshake.**  In TLS 1.3 the server flight is
  ServerHello (plaintext, ``0x16``), then ChangeCipherSpec (``0x14``), then the rest of
  the handshake wrapped in ``0x17`` *application_data* records.  Concatenating only
  ``0x16`` payloads therefore excludes the encrypted messages for free -- and is why
  Certificate/CertificateVerify (and thus PQC *signatures*) are simply not observable.
* **HelloRetryRequest is a ServerHello.**  It reuses the ServerHello message type and is
  distinguished only by a magic ``random`` value, and its ``key_share`` carries a bare
  2-byte *selected_group* instead of a KeyShareEntry.  Parsing it as a normal ServerHello
  reads the group length as the group -- a silent, wrong ``negotiated_group``.
* **Truncation is normal, not exceptional.**  The window is 8 KB and a ServerHello with a
  1216-byte PQC key share plus certificates overruns it routinely.  Every read is bounds
  checked and a short buffer yields partial fields, never an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from countdown.config import get_logger
from countdown.crypto import kem_registry

log = get_logger(__name__)

# -- record layer -----------------------------------------------------------------------

CT_CHANGE_CIPHER_SPEC = 0x14
CT_ALERT = 0x15
CT_HANDSHAKE = 0x16
CT_APPLICATION_DATA = 0x17

#: Handshake message types we care about.
HS_CLIENT_HELLO = 0x01
HS_SERVER_HELLO = 0x02
HS_NEW_SESSION_TICKET = 0x04
HS_CERTIFICATE = 0x0B
HS_SERVER_KEY_EXCHANGE = 0x0C
HS_CERTIFICATE_REQUEST = 0x0D
HS_SERVER_HELLO_DONE = 0x0E

#: TLS 1.3 marks a HelloRetryRequest by setting ServerHello.random to SHA-256("HelloRetryRequest").
HELLO_RETRY_REQUEST_RANDOM = bytes.fromhex(
    "CF21AD74E59A6111BE1D8C021E65B891C2A211167ABB8C5E079E09E2C8A8339C"
)

#: Extension ids (RFC 8446 + IANA).
EXT_SERVER_NAME = 0
EXT_SUPPORTED_GROUPS = 10
EXT_SIGNATURE_ALGORITHMS = 13
EXT_ALPN = 16
EXT_SIGNATURE_ALGORITHMS_CERT = 50
EXT_SUPPORTED_VERSIONS = 43
EXT_PRE_SHARED_KEY = 41
EXT_PSK_KEY_EXCHANGE_MODES = 45
EXT_KEY_SHARE = 51
EXT_ENCRYPTED_CLIENT_HELLO = 0xFE0D  # 65037

_VERSION_NAMES = {
    0x0300: "ssl3.0",
    0x0301: "1.0",
    0x0302: "1.1",
    0x0303: "1.2",
    0x0304: "1.3",
}

#: Maximum plaintext record body (RFC 8446 5.1) -- a larger length means we are not
#: looking at a TLS record and should stop rather than chase a bogus offset.
_MAX_RECORD = 0x4000 + 2048


# -- cipher suites ----------------------------------------------------------------------
# Only what we actually need: a display name, and (for TLS 1.2) the key-exchange family,
# since in 1.2 the key exchange is baked into the suite rather than negotiated separately.

KEX_ECDHE = "ECDHE"
KEX_DHE = "DHE"
KEX_RSA = "RSA"
KEX_ECDH_STATIC = "ECDH"
KEX_PSK = "PSK"
KEX_TLS13 = "TLS1.3"  # key exchange is in key_share, not the suite

_CIPHER_SUITES: dict[int, tuple[str, str | None]] = {
    # TLS 1.3 -- AEAD only; the group comes from key_share.
    0x1301: ("TLS_AES_128_GCM_SHA256", KEX_TLS13),
    0x1302: ("TLS_AES_256_GCM_SHA384", KEX_TLS13),
    0x1303: ("TLS_CHACHA20_POLY1305_SHA256", KEX_TLS13),
    0x1304: ("TLS_AES_128_CCM_SHA256", KEX_TLS13),
    0x1305: ("TLS_AES_128_CCM_8_SHA256", KEX_TLS13),
    # TLS 1.2 ECDHE
    0xC02B: ("TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", KEX_ECDHE),
    0xC02C: ("TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", KEX_ECDHE),
    0xC02F: ("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", KEX_ECDHE),
    0xC030: ("TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", KEX_ECDHE),
    0xC009: ("TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", KEX_ECDHE),
    0xC00A: ("TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", KEX_ECDHE),
    0xC013: ("TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", KEX_ECDHE),
    0xC014: ("TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", KEX_ECDHE),
    0xC023: ("TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256", KEX_ECDHE),
    0xC024: ("TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384", KEX_ECDHE),
    0xC027: ("TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256", KEX_ECDHE),
    0xC028: ("TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384", KEX_ECDHE),
    0xCCA8: ("TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", KEX_ECDHE),
    0xCCA9: ("TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", KEX_ECDHE),
    0xCCAA: ("TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256", KEX_DHE),
    # TLS 1.2 DHE / static RSA
    0x009C: ("TLS_RSA_WITH_AES_128_GCM_SHA256", KEX_RSA),
    0x009D: ("TLS_RSA_WITH_AES_256_GCM_SHA384", KEX_RSA),
    0x009E: ("TLS_DHE_RSA_WITH_AES_128_GCM_SHA256", KEX_DHE),
    0x009F: ("TLS_DHE_RSA_WITH_AES_256_GCM_SHA384", KEX_DHE),
    0x002F: ("TLS_RSA_WITH_AES_128_CBC_SHA", KEX_RSA),
    0x0035: ("TLS_RSA_WITH_AES_256_CBC_SHA", KEX_RSA),
    0x003C: ("TLS_RSA_WITH_AES_128_CBC_SHA256", KEX_RSA),
    0x003D: ("TLS_RSA_WITH_AES_256_CBC_SHA256", KEX_RSA),
    0x0033: ("TLS_DHE_RSA_WITH_AES_128_CBC_SHA", KEX_DHE),
    0x0039: ("TLS_DHE_RSA_WITH_AES_256_CBC_SHA", KEX_DHE),
    0x0067: ("TLS_DHE_RSA_WITH_AES_128_CBC_SHA256", KEX_DHE),
    0x006B: ("TLS_DHE_RSA_WITH_AES_256_CBC_SHA256", KEX_DHE),
    0x000A: ("TLS_RSA_WITH_3DES_EDE_CBC_SHA", KEX_RSA),
    0x0005: ("TLS_RSA_WITH_RC4_128_SHA", KEX_RSA),
    0x0004: ("TLS_RSA_WITH_RC4_128_MD5", KEX_RSA),
    0x00FF: ("TLS_EMPTY_RENEGOTIATION_INFO_SCSV", None),
}


def cipher_suite_name(codepoint: int | None) -> str | None:
    """Display name for a cipher suite; ``unknown(0x....)`` when unlisted."""
    if codepoint is None:
        return None
    if kem_registry.is_grease(codepoint):
        return f"GREASE(0x{codepoint:04X})"
    entry = _CIPHER_SUITES.get(codepoint)
    return entry[0] if entry else f"unknown(0x{codepoint:04X})"


def kex_family(codepoint: int | None) -> str | None:
    """Key-exchange family implied by a TLS 1.2 cipher suite.

    ``None`` when the suite is unknown -- guessing an ECDHE/RSA split from a codepoint we
    do not recognise would be inventing evidence.  TLS 1.3 suites return ``TLS1.3``: the
    exchange is negotiated in ``key_share``, not implied by the suite.
    """
    if codepoint is None:
        return None
    entry = _CIPHER_SUITES.get(codepoint)
    return entry[1] if entry else None


# -- safe byte reader -------------------------------------------------------------------

class _Truncated(Exception):
    """A read ran past the end of the captured window.  Expected, not exceptional."""


class _Reader:
    """Bounds-checked big-endian reader over a byte window.

    Every accessor raises ``_Truncated`` rather than returning short data, so a caller can
    keep whatever it parsed before the cut without ever reading adjacent, unrelated bytes.
    """

    __slots__ = ("buf", "pos", "end")

    def __init__(self, buf: bytes, pos: int = 0, end: int | None = None) -> None:
        self.buf = buf
        self.pos = pos
        self.end = len(buf) if end is None else end

    @property
    def remaining(self) -> int:
        return self.end - self.pos

    def __bool__(self) -> bool:
        return self.remaining > 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > self.end:
            raise _Truncated(f"want {n}, have {self.remaining}")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self.take(2), "big")

    def u24(self) -> int:
        return int.from_bytes(self.take(3), "big")

    def vector(self, len_bytes: int) -> bytes:
        """A length-prefixed vector: ``len_bytes`` of length, then that many bytes."""
        n = int.from_bytes(self.take(len_bytes), "big")
        return self.take(n)

    def sub(self, n: int) -> "_Reader":
        """A reader over the next ``n`` bytes, advancing this one past them."""
        chunk = self.take(n)
        return _Reader(chunk)


def _u16_list(blob: bytes) -> list[int]:
    """Split a byte string into big-endian u16s, ignoring a trailing odd byte."""
    return [int.from_bytes(blob[i:i + 2], "big") for i in range(0, len(blob) - 1, 2)]


# -- parsed result ----------------------------------------------------------------------

@dataclass(slots=True)
class Handshake:
    """Everything Stage 3 needs, plus an honest account of what could not be read.

    ``negotiated_group`` is the single authoritative field: it is what the *server*
    selected, and the PQC verdict keys off it.  ``offered_groups`` is what the client was
    merely willing to do and is a strictly weaker signal.
    """

    # -- version / suite ---------------------------------------------------------------
    tls_version: str | None = None
    legacy_version: str | None = None
    cipher_suite: int | None = None          # selected (from ServerHello)
    offered_cipher_suites: list[int] = field(default_factory=list)

    # -- key exchange ------------------------------------------------------------------
    negotiated_group: int | None = None      # ServerHello key_share -- authoritative
    offered_groups: list[int] = field(default_factory=list)   # ClientHello supported_groups
    client_key_share_groups: list[int] = field(default_factory=list)  # groups client sent shares for
    hrr_selected_group: int | None = None    # HelloRetryRequest's demanded group

    # -- signatures --------------------------------------------------------------------
    sig_algs: list[int] = field(default_factory=list)
    sig_algs_cert: list[int] = field(default_factory=list)

    # -- context -----------------------------------------------------------------------
    sni: str | None = None
    alpn: list[str] = field(default_factory=list)
    ech: bool = False
    transport: str = "tls"                   # tls | quic

    # -- TLS 1.2 specifics -------------------------------------------------------------
    kex_family: str | None = None            # ECDHE / DHE / RSA / TLS1.3
    server_kex_group: int | None = None      # ServerKeyExchange named_curve (TLS 1.2 ECDHE)

    # -- provenance --------------------------------------------------------------------
    side: str = "none"                       # client | server | both | none
    saw_client_hello: bool = False
    saw_server_hello: bool = False
    saw_hello_retry_request: bool = False
    parse_ok: bool = False
    truncated: bool = False
    errors: list[str] = field(default_factory=list)

    # -- flow context, stamped by ``parse_handshake`` ----------------------------------
    # Carried on the Handshake so that ``pqc_verdict(hs)`` needs no second argument: the
    # tunnelled short-circuit and the "incomplete capture => not_observable, not
    # classical" rule are both properties of how the bytes were captured, not of the
    # bytes themselves, and the verdict is wrong without them.
    l7_hint: str | None = None
    handshake_incomplete: bool = False
    had_bytes: bool = False                  # the flow carried any handshake window at all

    # -- derived -----------------------------------------------------------------------
    @property
    def negotiated_group_name(self) -> str | None:
        return kem_registry.name_of(self.negotiated_group) if self.negotiated_group is not None else None

    @property
    def cipher_suite_name(self) -> str | None:
        return cipher_suite_name(self.cipher_suite)

    @property
    def is_tls13(self) -> bool:
        return self.tls_version == "1.3"

    @property
    def has_group(self) -> bool:
        """True when *some* key-exchange group was recovered, negotiated or offered."""
        return (self.negotiated_group is not None
                or self.server_kex_group is not None
                or bool(self.offered_groups))

    def note(self, msg: str) -> None:
        if msg not in self.errors:
            self.errors.append(msg)


# -- record iteration -------------------------------------------------------------------

@dataclass(slots=True)
class TlsRecord:
    content_type: int
    version: int
    payload: bytes
    complete: bool  # False when the window cut the body short


def iter_records(buf: bytes) -> Iterator[TlsRecord]:
    """Yield TLS records from a byte window, stopping at the first thing that is not one.

    A truncated final record is still yielded (``complete=False``) with the bytes we have:
    an 8 KB window routinely cuts a ServerHello carrying a 1216-byte PQC key share, and
    the fields we need usually sit before the cut.
    """
    pos, n = 0, len(buf)
    while pos + 5 <= n:
        ctype = buf[pos]
        version = int.from_bytes(buf[pos + 1:pos + 3], "big")
        length = int.from_bytes(buf[pos + 3:pos + 5], "big")
        # Structural sanity: a real record has a known type, a 0x03xx version and a
        # plausible length.  Anything else means we have wandered out of the record layer.
        if ctype not in (CT_CHANGE_CIPHER_SPEC, CT_ALERT, CT_HANDSHAKE, CT_APPLICATION_DATA):
            return
        if (version >> 8) != 0x03 or length == 0 or length > _MAX_RECORD:
            return
        body = buf[pos + 5:pos + 5 + length]
        yield TlsRecord(ctype, version, body, complete=len(body) == length)
        if len(body) < length:
            return
        pos += 5 + length


def _handshake_stream(buf: bytes) -> tuple[bytes, bool, bool]:
    """Concatenate the plaintext handshake records in ``buf``.

    Returns ``(stream, saw_any_record, truncated)``.  Only ``0x16`` records contribute --
    in TLS 1.3 everything after the ServerHello is wrapped in ``0x17`` records that are
    encrypted, and feeding those to the message parser would produce garbage messages.
    """
    chunks: list[bytes] = []
    saw_record = False
    truncated = False
    for rec in iter_records(buf):
        saw_record = True
        if not rec.complete:
            truncated = True
        if rec.content_type == CT_HANDSHAKE:
            chunks.append(rec.payload)
        elif rec.content_type == CT_APPLICATION_DATA:
            # TLS 1.3 encrypted flight begins here; nothing further is readable.
            break
    return b"".join(chunks), saw_record, truncated


def iter_handshake_messages(stream: bytes) -> Iterator[tuple[int, bytes, bool]]:
    """Yield ``(msg_type, body, complete)`` from a concatenated handshake byte stream."""
    r = _Reader(stream)
    while r.remaining >= 4:
        try:
            msg_type = r.u8()
            length = r.u24()
        except _Truncated:
            return
        body = r.buf[r.pos:r.pos + length]
        complete = len(body) == length
        r.pos += len(body)
        yield msg_type, body, complete
        if not complete:
            return


# -- extension parsing ------------------------------------------------------------------

def _parse_extensions(r: _Reader, hs: Handshake, *, is_server: bool, is_hrr: bool) -> None:
    """Walk the extension block, filling ``hs``.  Unknown extensions are skipped."""
    try:
        ext_block = _Reader(r.vector(2))
    except _Truncated:
        hs.truncated = True
        hs.note("extensions block truncated")
        return

    while ext_block.remaining >= 4:
        try:
            ext_type = ext_block.u16()
            ext_data = ext_block.vector(2)
        except _Truncated:
            hs.truncated = True
            hs.note("extension truncated")
            return
        try:
            _parse_one_extension(ext_type, ext_data, hs, is_server=is_server, is_hrr=is_hrr)
        except _Truncated:
            hs.truncated = True
            hs.note(f"extension {ext_type} truncated")
        except Exception as exc:  # never let one bad extension lose the rest
            hs.note(f"extension {ext_type} error: {exc}")


def _parse_one_extension(ext_type: int, data: bytes, hs: Handshake, *,
                         is_server: bool, is_hrr: bool) -> None:
    r = _Reader(data)

    if ext_type == EXT_SUPPORTED_VERSIONS:
        if is_server:
            # ServerHello: a single selected version.
            if len(data) >= 2:
                hs.tls_version = _VERSION_NAMES.get(int.from_bytes(data[:2], "big"))
        else:
            # ClientHello: a 1-byte-length list of offered versions; 1.3 wins if offered.
            versions = kem_registry.strip_grease(_u16_list(r.vector(1)))
            if versions and not is_server:
                best = max(versions) if versions else None
                # Only *raise* the version: a ClientHello offering 1.3 tells us the client
                # can do 1.3, but the server decides.  Recorded, overridden by ServerHello.
                if best is not None:
                    hs.tls_version = _VERSION_NAMES.get(best) or hs.tls_version

    elif ext_type == EXT_SUPPORTED_GROUPS:
        hs.offered_groups = kem_registry.strip_grease(_u16_list(r.vector(2)))

    elif ext_type == EXT_KEY_SHARE:
        if is_hrr:
            # HelloRetryRequest: a bare 2-byte selected_group, NOT a KeyShareEntry.  The
            # server is demanding a group the client did not send a share for -- it is a
            # selection, but not yet a completed exchange.
            if len(data) >= 2:
                hs.hrr_selected_group = int.from_bytes(data[:2], "big")
        elif is_server:
            # ServerHello: exactly one KeyShareEntry -- the authoritative selection.
            group = r.u16()
            if not kem_registry.is_grease(group):
                hs.negotiated_group = group
        else:
            # ClientHello: a list of KeyShareEntry the client actually generated keys for.
            shares = _Reader(r.vector(2))
            groups: list[int] = []
            while shares.remaining >= 4:
                g = shares.u16()
                shares.vector(2)  # skip the key_exchange blob
                if not kem_registry.is_grease(g):
                    groups.append(g)
            hs.client_key_share_groups = groups

    elif ext_type == EXT_SIGNATURE_ALGORITHMS:
        hs.sig_algs = kem_registry.strip_grease(_u16_list(r.vector(2)))

    elif ext_type == EXT_SIGNATURE_ALGORITHMS_CERT:
        hs.sig_algs_cert = kem_registry.strip_grease(_u16_list(r.vector(2)))

    elif ext_type == EXT_SERVER_NAME:
        # ServerNameList; only host_name (type 0) is defined.
        names = _Reader(r.vector(2))
        while names.remaining >= 3:
            name_type = names.u8()
            host = names.vector(2)
            if name_type == 0:
                # SNI is an ASCII DNS name (IDNs arrive already punycoded).  Decoded with
                # utf-8/replace rather than the `idna` codec, which raises on any name it
                # considers malformed -- and a hostile or merely odd SNI must not be able
                # to cost us the rest of the extension block.
                hs.sni = host.decode("utf-8", errors="replace") or None
                break

    elif ext_type == EXT_ALPN:
        protos = _Reader(r.vector(2))
        out: list[str] = []
        while protos.remaining >= 1:
            out.append(protos.vector(1).decode("ascii", errors="replace"))
        hs.alpn = [p for p in out if p]

    elif ext_type == EXT_ENCRYPTED_CLIENT_HELLO:
        # The inner ClientHello (real SNI, real groups) is encrypted to the provider's
        # key.  We record the fact and never attempt recovery -- see PHASE-2 sec. 9.
        hs.ech = True


# -- message parsing --------------------------------------------------------------------

def _parse_client_hello(body: bytes, hs: Handshake) -> None:
    r = _Reader(body)
    hs.legacy_version = _VERSION_NAMES.get(r.u16())
    if hs.tls_version is None:
        hs.tls_version = hs.legacy_version
    r.take(32)                      # random
    r.vector(1)                     # legacy_session_id
    hs.offered_cipher_suites = kem_registry.strip_grease(_u16_list(r.vector(2)))
    r.vector(1)                     # legacy_compression_methods
    if r.remaining >= 2:
        _parse_extensions(r, hs, is_server=False, is_hrr=False)
    hs.saw_client_hello = True


def _parse_server_hello(body: bytes, hs: Handshake) -> None:
    r = _Reader(body)
    hs.legacy_version = _VERSION_NAMES.get(r.u16())
    random = r.take(32)
    is_hrr = random == HELLO_RETRY_REQUEST_RANDOM
    if is_hrr:
        hs.saw_hello_retry_request = True
    r.vector(1)                     # legacy_session_id_echo
    suite = r.u16()
    if not kem_registry.is_grease(suite):
        hs.cipher_suite = suite
    r.u8()                          # legacy_compression_method
    # The ServerHello's supported_versions extension is what actually names TLS 1.3;
    # legacy_version is pinned at 0x0303 there.  Fall back to legacy for TLS 1.2 and below.
    if hs.tls_version is None or not hs.saw_client_hello:
        hs.tls_version = hs.legacy_version
    if r.remaining >= 2:
        _parse_extensions(r, hs, is_server=True, is_hrr=is_hrr)
    if hs.tls_version is None:
        hs.tls_version = hs.legacy_version
    hs.saw_server_hello = True


def _parse_server_key_exchange(body: bytes, hs: Handshake) -> None:
    """TLS 1.2 ECDHE ServerKeyExchange -- the only place 1.2 names its curve.

    Parsed only when the negotiated suite is ECDHE: for DHE the same bytes are ``dh_p``,
    and reading a prime's first two bytes as a named-curve codepoint would be nonsense.
    """
    if hs.kex_family != KEX_ECDHE:
        return
    r = _Reader(body)
    curve_type = r.u8()
    if curve_type != 3:  # 3 = named_curve; 1/2 are deprecated explicit curves
        hs.note(f"TLS1.2 ServerKeyExchange uses curve_type {curve_type}, not named_curve")
        return
    group = r.u16()
    if not kem_registry.is_grease(group):
        hs.server_kex_group = group


# -- entry points -----------------------------------------------------------------------

def parse_message_stream(stream: bytes, hs: Handshake) -> bool:
    """Parse a bare handshake-message stream (no record layer) into ``hs``.

    Shared with the QUIC path: QUIC has no TLS record layer at all -- CRYPTO frames carry
    handshake messages directly -- so once those frames are reassembled the message-level
    parsing is identical to TCP's.
    """
    found = False
    for msg_type, body, complete in iter_handshake_messages(stream):
        if not complete:
            hs.truncated = True
        try:
            if msg_type == HS_CLIENT_HELLO:
                _parse_client_hello(body, hs)
                found = True
            elif msg_type == HS_SERVER_HELLO:
                _parse_server_hello(body, hs)
                found = True
                # In TLS 1.3 nothing after a *real* ServerHello is readable; in 1.2 we
                # continue for the ServerKeyExchange.
                #
                # A HelloRetryRequest is emphatically not a stopping point: it reuses the
                # ServerHello message type, and the genuine ServerHello -- the one that
                # names the group actually used -- follows it in the same stream after the
                # client's second ClientHello.  Breaking here left every HRR handshake
                # reported from the retry request instead of the real selection.
                if hs.tls_version == "1.3" and not hs.saw_hello_retry_request:
                    break
                if hs.saw_hello_retry_request and hs.negotiated_group is not None:
                    break  # the post-retry ServerHello has now been read
            elif msg_type == HS_SERVER_KEY_EXCHANGE:
                # kex_family must be resolved from the selected suite first.
                if hs.kex_family is None:
                    hs.kex_family = kex_family(hs.cipher_suite)
                _parse_server_key_exchange(body, hs)
            elif msg_type == HS_NEW_SESSION_TICKET:
                # A window that opens here started after the handshake completed.
                hs.note("window opens on NewSessionTicket -- capture began mid-session")
        except _Truncated:
            hs.truncated = True
            hs.note(f"handshake message {msg_type} truncated")
        except Exception as exc:
            hs.note(f"handshake message {msg_type} error: {exc}")
    return found


def parse_tls_stream(buf: bytes, hs: Handshake) -> bool:
    """Parse one direction's TCP window into ``hs``.  Returns True if a Hello was found.

    The direction is not a parameter: which side we are looking at is settled by the
    handshake message type itself, and trusting a caller's claim over the bytes would
    misparse a capture whose client/server orientation the reassembler had to guess.
    """
    if not buf:
        return False
    stream, saw_record, truncated = _handshake_stream(buf)
    if truncated:
        hs.truncated = True
    if not saw_record:
        hs.note("no TLS record found in window")
        return False
    if not stream:
        hs.note("records present but no plaintext handshake (mid-session or encrypted)")
        return False
    return parse_message_stream(stream, hs)


def parse_tls_handshake(client_bytes: bytes | None, server_bytes: bytes | None) -> Handshake:
    """Parse both directions of a TLS handshake window into one ``Handshake``.

    The client side is parsed first so that a ClientHello's ``supported_versions`` is in
    place before the ServerHello overrides it with the server's actual selection.
    """
    hs = Handshake()
    got_client = parse_tls_stream(client_bytes or b"", hs)
    got_server = parse_tls_stream(server_bytes or b"", hs)

    if got_client and got_server:
        hs.side = "both"
    elif got_client:
        hs.side = "client"
    elif got_server:
        hs.side = "server"
    else:
        hs.side = "none"

    if hs.kex_family is None:
        hs.kex_family = kex_family(hs.cipher_suite)

    # A parse is "ok" when we recovered a Hello and at least one field worth acting on.
    hs.parse_ok = bool(
        (hs.saw_client_hello or hs.saw_server_hello)
        and (hs.tls_version is not None or hs.cipher_suite is not None or hs.has_group)
    )
    return hs
