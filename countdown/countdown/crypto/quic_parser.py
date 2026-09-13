"""QUIC Initial decryption -> ClientHello recovery (Phase 2, task 2.5).

QUIC encrypts its Initial packets, but **not secretly**: RFC 9001 derives the Initial keys
from a fixed, published salt and the client's original Destination Connection ID, which is
in the clear in the packet header.  Anyone who can see the packet can derive the keys.  The
encryption exists to stop middlebox ossification, not to hide the handshake -- so the
ClientHello inside is exactly as observable as a TCP one, just behind four HKDF steps.

Concretely, per Initial packet:

1. Read the long header -> version, DCID.
2. ``initial_secret = HKDF-Extract(initial_salt[version], original_dcid)``, then
   ``HKDF-Expand-Label`` to per-direction key / iv / hp.
3. Remove **header protection** (an AES-ECB mask over a ciphertext sample) to reveal the
   packet number and the low bits of the first byte -- these are needed before the AEAD
   can run, because they are part of its associated data.
4. AEAD-decrypt (AES-128-GCM) with ``nonce = iv XOR packet_number``.
5. Parse CRYPTO frames out of the plaintext and reassemble them by offset into the TLS
   handshake stream, which in QUIC carries handshake messages *directly* -- there is no
   TLS record layer.

Two limits worth stating plainly:

* **The server side needs the client side.**  Both directions' Initial keys derive from the
  client's *original* DCID.  A capture with only the server's Initial cannot be decrypted
  at all, and is reported as such rather than guessed at.
* **Version scope.**  v1 (RFC 9000) is fully supported and v2 (RFC 9369) shares the
  mechanism with different salt and labels, so both are implemented; anything else is
  logged and skipped rather than being decrypted with the wrong salt into noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from countdown.config import get_logger
from countdown.crypto.tls_parser import Handshake, parse_message_stream

log = get_logger(__name__)

# -- version parameters (RFC 9001 s5.2, RFC 9369 s3.3) -----------------------------------

QUIC_V1 = 0x00000001
QUIC_V2 = 0x6B3343CF

#: Per-version Initial salt and HKDF label prefix.  v2 deliberately changes both so that a
#: v1 middlebox cannot decrypt v2 Initials with hard-coded v1 parameters.
_VERSIONS: dict[int, tuple[bytes, bytes]] = {
    QUIC_V1: (bytes.fromhex("38762cf7f55934b34d179ae6a4c80cadccbb7f0a"), b"quic"),
    QUIC_V2: (bytes.fromhex("0dede3def700a6db819381be6e269dcbf9bd2ed9"), b"quicv2"),
}

#: Long-header packet-type bits (byte0 >> 4 & 0x3) for an Initial packet.  v2 renumbered
#: the types, so decoding a v2 Initial as v1 would silently pick the wrong packet.
_INITIAL_TYPE = {QUIC_V1: 0, QUIC_V2: 1}

_SAMPLE_LEN = 16
_AEAD_TAG_LEN = 16

# -- frame types we act on ---------------------------------------------------------------
FRAME_PADDING = 0x00
FRAME_PING = 0x01
FRAME_ACK = 0x02
FRAME_ACK_ECN = 0x03
FRAME_CRYPTO = 0x06
FRAME_CONNECTION_CLOSE = 0x1C
FRAME_CONNECTION_CLOSE_APP = 0x1D

#: Guard against a malformed length field turning into a multi-GB allocation.
_MAX_CRYPTO_STREAM = 1 << 20


# -- HKDF (RFC 8446 s7.1, as used by RFC 9001) -------------------------------------------

def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    import hmac
    import hashlib
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand_label(secret: bytes, label: bytes, length: int) -> bytes:
    """TLS 1.3 HKDF-Expand-Label with an empty context, which is all QUIC needs."""
    import hmac
    import hashlib

    full_label = b"tls13 " + label
    info = length.to_bytes(2, "big") + bytes([len(full_label)]) + full_label + b"\x00"
    out, t, counter = b"", b"", 1
    while len(out) < length:
        t = hmac.new(secret, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:length]


@dataclass(frozen=True, slots=True)
class InitialKeys:
    key: bytes
    iv: bytes
    hp: bytes


def derive_initial_keys(version: int, dcid: bytes, *, is_server: bool) -> InitialKeys | None:
    """Derive one direction's Initial key/iv/hp from the client's original DCID.

    Returns ``None`` for a version whose salt we do not know -- decrypting with the wrong
    salt yields noise that would parse as garbage frames rather than failing cleanly.
    """
    params = _VERSIONS.get(version)
    if params is None:
        return None
    salt, prefix = params
    initial_secret = _hkdf_extract(salt, dcid)
    label = b"server in" if is_server else b"client in"
    secret = _hkdf_expand_label(initial_secret, label, 32)
    return InitialKeys(
        key=_hkdf_expand_label(secret, prefix + b" key", 16),
        iv=_hkdf_expand_label(secret, prefix + b" iv", 12),
        hp=_hkdf_expand_label(secret, prefix + b" hp", 16),
    )


# -- varints (RFC 9000 s16) ---------------------------------------------------------------

def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    """``(value, new_pos)``.  The top two bits of the first byte give the encoded length."""
    if pos >= len(buf):
        raise ValueError("varint past end of buffer")
    prefix = buf[pos] >> 6
    n = 1 << prefix
    if pos + n > len(buf):
        raise ValueError("truncated varint")
    value = buf[pos] & 0x3F
    for i in range(1, n):
        value = (value << 8) | buf[pos + i]
    return value, pos + n


# -- long header --------------------------------------------------------------------------

@dataclass(slots=True)
class LongHeaderPacket:
    version: int
    dcid: bytes
    scid: bytes
    is_initial: bool
    pn_offset: int          # start of the (protected) packet number
    payload_end: int        # end of this packet within the buffer
    packet_end: int         # where the next coalesced packet starts
    header_start: int


def _parse_long_header(buf: bytes, pos: int) -> LongHeaderPacket | None:
    """Structurally decode one long-header packet starting at ``pos``.

    Returns ``None`` when ``pos`` is not a long-header packet (a short header, padding, or
    the end of the useful buffer), which is the signal to stop walking.
    """
    start = pos
    if pos + 7 > len(buf):
        return None
    first = buf[pos]
    if (first & 0xC0) != 0xC0:      # header form + fixed bit: not a long header
        return None
    version = int.from_bytes(buf[pos + 1:pos + 5], "big")
    pos += 5
    if pos >= len(buf):
        return None
    dcid_len = buf[pos]
    pos += 1
    if dcid_len > 20 or pos + dcid_len > len(buf):
        return None
    dcid = buf[pos:pos + dcid_len]
    pos += dcid_len
    if pos >= len(buf):
        return None
    scid_len = buf[pos]
    pos += 1
    if scid_len > 20 or pos + scid_len > len(buf):
        return None
    scid = buf[pos:pos + scid_len]
    pos += scid_len

    ptype = (first >> 4) & 0x03
    is_initial = _INITIAL_TYPE.get(version) == ptype

    if is_initial:
        try:
            token_len, pos = _varint(buf, pos)
        except ValueError:
            return None
        if token_len > len(buf) - pos:
            return None
        pos += token_len

    try:
        length, pos = _varint(buf, pos)
    except ValueError:
        return None
    if length <= 0 or length > len(buf):
        return None

    pn_offset = pos
    packet_end = pn_offset + length
    return LongHeaderPacket(
        version=version, dcid=dcid, scid=scid, is_initial=is_initial,
        pn_offset=pn_offset, payload_end=min(packet_end, len(buf)),
        packet_end=packet_end, header_start=start,
    )


# -- header protection + AEAD ---------------------------------------------------------------

def _remove_header_protection(buf: bytes, pkt: LongHeaderPacket,
                              keys: InitialKeys) -> tuple[bytes, int, int] | None:
    """Undo header protection.

    Returns ``(header, packet_number, pn_len)``, where ``header`` is the unprotected bytes
    that form the AEAD associated data.  The sample is taken at a *fixed* offset (4 bytes
    past the packet-number field) precisely because the real packet-number length is still
    hidden at this point.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    sample_off = pkt.pn_offset + 4
    if sample_off + _SAMPLE_LEN > len(buf):
        return None

    sample = buf[sample_off:sample_off + _SAMPLE_LEN]
    encryptor = Cipher(algorithms.AES(keys.hp), modes.ECB()).encryptor()
    mask = encryptor.update(sample) + encryptor.finalize()

    first = buf[pkt.header_start] ^ (mask[0] & 0x0F)   # long header: low 4 bits protected
    pn_len = (first & 0x03) + 1
    if pkt.pn_offset + pn_len > len(buf):
        return None

    pn_bytes = bytes(
        buf[pkt.pn_offset + i] ^ mask[1 + i] for i in range(pn_len)
    )
    packet_number = int.from_bytes(pn_bytes, "big")

    header = bytearray(buf[pkt.header_start:pkt.pn_offset + pn_len])
    header[0] = first
    header[pkt.pn_offset - pkt.header_start:] = pn_bytes
    return bytes(header), packet_number, pn_len


def _decrypt_payload(buf: bytes, pkt: LongHeaderPacket, keys: InitialKeys,
                     header: bytes, packet_number: int, pn_len: int) -> bytes | None:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    body_start = pkt.pn_offset + pn_len
    body_end = min(pkt.packet_end, len(buf))
    if body_end - body_start <= _AEAD_TAG_LEN:
        return None
    ciphertext = buf[body_start:body_end]

    # nonce = iv XOR the packet number, left-padded to the iv length.
    pn_padded = packet_number.to_bytes(len(keys.iv), "big")
    nonce = bytes(a ^ b for a, b in zip(keys.iv, pn_padded))
    try:
        return AESGCM(keys.key).decrypt(nonce, ciphertext, header)
    except InvalidTag:
        # Truncated by the capture window, a Retry-changed DCID, or simply not an Initial.
        return None


# -- frame parsing --------------------------------------------------------------------------

def _collect_crypto_frames(plaintext: bytes, chunks: dict[int, bytes]) -> None:
    """Extract CRYPTO frames into ``chunks`` keyed by stream offset.

    Only the frame types that legitimately share an Initial packet are decoded; anything
    else stops the walk, since frame parsing is positional and a misread type would
    desynchronise every frame after it.
    """
    pos, n = 0, len(plaintext)
    while pos < n:
        try:
            ftype, pos = _varint(plaintext, pos)
        except ValueError:
            return
        if ftype == FRAME_PADDING:
            # Runs of padding are common (Initials are padded to 1200 bytes); skip fast.
            while pos < n and plaintext[pos] == 0x00:
                pos += 1
            continue
        if ftype == FRAME_PING:
            continue
        if ftype in (FRAME_ACK, FRAME_ACK_ECN):
            try:
                _largest, pos = _varint(plaintext, pos)
                _delay, pos = _varint(plaintext, pos)
                range_count, pos = _varint(plaintext, pos)
                _first_range, pos = _varint(plaintext, pos)
                for _ in range(min(range_count, 1024)):
                    _gap, pos = _varint(plaintext, pos)
                    _len, pos = _varint(plaintext, pos)
                if ftype == FRAME_ACK_ECN:
                    for _ in range(3):
                        _ect, pos = _varint(plaintext, pos)
            except ValueError:
                return
            continue
        if ftype == FRAME_CRYPTO:
            try:
                offset, pos = _varint(plaintext, pos)
                length, pos = _varint(plaintext, pos)
            except ValueError:
                return
            if length < 0 or pos + length > n or offset > _MAX_CRYPTO_STREAM:
                return
            chunks[offset] = plaintext[pos:pos + length]
            pos += length
            continue
        if ftype in (FRAME_CONNECTION_CLOSE, FRAME_CONNECTION_CLOSE_APP):
            return
        # Any other frame type in an Initial: we cannot know its length, so stop rather
        # than risk reading a bogus CRYPTO frame out of misaligned bytes.
        return


def _reassemble(chunks: dict[int, bytes]) -> bytes:
    """Splice offset-keyed CRYPTO chunks into a contiguous prefix, stopping at the first gap."""
    out = bytearray()
    for offset in sorted(chunks):
        chunk = chunks[offset]
        if offset > len(out):
            break                      # hole: everything past it is unusable
        end_of_new = offset + len(chunk)
        if end_of_new <= len(out):
            continue                   # wholly duplicate
        out += chunk[len(out) - offset:]
    return bytes(out)


# -- direction decode -----------------------------------------------------------------------

def _walk_initials(buf: bytes, keys_for: "callable", hs: Handshake) -> dict[int, bytes]:
    """Walk coalesced/consecutive QUIC packets, decrypting every Initial we can."""
    chunks: dict[int, bytes] = {}
    pos, guard = 0, 0
    while pos < len(buf) and guard < 64:
        guard += 1
        pkt = _parse_long_header(buf, pos)
        if pkt is None:
            break
        if not pkt.is_initial:
            # Handshake/0-RTT packets use keys derived from the TLS secrets, which a
            # passive observer genuinely cannot compute.  Skip past and keep looking.
            if pkt.packet_end <= pos:
                break
            pos = pkt.packet_end
            continue

        keys = keys_for(pkt.version, pkt.dcid)
        if keys is None:
            hs.note(f"unsupported QUIC version 0x{pkt.version:08X}; Initial not decrypted")
            if pkt.packet_end <= pos:
                break
            pos = pkt.packet_end
            continue

        unprotected = _remove_header_protection(buf, pkt, keys)
        if unprotected is None:
            hs.truncated = True
            hs.note("QUIC Initial too short to remove header protection")
            break
        header, packet_number, pn_len = unprotected
        plaintext = _decrypt_payload(buf, pkt, keys, header, packet_number, pn_len)
        if plaintext is None:
            hs.note("QUIC Initial AEAD authentication failed "
                    "(window truncated, or a Retry changed the connection ID)")
            if pkt.packet_end <= pos:
                break
            pos = pkt.packet_end
            continue

        _collect_crypto_frames(plaintext, chunks)
        if pkt.packet_end <= pos:
            break
        pos = pkt.packet_end
    return chunks


def _original_dcid(buf: bytes) -> tuple[int, bytes] | None:
    """``(version, dcid)`` from the client's first Initial -- the root of every Initial key."""
    pkt = _parse_long_header(buf, 0)
    if pkt is None or not pkt.is_initial:
        return None
    return pkt.version, pkt.dcid


def parse_quic_handshake(client_bytes: bytes | None, server_bytes: bytes | None) -> Handshake:
    """Recover the TLS handshake from a QUIC flow's Initial packets."""
    hs = Handshake()
    hs.transport = "quic"
    client = client_bytes or b""
    server = server_bytes or b""

    origin = _original_dcid(client)
    if origin is None:
        # No client Initial: the server's keys derive from the *client's* original DCID,
        # which we do not have.  This is unrecoverable, not merely unparsed.
        hs.note("no client QUIC Initial in the window; Initial keys cannot be derived "
                "(they are rooted in the client's original destination connection ID)")
        hs.parse_ok = False
        return hs

    version, odcid = origin
    if version not in _VERSIONS:
        hs.note(f"QUIC version 0x{version:08X} is not supported (v1/v2 only)")
        hs.parse_ok = False
        return hs

    client_chunks = _walk_initials(
        client, lambda v, _d: derive_initial_keys(v, odcid, is_server=False), hs)
    # The server's Initial carries a different DCID (the client's SCID) but its keys are
    # still derived from the client's ORIGINAL DCID -- so pass odcid, not the packet's own.
    server_chunks = _walk_initials(
        server, lambda v, _d: derive_initial_keys(v, odcid, is_server=True), hs)

    got_client = got_server = False
    client_stream = _reassemble(client_chunks)
    if client_stream:
        got_client = parse_message_stream(client_stream, hs)
    server_stream = _reassemble(server_chunks)
    if server_stream:
        got_server = parse_message_stream(server_stream, hs)

    if got_client and got_server:
        hs.side = "both"
    elif got_client:
        hs.side = "client"
    elif got_server:
        hs.side = "server"
    else:
        hs.side = "none"
        hs.note("QUIC Initials decrypted but no ClientHello/ServerHello recovered")

    # QUIC mandates TLS 1.3; a QUIC flow with no supported_versions is still 1.3.
    if (got_client or got_server) and hs.tls_version is None:
        hs.tls_version = "1.3"

    hs.parse_ok = bool(
        (hs.saw_client_hello or hs.saw_server_hello)
        and (hs.tls_version is not None or hs.cipher_suite is not None or hs.has_group)
    )
    return hs
