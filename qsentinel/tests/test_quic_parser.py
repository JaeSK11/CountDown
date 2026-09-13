"""Phase 2, task 2.5 -- QUIC Initial decryption.

The key schedule is checked against **RFC 9001 Appendix A**, not against our own encryptor.
That matters: a round-trip test (encrypt with our keys, decrypt with our keys) passes even
if every derived secret is wrong, so it would validate nothing about interoperability.  The
RFC's published vectors are an independent oracle for the HKDF chain, the header-protection
mask and the AEAD, and Appendix A.2 additionally carries a real ClientHello, so the whole
path -- keys, unprotect, decrypt, CRYPTO reassembly, TLS parse -- is exercised end to end
on bytes we did not produce.
"""

from __future__ import annotations

from pathlib import Path

from qsentinel.crypto.quic_parser import (
    QUIC_V1,
    _collect_crypto_frames,
    _parse_long_header,
    _reassemble,
    _varint,
    derive_initial_keys,
    parse_quic_handshake,
)

# -- RFC 9001 Appendix A -------------------------------------------------------------------

#: A.1 -- the client-chosen Destination Connection ID every value below derives from.
RFC9001_DCID = bytes.fromhex("8394c8f03e515708")

RFC9001_CLIENT_KEY = bytes.fromhex("1f369613dd76d5467730efcbe3b1a22d")
RFC9001_CLIENT_IV = bytes.fromhex("fa044b2f42a3fd3b46fb255c")
RFC9001_CLIENT_HP = bytes.fromhex("9f50449e04a0e810283a1e9933adedd2")

RFC9001_SERVER_KEY = bytes.fromhex("cf3a5331653c364c88f0f379b6067e37")
RFC9001_SERVER_IV = bytes.fromhex("0ac1493ca1905853b0bba03e")
RFC9001_SERVER_HP = bytes.fromhex("c206b8d9b9f0f37644430b490eeaa314")

GOLDEN = Path(__file__).parent / "golden" / "quic"


def _golden(name: str) -> bytes:
    """One of the RFC 9001 Appendix A packets, verbatim (see golden/quic/README.md)."""
    return bytes.fromhex((GOLDEN / name).read_text().strip())


def rfc9001_client_initial() -> bytes:
    """A.2: the complete 1200-byte protected client Initial, carrying a real ClientHello."""
    return _golden("rfc9001_a2_client_initial.hex")


def rfc9001_server_initial() -> bytes:
    """A.3: the complete protected server Initial, carrying the matching ServerHello."""
    return _golden("rfc9001_a3_server_initial.hex")


# -- key schedule ---------------------------------------------------------------------------

def test_client_initial_keys_match_rfc9001():
    keys = derive_initial_keys(QUIC_V1, RFC9001_DCID, is_server=False)
    assert keys is not None
    assert keys.key == RFC9001_CLIENT_KEY
    assert keys.iv == RFC9001_CLIENT_IV
    assert keys.hp == RFC9001_CLIENT_HP


def test_server_initial_keys_match_rfc9001():
    keys = derive_initial_keys(QUIC_V1, RFC9001_DCID, is_server=True)
    assert keys is not None
    assert keys.key == RFC9001_SERVER_KEY
    assert keys.iv == RFC9001_SERVER_IV
    assert keys.hp == RFC9001_SERVER_HP


def test_client_and_server_secrets_differ():
    c = derive_initial_keys(QUIC_V1, RFC9001_DCID, is_server=False)
    s = derive_initial_keys(QUIC_V1, RFC9001_DCID, is_server=True)
    assert c.key != s.key and c.iv != s.iv and c.hp != s.hp


def test_v2_uses_a_different_salt_and_labels():
    """v2 deliberately changes both, so v1 parameters must not silently 'work'."""
    v1 = derive_initial_keys(0x00000001, RFC9001_DCID, is_server=False)
    v2 = derive_initial_keys(0x6B3343CF, RFC9001_DCID, is_server=False)
    assert v2 is not None
    assert v1.key != v2.key


def test_unknown_version_yields_no_keys():
    """Decrypting with the wrong salt produces noise that parses as garbage frames."""
    assert derive_initial_keys(0xDEADBEEF, RFC9001_DCID, is_server=False) is None
    assert derive_initial_keys(0xFF00001D, RFC9001_DCID, is_server=False) is None


# -- end-to-end on the RFC's own packet --------------------------------------------------------

def test_long_header_of_the_client_initial_decodes():
    pkt = _parse_long_header(rfc9001_client_initial(), 0)
    assert pkt is not None
    assert pkt.version == QUIC_V1
    assert pkt.is_initial
    assert pkt.dcid == RFC9001_DCID
    assert pkt.packet_end == len(rfc9001_client_initial())


def test_full_handshake_recovered_from_the_rfc_vectors():
    """The whole path on bytes we did not produce: derive keys, strip header protection,
    AEAD-decrypt, reassemble CRYPTO frames, parse the TLS messages out of both directions.

    Every asserted value is stated in the RFC's own plaintext dumps in A.2 and A.3.
    """
    hs = parse_quic_handshake(rfc9001_client_initial(), rfc9001_server_initial())
    assert hs.parse_ok
    assert hs.transport == "quic"
    assert hs.side == "both"
    assert hs.saw_client_hello and hs.saw_server_hello
    assert hs.tls_version == "1.3"
    assert hs.sni == "example.com"
    assert hs.offered_groups == [0x001D, 0x0017, 0x0018]
    assert hs.negotiated_group == 0x001D           # A.3's ServerHello key_share
    assert hs.negotiated_group_name == "x25519"
    assert hs.cipher_suite == 0x1301
    assert not hs.errors


def test_client_initial_alone_yields_the_client_hello():
    hs = parse_quic_handshake(rfc9001_client_initial(), b"")
    assert hs.saw_client_hello
    assert hs.side == "client"
    assert hs.sni == "example.com"
    assert hs.negotiated_group is None             # no ServerHello: nothing was selected


def test_truncated_initial_fails_closed_not_open():
    """A packet the capture window cut must yield no handshake, never a fabricated one:
    the AEAD tag covers the whole payload, so a short read cannot authenticate."""
    hs = parse_quic_handshake(rfc9001_client_initial()[:400], b"")
    assert not hs.parse_ok
    assert hs.negotiated_group is None
    assert hs.sni is None
    assert hs.errors


def test_wrong_dcid_does_not_decrypt():
    """Flipping one byte of the connection ID changes every derived secret."""
    pkt = bytearray(rfc9001_client_initial())
    pkt[6] ^= 0xFF                                  # first byte of the DCID
    hs = parse_quic_handshake(bytes(pkt), b"")
    assert not hs.parse_ok
    assert hs.sni is None


# -- structural decoding -------------------------------------------------------------------------

def test_varint_all_four_encodings():
    assert _varint(bytes.fromhex("25"), 0) == (37, 1)
    assert _varint(bytes.fromhex("7bbd"), 0) == (15293, 2)
    assert _varint(bytes.fromhex("9d7f3e7d"), 0) == (494878333, 4)
    assert _varint(bytes.fromhex("c2197c5eff14e88c"), 0) == (151288809941952652, 8)


def test_varint_rejects_truncation():
    import pytest
    with pytest.raises(ValueError):
        _varint(b"\x40", 0)          # 2-byte form with only 1 byte present
    with pytest.raises(ValueError):
        _varint(b"", 0)


def test_short_header_is_not_a_long_header():
    assert _parse_long_header(b"\x40" + b"\x00" * 20, 0) is None
    assert _parse_long_header(b"", 0) is None
    assert _parse_long_header(b"\xc0", 0) is None


def test_oversized_connection_id_is_rejected():
    """CIDs are capped at 20 bytes; a larger length means this is not a QUIC header."""
    buf = bytes([0xC3]) + (1).to_bytes(4, "big") + bytes([21]) + b"\x00" * 40
    assert _parse_long_header(buf, 0) is None


# -- CRYPTO frame handling ---------------------------------------------------------------------------

def test_crypto_frames_are_collected_by_offset():
    # frame type 0x06, offset 0, length 4, then data
    plaintext = b"\x06\x00\x04ABCD" + b"\x06\x04\x04EFGH"
    chunks: dict[int, bytes] = {}
    _collect_crypto_frames(plaintext, chunks)
    assert chunks == {0: b"ABCD", 4: b"EFGH"}
    assert _reassemble(chunks) == b"ABCDEFGH"


def test_padding_and_ping_are_skipped():
    plaintext = b"\x00" * 20 + b"\x01" + b"\x06\x00\x03XYZ"
    chunks: dict[int, bytes] = {}
    _collect_crypto_frames(plaintext, chunks)
    assert chunks == {0: b"XYZ"}


def test_reassembly_stops_at_a_gap():
    """A hole means everything past it is unusable -- splicing across it would corrupt
    the handshake stream and produce nonsense messages."""
    assert _reassemble({0: b"AAAA", 8: b"CCCC"}) == b"AAAA"
    assert _reassemble({4: b"BBBB"}) == b""            # nothing at offset 0
    assert _reassemble({0: b"AAAA", 4: b"BBBB"}) == b"AAAABBBB"


def test_duplicate_and_overlapping_chunks():
    assert _reassemble({0: b"AAAA", 2: b"AABB"}) == b"AAAABB"
    assert _reassemble({0: b"AAAABB", 0: b"AAAABB"}) == b"AAAABB"


def test_unknown_frame_type_stops_the_walk():
    """Frame parsing is positional; guessing past an unknown type desynchronises it."""
    chunks: dict[int, bytes] = {}
    _collect_crypto_frames(b"\x1f\x00\x00" + b"\x06\x00\x03XYZ", chunks)
    assert chunks == {}


# -- server side without a client Initial ----------------------------------------------------------

def test_server_only_capture_is_unrecoverable_and_says_so():
    """Both directions' keys root in the client's original DCID.  Without the client's
    Initial the server's cannot be decrypted at all -- that must be reported, not guessed."""
    server_only = bytes([0xC3]) + (1).to_bytes(4, "big") + b"\x08" + b"\x11" * 8 \
        + b"\x04" + b"\x22" * 4 + b"\x00" + b"\x44\x00" + b"\x00" * 100
    hs = parse_quic_handshake(b"", server_only)
    assert not hs.parse_ok
    assert hs.negotiated_group is None
    assert any("client" in e.lower() for e in hs.errors)


def test_garbage_never_raises():
    for buf in (b"", b"\x00" * 50, b"\xFF" * 200, b"GET / HTTP/1.1\r\n"):
        hs = parse_quic_handshake(buf, buf)
        assert not hs.parse_ok
        assert hs.transport == "quic"
