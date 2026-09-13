"""Phase 2, tasks 2.2-2.3 -- the TLS record/handshake parser.

Golden vectors are built here rather than checked in as opaque blobs so that each one
states, in code, exactly which wire construct it is exercising.  The end-to-end check
against *real* captured bytes lives in ``test_pqc_verdict.py``; these are the unit-level
guarantees, and they concentrate on the constructs that silently produce a WRONG group
rather than an obviously failed parse:

* a HelloRetryRequest, whose key_share holds a bare group id instead of a KeyShareEntry
* GREASE, which is syntactically indistinguishable from a real codepoint
* truncation, which is the normal case at an 8 KB window rather than an error
"""

from __future__ import annotations

import struct

import pytest

from countdown.crypto.tls_parser import (
    HELLO_RETRY_REQUEST_RANDOM,
    Handshake,
    cipher_suite_name,
    iter_records,
    kex_family,
    parse_tls_handshake,
)

# -- builders ------------------------------------------------------------------------------


def rec(body: bytes, ctype: int = 0x16, version: bytes = b"\x03\x03") -> bytes:
    return bytes([ctype]) + version + len(body).to_bytes(2, "big") + body


def msg(msg_type: int, body: bytes) -> bytes:
    return bytes([msg_type]) + len(body).to_bytes(3, "big") + body


def ext(ext_type: int, body: bytes) -> bytes:
    return struct.pack(">HH", ext_type, len(body)) + body


def vec(body: bytes, n: int) -> bytes:
    return len(body).to_bytes(n, "big") + body


def u16s(values) -> bytes:
    return b"".join(struct.pack(">H", v) for v in values)


def client_hello(groups=(0x11EC, 0x001D), key_shares=((0x001D, b"\x11" * 32),),
                 suites=(0x1301, 0x1302), sni=b"example.com", sig_algs=(0x0403, 0x0804),
                 alpn=(b"h2",), ech=False, extra_exts=b"") -> bytes:
    exts = b""
    if sni:
        exts += ext(0, vec(b"\x00" + vec(sni, 2), 2))
    exts += ext(10, vec(u16s(groups), 2))
    exts += ext(13, vec(u16s(sig_algs), 2))
    exts += ext(43, vec(struct.pack(">H", 0x0304), 1))
    ks = b"".join(struct.pack(">H", g) + vec(k, 2) for g, k in key_shares)
    exts += ext(51, vec(ks, 2))
    if alpn:
        exts += ext(16, vec(b"".join(vec(p, 1) for p in alpn), 2))
    if ech:
        exts += ext(0xFE0D, b"\x00\x01\x02\x03")
    exts += extra_exts
    body = (struct.pack(">H", 0x0303) + b"\xAA" * 32 + vec(b"\xBB" * 32, 1)
            + vec(u16s(suites), 2) + vec(b"\x00", 1) + vec(exts, 2))
    return rec(msg(0x01, body), version=b"\x03\x01")


def server_hello(group=0x001D, suite=0x1301, tls13=True, hrr=False,
                 key_share=True) -> bytes:
    exts = b""
    if tls13:
        exts += ext(43, struct.pack(">H", 0x0304))
    if key_share:
        if hrr:
            exts += ext(51, struct.pack(">H", group))          # bare selected_group
        else:
            exts += ext(51, struct.pack(">H", group) + vec(b"\xCC" * 32, 2))
    random = HELLO_RETRY_REQUEST_RANDOM if hrr else b"\xDD" * 32
    body = (struct.pack(">H", 0x0303) + random + vec(b"\xBB" * 32, 1)
            + struct.pack(">H", suite) + b"\x00" + vec(exts, 2))
    return rec(msg(0x02, body))


# -- record layer ----------------------------------------------------------------------------

def test_iter_records_walks_a_multi_record_window():
    buf = rec(b"\x01\x02\x03") + rec(b"\x04" * 10, ctype=0x17) + rec(b"\x05", ctype=0x15)
    got = list(iter_records(buf))
    assert [r.content_type for r in got] == [0x16, 0x17, 0x15]
    assert all(r.complete for r in got)


def test_iter_records_yields_a_truncated_tail_then_stops():
    full = rec(b"\xAA" * 100)
    got = list(iter_records(full[:60]))
    assert len(got) == 1
    assert not got[0].complete
    assert len(got[0].payload) == 55


def test_iter_records_stops_on_non_tls_bytes():
    assert list(iter_records(b"GET / HTTP/1.1\r\n\r\n")) == []
    assert list(iter_records(b"")) == []
    # A valid record followed by junk: keep the first, stop cleanly.
    assert len(list(iter_records(rec(b"\x01\x02") + b"\xFF" * 40))) == 1


# -- ClientHello ------------------------------------------------------------------------------

def test_client_hello_fields():
    hs = parse_tls_handshake(client_hello(), None)
    assert hs.parse_ok and hs.saw_client_hello and hs.side == "client"
    assert hs.tls_version == "1.3"
    assert hs.legacy_version == "1.2"
    assert hs.offered_groups == [0x11EC, 0x001D]
    assert hs.client_key_share_groups == [0x001D]
    assert hs.offered_cipher_suites == [0x1301, 0x1302]
    assert hs.sni == "example.com"
    assert hs.alpn == ["h2"]
    assert hs.sig_algs == [0x0403, 0x0804]
    assert not hs.ech


def test_ech_is_detected_not_decrypted():
    hs = parse_tls_handshake(client_hello(ech=True), None)
    assert hs.ech is True


# -- ServerHello: the authoritative field --------------------------------------------------------

def test_server_hello_negotiated_group_is_extracted():
    hs = parse_tls_handshake(None, server_hello(group=0x11EC, suite=0x1302))
    assert hs.saw_server_hello and hs.side == "server"
    assert hs.negotiated_group == 0x11EC
    assert hs.negotiated_group_name == "X25519MLKEM768"
    assert hs.cipher_suite == 0x1302
    assert hs.tls_version == "1.3"


def test_both_sides_merge_and_server_wins():
    """The client offering PQC must not overwrite the server's classical selection."""
    hs = parse_tls_handshake(client_hello(groups=(0x11EC, 0x001D)),
                             server_hello(group=0x001D))
    assert hs.side == "both"
    assert hs.offered_groups == [0x11EC, 0x001D]
    assert hs.negotiated_group == 0x001D          # what actually happened


def test_hello_retry_request_key_share_is_a_bare_group():
    """HRR reuses the ServerHello type; its key_share is a group id, not a KeyShareEntry.

    Parsed as a normal ServerHello, the 2-byte length prefix of the (absent) key would be
    read as the group -- a wrong answer that looks perfectly well-formed.
    """
    hs = parse_tls_handshake(None, server_hello(group=0x0017, hrr=True))
    assert hs.saw_hello_retry_request
    assert hs.hrr_selected_group == 0x0017
    assert hs.negotiated_group is None            # no exchange has completed yet


def test_hello_retry_request_does_not_hide_the_real_server_hello():
    """The genuine ServerHello follows the retry in the same stream and must win.

    Regression: the parser used to stop at the first ServerHello-typed message, so every
    HRR handshake reported the retry's demanded group and no negotiated group at all --
    5% of the CSTNET corpus.
    """
    stream = server_hello(group=0x0017, hrr=True) + server_hello(group=0x0017)
    hs = parse_tls_handshake(None, stream)
    assert hs.saw_hello_retry_request
    assert hs.hrr_selected_group == 0x0017
    assert hs.negotiated_group == 0x0017          # recovered from the real ServerHello


# -- GREASE (RFC 8701) ----------------------------------------------------------------------------

def test_grease_is_dropped_from_every_list():
    hs = parse_tls_handshake(
        client_hello(groups=(0x0A0A, 0x11EC, 0xFAFA, 0x001D),
                     suites=(0x1A1A, 0x1301), sig_algs=(0x2A2A, 0x0403),
                     key_shares=((0x3A3A, b"\x00"), (0x001D, b"\x11" * 32))),
        None)
    assert hs.offered_groups == [0x11EC, 0x001D]
    assert hs.offered_cipher_suites == [0x1301]
    assert hs.sig_algs == [0x0403]
    assert hs.client_key_share_groups == [0x001D]


def test_grease_never_becomes_the_negotiated_group():
    hs = parse_tls_handshake(None, server_hello(group=0xDADA))
    assert hs.negotiated_group is None


# -- truncation is normal ---------------------------------------------------------------------------

@pytest.mark.parametrize("cut", [10, 40, 80, 150, 300])
def test_truncated_client_hello_never_raises(cut):
    hs = parse_tls_handshake(client_hello()[:cut], None)
    assert isinstance(hs, Handshake)   # partial fields are fine; an exception is not


def test_truncated_server_hello_keeps_fields_before_the_cut():
    full = server_hello(group=0x11EC, suite=0x1302)
    hs = parse_tls_handshake(None, full[:20])   # cuts inside the random
    assert hs.negotiated_group is None
    assert hs.truncated or not hs.parse_ok


def test_empty_and_garbage_input():
    for buf in (b"", b"\x00", b"\xFF" * 500, b"GET / HTTP/1.1\r\n"):
        hs = parse_tls_handshake(buf, buf)
        assert not hs.parse_ok
        assert hs.side == "none"
        assert hs.negotiated_group is None


def test_tls13_encrypted_flight_is_not_parsed_as_handshake():
    """Post-ServerHello 0x17 records are encrypted; feeding them to the message parser
    would manufacture handshake messages out of ciphertext."""
    buf = server_hello(group=0x001D) + rec(b"\x01" + b"\xEE" * 200, ctype=0x17)
    hs = parse_tls_handshake(None, buf)
    assert hs.negotiated_group == 0x001D
    assert not hs.saw_client_hello


def test_window_opening_mid_session_is_reported():
    """A NewSessionTicket at offset 0 means the capture began after the handshake."""
    hs = parse_tls_handshake(None, rec(msg(0x04, b"\x00" * 40)))
    assert not hs.saw_server_hello
    assert any("mid-session" in e for e in hs.errors)


# -- TLS 1.2 fallback ---------------------------------------------------------------------------------

def test_tls12_server_key_exchange_named_curve():
    sh = rec(msg(0x02, struct.pack(">H", 0x0303) + b"\xDD" * 32 + vec(b"", 1)
                 + struct.pack(">H", 0xC02F) + b"\x00" + vec(b"", 2)))
    ske = rec(msg(0x0C, b"\x03" + struct.pack(">H", 0x0017) + vec(b"\x04" + b"\x11" * 64, 1)))
    hs = parse_tls_handshake(None, sh + ske)
    assert hs.tls_version == "1.2"
    assert hs.kex_family == "ECDHE"
    assert hs.server_kex_group == 0x0017


def test_tls12_dhe_server_key_exchange_is_not_read_as_a_curve():
    """For DHE the same bytes are dh_p; reading them as a named curve invents a group."""
    sh = rec(msg(0x02, struct.pack(">H", 0x0303) + b"\xDD" * 32 + vec(b"", 1)
                 + struct.pack(">H", 0x009E) + b"\x00" + vec(b"", 2)))
    ske = rec(msg(0x0C, b"\x03" + struct.pack(">H", 0x0017) + b"\x00" * 40))
    hs = parse_tls_handshake(None, sh + ske)
    assert hs.kex_family == "DHE"
    assert hs.server_kex_group is None


@pytest.mark.parametrize("suite,family", [
    (0x1301, "TLS1.3"), (0xC02F, "ECDHE"), (0xC030, "ECDHE"),
    (0x009E, "DHE"), (0x009C, "RSA"), (0xCCA9, "ECDHE"),
])
def test_kex_family_from_cipher_suite(suite, family):
    assert kex_family(suite) == family


def test_unknown_cipher_suite_yields_no_family():
    assert kex_family(0xABCD) is None
    assert "unknown" in cipher_suite_name(0xABCD)
