"""Phase 1 tasks 1.6-1.7: TCP head reassembly and the coarse L7 hint."""

from __future__ import annotations

import pytest

from qsentinel.flows.handshake import (
    HeadBuffer,
    describe_records,
    is_known_quic_version,
    l7_hint,
    looks_like_tls,
    looks_like_tls_record,
    opens_handshake,
    quic_long_header,
    records_complete,
    starts_at_stream_start,
)
from tests.conftest import client_hello, quic_initial, server_hello, tls_record

ISN = 1000


def _tcp_buffer(window: int = 8192, with_syn: bool = True) -> HeadBuffer:
    hb = HeadBuffer(window=window, is_tcp=True)
    if with_syn:
        hb.note_syn(ISN)
    return hb


# --------------------------------------------------------------------------------------
# Head reassembly
# --------------------------------------------------------------------------------------


def test_in_order_segments_concatenate():
    hb = _tcp_buffer()
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"BBBB", ISN + 5)
    assert hb.data() == b"AAAABBBB"
    assert hb.complete and not hb.has_gap


def test_out_of_order_segments_are_ordered_by_sequence():
    hb = _tcp_buffer()
    hb.add(b"CCCC", ISN + 9)
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"BBBB", ISN + 5)
    assert hb.data() == b"AAAABBBBCCCC"
    assert not hb.has_gap


def test_retransmitted_segment_contributes_no_duplicate_bytes():
    """DoD 1.6: a retransmit must not appear twice in the reassembled window."""
    hb = _tcp_buffer()
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"BBBB", ISN + 5)
    assert hb.data() == b"AAAABBBB"
    assert hb.segments_seen == 4  # all four were seen, only two contributed


def test_overlapping_segment_keeps_only_the_new_tail():
    hb = _tcp_buffer()
    hb.add(b"AAAABBBB", ISN + 1)
    hb.add(b"BBBBCCCC", ISN + 5)  # first half overlaps
    assert hb.data() == b"AAAABBBBCCCC"


def test_segment_entirely_before_the_window_is_dropped():
    hb = _tcp_buffer()
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"XY", ISN - 3)  # wholly before the origin
    assert hb.data() == b"AAAA"


def test_gap_stops_the_buffer_and_is_reported():
    """A hole must be visible, not silently closed up: Phase 2 would misparse."""
    hb = _tcp_buffer()
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"CCCC", ISN + 101)  # 96-byte hole
    assert hb.data() == b"AAAA"
    assert hb.has_gap and not hb.complete


def test_a_filled_gap_drains_the_held_segments():
    hb = _tcp_buffer()
    hb.add(b"AAAA", ISN + 1)
    hb.add(b"CCCC", ISN + 9)
    assert hb.has_gap
    hb.add(b"BBBB", ISN + 5)
    assert hb.data() == b"AAAABBBBCCCC"
    assert not hb.has_gap


def test_buffer_stops_at_the_window():
    hb = _tcp_buffer(window=16)
    hb.add(b"A" * 100, ISN + 1)
    assert len(hb) == 16 and hb.full
    hb.add(b"B" * 10, ISN + 101)
    assert hb.data() == b"A" * 16


def test_sequence_number_wraparound_is_handled():
    """A stream that wraps past 2**32 must not be read as a 4 GB backwards jump."""
    base = (1 << 32) - 3
    hb = HeadBuffer(window=64, is_tcp=True)
    hb.note_syn(base - 1)
    hb.add(b"AAAA", base)
    hb.add(b"BBBB", (base + 4) % (1 << 32))  # wraps to 1
    assert hb.data() == b"AAAABBBB"


def test_missing_syn_still_captures_but_is_not_trusted_blindly():
    """Some dataset pcaps start mid-flow; the bytes are kept, the claim is not."""
    hb = HeadBuffer(window=64, is_tcp=True)
    hb.add(b"\x00" * 20, 5000)
    assert len(hb) == 20
    assert not hb.isn_known and not hb.trusted_start and not hb.complete


def test_missing_syn_is_trusted_when_the_window_opens_on_a_tls_record():
    """Every CSTNET capture is SYN-less but record-aligned; rejecting them loses 46k files."""
    hb = HeadBuffer(window=8192, is_tcp=True)
    hb.add(server_hello(200), 5000)
    assert not hb.isn_known
    assert hb.trusted_start and hb.complete


def test_udp_payloads_append_in_arrival_order():
    hb = HeadBuffer(window=64, is_tcp=False)
    hb.add(b"AAAA")
    hb.add(b"BBBB")
    assert hb.data() == b"AAAABBBB"


def test_out_of_order_holding_area_is_bounded():
    """A pathological stream must not grow the pending map without limit."""
    hb = _tcp_buffer(window=8192)
    for i in range(200):
        hb.add(b"X" * 4, ISN + 101 + i * 8)  # never fills the first hole
    assert hb.has_gap and hb.data() == b""
    assert len(hb._pending) <= 32


def test_multi_segment_server_hello_is_recovered_exactly():
    sh = server_hello(900)
    segments = {0: sh[0:400], 400: sh[400:700], 700: sh[700:]}
    hb = _tcp_buffer()
    for off in (400, 0, 400, 700):  # out of order, with the middle duplicated
        hb.add(segments[off], ISN + 1 + off)
    assert hb.data() == sh
    assert records_complete(hb.data())


# --------------------------------------------------------------------------------------
# L7 hint
# --------------------------------------------------------------------------------------


def test_tls_hint_from_handshake_record():
    assert l7_hint("tcp", client_hello(), server_hello(), 51000, 443) == "tls"


def test_tls_hint_needs_a_plausible_record_header():
    assert not looks_like_tls(b"\x16\x99\x01\x00\x10")     # bogus major version
    assert not looks_like_tls(b"\x16\x03\x01\xff\xff")     # length beyond 2**14
    assert looks_like_tls(b"\x16\x03\x01\x01\x00")


def test_quic_hint_from_long_header():
    assert l7_hint("udp", quic_initial(), b"", 55000, 443) == "quic"
    ok, version = quic_long_header(quic_initial())
    assert ok and version == 1


def test_quic_rejects_random_udp():
    assert quic_long_header(b"\xc3" + b"\xaa" * 40)[0] is False   # unknown version
    assert quic_long_header(b"\x40" + b"\x00" * 40)[0] is False   # short header
    assert quic_long_header(b"\xc3\x00\x00\x00\x01\xff" + b"\x00" * 40)[0] is False  # dcid > 20


@pytest.mark.parametrize("version,expected", [
    (0x00000001, True), (0x709A50C4, True), (0xFF00001D, True),
    (0x51303433, True), (0x12345678, False),
])
def test_known_quic_versions(version, expected):
    assert is_known_quic_version(version) is expected


def test_tunnel_hints():
    assert l7_hint("udp", b"\x00" * 60, b"", 1025, 500) == "tunnel_ike"
    assert l7_hint("udp", b"\x01\x00\x00\x00" + b"\x00" * 60, b"", 1025, 51820) == "tunnel_wireguard"
    assert l7_hint("udp", b"\x38" + b"\x00" * 40, b"", 1025, 1194) == "tunnel_openvpn"


def test_tor_hint_from_context_and_port():
    assert l7_hint("tcp", client_hello(), server_hello(), 51000, 9001) == "tor"
    assert l7_hint("tcp", client_hello(), server_hello(), 51000, 443,
                   context_hint="tor") == "tor"


def test_context_hint_never_overrides_wire_evidence():
    """A flow that plainly looks like QUIC keeps its wire hint even inside a Tor dataset."""
    assert l7_hint("udp", quic_initial(), b"", 55000, 443, context_hint="tor") == "quic"


def test_fallback_hints():
    assert l7_hint("tcp", b"", b"", 1234, 5678) == "opaque"
    assert l7_hint("tcp", b"hello there", b"", 1234, 5678) == "tcp"
    assert l7_hint("udp", b"\xaa" * 40, b"", 1234, 5678) == "udp"


def test_opens_handshake_rejects_mid_session_handshake_records():
    """NewSessionTicket looks like 0x16 0x03 but means the handshake already finished."""
    assert opens_handshake(server_hello())
    assert opens_handshake(client_hello())
    assert not opens_handshake(tls_record(0x16, b"\x04" + b"\x00" * 50))  # NewSessionTicket
    assert not opens_handshake(tls_record(0x17, b"\x00" * 50))            # app data


def test_starts_at_stream_start():
    assert starts_at_stream_start(tls_record(0x17, b"\x00" * 30), is_tcp=True)
    assert not starts_at_stream_start(b"\xff" * 30, is_tcp=True)
    assert starts_at_stream_start(quic_initial(), is_tcp=False)


def test_describe_records_walks_the_record_layer():
    buf = server_hello(100) + tls_record(0x14, b"\x01") + tls_record(0x17, b"\x00" * 40)
    recs = describe_records(buf)
    assert [(t, n) for t, _v, n in recs] == [(0x16, 100), (0x14, 1), (0x17, 40)]


def test_records_complete_detects_a_truncated_first_record():
    whole = server_hello(100)
    assert records_complete(whole)
    assert not records_complete(whole[:50])
    assert looks_like_tls_record(whole[:50])  # header is fine; the body is not all there
