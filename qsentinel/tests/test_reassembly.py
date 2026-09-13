"""Phase 1 task 1.5: the flow table, expiry, and the metadata Phase 2 consumes."""

from __future__ import annotations

import numpy as np
import pytest

from qsentinel.flows import Reassembler
from qsentinel.flows.decode import ParseStats
from qsentinel.flows.reassembly import (
    EXPIRY_END_OF_CAPTURE,
    EXPIRY_FIN,
    EXPIRY_IDLE,
    EXPIRY_RST,
    EXPIRY_TABLE_FULL,
    canonical_key,
)
from qsentinel.schema import BACKWARD, FORWARD
from qsentinel.sources import DirectorySource, PcapSource
from tests.conftest import (
    CLIENT_IP,
    CLIENT_PORT,
    SERVER_IP,
    SERVER_PORT,
    payload_frame,
    write_pcap,
)


def run(pcap, **kw):
    return list(Reassembler(**kw).run(PcapSource(pcap)))


# --------------------------------------------------------------------------------------
# Flow table
# --------------------------------------------------------------------------------------


def test_emits_one_flow_per_bidirectional_five_tuple(simple_pcap):
    flows = run(simple_pcap)
    assert len(flows) == 1
    f = flows[0]
    assert f.n_packets == 11
    assert set(np.unique(f.directions)) == {FORWARD, BACKWARD}


def test_client_is_the_syn_sender(simple_pcap):
    f = run(simple_pcap)[0]
    assert (f.client_ip, f.client_port) == (CLIENT_IP, CLIENT_PORT)
    assert (f.server_ip, f.server_port) == (SERVER_IP, SERVER_PORT)
    assert f.directions[0] == FORWARD


def test_client_inferred_from_service_port_without_a_syn(server_first_pcap):
    """Several dataset captures start mid-flow with the server speaking first."""
    f = run(server_first_pcap)[0]
    assert f.client_ip == CLIENT_IP
    assert f.directions[0] == BACKWARD  # first packet was server -> client


def test_canonical_key_is_direction_independent():
    a = canonical_key(CLIENT_IP, CLIENT_PORT, SERVER_IP, SERVER_PORT, "tcp")
    b = canonical_key(SERVER_IP, SERVER_PORT, CLIENT_IP, CLIENT_PORT, "tcp")
    assert a == b


def test_short_flows_are_dropped_and_counted(two_flow_pcap):
    stats = ParseStats()
    flows = list(Reassembler(min_packets=100).run(PcapSource(two_flow_pcap), stats=stats))
    assert flows == []
    assert stats.flows_dropped_short == 2


def test_stats_account_for_every_frame(garbage_pcap):
    stats = ParseStats()
    list(Reassembler().run(PcapSource(garbage_pcap), stats=stats))
    d = stats.as_dict()
    assert d["packets_used"] + d["non_ip"] + d["non_tcp_udp"] + d["malformed"] == d["packets_read"]


# --------------------------------------------------------------------------------------
# Expiry
# --------------------------------------------------------------------------------------


def test_idle_timeout_splits_a_flow(timeout_pcap):
    flows = run(timeout_pcap)
    assert len(flows) == 2
    assert flows[0].expiry_reason == EXPIRY_IDLE


def test_fin_teardown_closes_the_flow(teardown_pcap):
    """DoD 1.5: a torn-down connection is emitted, and later traffic is a new flow."""
    flows = run(teardown_pcap, min_packets=1)
    assert len(flows) == 2
    assert flows[0].expiry_reason == EXPIRY_FIN
    assert flows[1].expiry_reason == EXPIRY_END_OF_CAPTURE


def test_fin_grace_keeps_the_final_ack_in_the_same_flow(teardown_pcap):
    first = run(teardown_pcap, min_packets=1)[0]
    assert first.n_packets == 11  # 8 data + 2 FIN + the trailing ACK


def test_rst_closes_immediately(tmp_path):
    pk, t = [], 0.0
    for _ in range(4):
        t += 0.1
        pk.append((t, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"\x00" * 60)))
    t += 0.1
    pk.append((t, payload_frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, b"", flags=0x04)))
    for _ in range(4):
        t += 0.1
        pk.append((t, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"\x00" * 60)))
    flows = run(write_pcap(tmp_path / "rst.pcap", pk), min_packets=1)
    assert len(flows) == 2
    assert flows[0].expiry_reason == EXPIRY_RST
    assert flows[0].n_packets == 5


def test_max_duration_caps_a_long_lived_flow(tmp_path):
    pk = [(i * 10.0, payload_frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, b"\x00" * 60))
          for i in range(12)]
    flows = run(write_pcap(tmp_path / "long.pcap", pk), min_packets=1,
                max_duration=30.0, idle_timeout=600.0, sweep_interval=1.0)
    assert len(flows) > 1
    assert all(f.duration <= 45.0 for f in flows)


def test_expiry_is_driven_by_the_packet_clock_not_wall_clock(timeout_pcap):
    """DoD: a pcap run must be reproducible regardless of how fast the file is read."""
    a = [(f.flow_id, f.expiry_reason, f.n_packets) for f in run(timeout_pcap)]
    b = [(f.flow_id, f.expiry_reason, f.n_packets) for f in run(timeout_pcap)]
    assert a == b


def test_table_full_evicts_the_stalest_flow_instead_of_stopping(tmp_path):
    """A live sensor must not stop reading because its flow table filled up."""
    pk = []
    for i in range(8):
        for j in range(4):
            pk.append((i * 0.1 + j * 0.01,
                       payload_frame(CLIENT_IP, SERVER_IP, 40000 + i, SERVER_PORT, b"\x00" * 60)))
    flows = run(write_pcap(tmp_path / "many.pcap", pk), min_packets=1, max_flows_per_pcap=3)
    assert len(flows) == 8  # every flow still emitted, none lost
    assert any(f.expiry_reason == EXPIRY_TABLE_FULL for f in flows)


# --------------------------------------------------------------------------------------
# Handshake capture + hint
# --------------------------------------------------------------------------------------


def test_split_server_hello_is_reassembled(tls_handshake_pcap):
    """The Phase 1 exit criterion: a record spanning segments is recovered intact."""
    f = run(tls_handshake_pcap)[0]
    assert f.l7_hint == "tls"
    assert f.handshake_server_bytes[:1] == b"\x16"
    assert len(f.handshake_server_bytes) == 905  # 5-byte header + 900-byte body, once
    assert f.handshake_client_bytes[:1] == b"\x16"
    assert not f.handshake_incomplete


def test_handshake_capture_can_be_disabled(tls_handshake_pcap):
    f = run(tls_handshake_pcap, capture_handshake=False)[0]
    assert f.handshake_client_bytes is None and f.handshake_server_bytes is None
    assert f.n_packets == 9  # sizes and timings are unaffected


def test_handshake_window_is_bounded(tls_handshake_pcap):
    f = run(tls_handshake_pcap, handshake_window=64)[0]
    assert len(f.handshake_server_bytes) == 64


def test_quic_flow_is_detected_and_initial_captured(quic_pcap):
    f = run(quic_pcap)[0]
    assert f.l7_hint == "quic"
    assert f.handshake_client_bytes[:1] == b"\xc3"
    assert not f.handshake_incomplete


def test_non_handshake_flows_are_not_flagged_incomplete(simple_pcap):
    """`handshake_incomplete` is about parseability, not about carrying no payload."""
    f = run(simple_pcap)[0]
    assert f.l7_hint in ("tcp", "opaque")
    assert f.handshake_incomplete is False


def test_context_hint_reaches_the_flow(tls_handshake_pcap):
    f = run(tls_handshake_pcap, context_hint="tor")[0]
    assert f.l7_hint == "tor"


def test_meta_records_per_direction_completeness(tls_handshake_pcap):
    f = run(tls_handshake_pcap)[0]
    assert f.meta["handshake_server_complete"] is True
    assert f.meta["handshake_bytes_seen"][1] > 0


def test_source_file_is_attached(tls_handshake_pcap):
    f = run(tls_handshake_pcap)[0]
    assert f.source_file == str(tls_handshake_pcap)
    assert f.flow_id.startswith("tls_handshake#")


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


def test_flows_are_not_merged_across_captures(tmp_path, simple_pcap):
    """The same 5-tuple in two files is two flows, not one."""
    import shutil

    d = tmp_path / "pair"
    d.mkdir()
    shutil.copy(simple_pcap, d / "a.pcap")
    shutil.copy(simple_pcap, d / "b.pcap")
    flows = list(Reassembler().run(DirectorySource(d)))
    assert len(flows) == 2
    assert {f.source_file for f in flows} == {str(d / "a.pcap"), str(d / "b.pcap")}


def test_streaming_and_compat_modes_agree_on_packet_content(two_flow_pcap):
    """Both modes see the same packets; only flow boundaries and ordering may differ."""
    streaming = list(Reassembler().run(PcapSource(two_flow_pcap)))
    compat = list(Reassembler(phase0_compat=True).run(PcapSource(two_flow_pcap)))
    assert sum(f.n_packets for f in streaming) == sum(f.n_packets for f in compat)


def test_compat_mode_disables_teardown(teardown_pcap):
    """Phase 0 had no FIN handling; enabling it would move flow boundaries."""
    compat = list(Reassembler(phase0_compat=True).run(PcapSource(teardown_pcap), stats=ParseStats()))
    assert len(compat) == 1
    streaming = run(teardown_pcap, min_packets=1)
    assert len(streaming) == 2
