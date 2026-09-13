"""Flow-assembly unit tests (task 0.3 DoD)."""

from __future__ import annotations

import numpy as np
import pytest

from qsentinel.flows.extract import FlowExtractor, ParseStats, open_capture
from qsentinel.schema import BACKWARD, FORWARD

from conftest import CLIENT_IP, CLIENT_PORT, SERVER_IP, SERVER_PORT


def test_single_flow_direction_and_five_tuple(simple_pcap):
    flows = FlowExtractor(min_packets=1).extract(simple_pcap)
    assert len(flows) == 1
    f = flows[0]

    # SYN sender is the client, so the five-tuple is oriented client -> server.
    assert f.five_tuple.src_ip == CLIENT_IP
    assert f.five_tuple.src_port == CLIENT_PORT
    assert f.five_tuple.dst_ip == SERVER_IP
    assert f.five_tuple.dst_port == SERVER_PORT
    assert f.five_tuple.l4_proto == "tcp"

    assert f.n_packets == 11  # 1 SYN + 5 up + 5 down
    assert set(np.unique(f.directions)) == {FORWARD, BACKWARD}
    assert (f.directions > 0).sum() == 6
    assert (f.directions < 0).sum() == 5
    assert f.directions[0] == FORWARD  # the SYN


def test_bidirectional_grouping_keeps_one_flow_per_five_tuple(two_flow_pcap):
    flows = FlowExtractor(min_packets=1).extract(two_flow_pcap)
    assert len(flows) == 2
    ports = sorted(f.five_tuple.src_port for f in flows)
    assert ports == [51000, 51001]
    for f in flows:
        # both directions landed in the same flow
        assert (f.directions > 0).sum() == 6
        assert (f.directions < 0).sum() == 6


def test_idle_timeout_splits_a_flow(timeout_pcap):
    joined = FlowExtractor(min_packets=1, flow_timeout=1000.0).extract(timeout_pcap)
    split = FlowExtractor(min_packets=1, flow_timeout=64.0).extract(timeout_pcap)
    assert len(joined) == 1
    assert len(split) == 2
    assert sum(f.n_packets for f in split) == joined[0].n_packets


def test_min_packets_drops_short_flows(two_flow_pcap):
    stats = ParseStats()
    flows = FlowExtractor(min_packets=100).extract(two_flow_pcap, stats=stats)
    assert flows == []
    assert stats.flows_dropped_short == 2


def test_client_inferred_from_service_port_when_no_syn(server_first_pcap):
    """Capture starts with a server packet and has no SYN: the port rule must still win."""
    flows = FlowExtractor(min_packets=1).extract(server_first_pcap)
    assert len(flows) == 1
    f = flows[0]
    assert f.five_tuple.src_ip == CLIENT_IP  # ephemeral side is the client
    assert f.five_tuple.dst_port == SERVER_PORT
    assert f.directions[0] == BACKWARD  # first packet really is server -> client


def test_sizes_are_wire_lengths(simple_pcap):
    f = FlowExtractor(min_packets=1).extract(simple_pcap)[0]
    # 14 B Ethernet + 20 B IP + 20 B TCP + payload
    assert f.sizes[0] == 54  # SYN, no payload
    assert 154 in f.sizes.tolist()  # 100 B payload
    assert 554 in f.sizes.tolist()  # 500 B payload


def test_timestamps_are_monotonic_and_duration_positive(simple_pcap):
    f = FlowExtractor(min_packets=1).extract(simple_pcap)[0]
    assert np.all(np.diff(f.timestamps) >= 0)
    assert f.duration > 0
    assert f.start_ts == pytest.approx(f.timestamps[0])
    assert f.end_ts == pytest.approx(f.timestamps[-1])


def test_extraction_is_deterministic(two_flow_pcap):
    a = FlowExtractor(min_packets=1).extract(two_flow_pcap)
    b = FlowExtractor(min_packets=1).extract(two_flow_pcap)
    assert [f.flow_id for f in a] == [f.flow_id for f in b]
    for x, y in zip(a, b):
        assert np.array_equal(x.sizes, y.sizes)
        assert np.array_equal(x.directions, y.directions)


def test_open_capture_sniffs_format_not_extension(tmp_path, simple_pcap):
    """A pcap named .pcapng (and vice versa) must still open -- PostQuantumTLS does this."""
    misnamed = tmp_path / "actually_pcap.pcapng"
    misnamed.write_bytes(simple_pcap.read_bytes())
    reader, fh = open_capture(misnamed)
    try:
        assert reader.datalink() == 1
    finally:
        fh.close()


def test_labels_are_attached_to_every_flow(two_flow_pcap):
    fields = {"traffic_type": "chat", "is_vpn": True}
    flows = FlowExtractor(min_packets=1).extract(
        two_flow_pcap, dataset="unit", label="chat", label_fields=fields
    )
    for f in flows:
        assert f.dataset == "unit"
        assert f.label == "chat"
        assert f.label_fields == fields
        # Phase-2 slots exist but stay empty in Phase 0.
        assert f.tls_version is None and f.kem_group is None and f.pqc_verdict is None


def test_parse_stats_account_for_every_frame(tmp_path, simple_pcap):
    """Non-IP and non-TCP/UDP frames must be counted, not silently discarded."""
    import dpkt
    from conftest import write_pcap

    arp = bytes(dpkt.ethernet.Ethernet(
        src=b"\xaa" * 6, dst=b"\xff" * 6, type=dpkt.ethernet.ETH_TYPE_ARP,
        data=dpkt.arp.ARP(),
    ))
    icmp_ip = dpkt.ip.IP(src=b"\x01\x02\x03\x04", dst=b"\x05\x06\x07\x08",
                         p=dpkt.ip.IP_PROTO_ICMP, data=dpkt.icmp.ICMP(type=8))
    icmp_ip.len = len(bytes(icmp_ip))
    icmp = bytes(dpkt.ethernet.Ethernet(
        src=b"\xaa" * 6, dst=b"\xbb" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=icmp_ip))

    mixed = tmp_path / "mixed.pcap"
    packets = [(0.0, arp), (0.1, icmp)]
    with open(simple_pcap, "rb") as fh:
        for i, (ts, buf) in enumerate(dpkt.pcap.Reader(fh)):
            packets.append((1.0 + i * 0.1, bytes(buf)))
    write_pcap(mixed, packets)

    stats = ParseStats()
    flows = FlowExtractor(min_packets=1).extract(mixed, stats=stats)
    assert stats.packets_read == len(packets)
    assert stats.non_ip == 1          # the ARP frame
    assert stats.non_tcp_udp == 1     # the ICMP packet
    assert stats.packets_used == 11
    assert stats.packets_used + stats.non_ip + stats.non_tcp_udp + stats.malformed \
        == stats.packets_read
    assert len(flows) == 1
