"""Phase 10 section 2a: flows dropped by ``min_packets`` are tallied by shape, so a SYN scan
is counted, not silently lost.  Synthetic captures; no models."""

from __future__ import annotations

from countdown.flows import Reassembler
from countdown.flows.short_flows import ShortFlowTally, shape_of
from countdown.sources import PcapSource
from tests.conftest import CLIENT_IP, SERVER_IP, payload_frame, write_pcap

SYN, ACK, RST, SYNACK = 0x02, 0x10, 0x04, 0x12
SCANNER = "198.51.100.7"


def _scan_capture(tmp_path):
    """A scanner probing 10 ports on one host (4 filtered, 3 closed, 3 open + reset), one
    real 6-packet connection, a stray ACK and a one-packet UDP query."""
    pk, t = [], 0.0

    def add(src, dst, sport, dport, flags, data=b""):
        nonlocal t
        t += 0.001
        pk.append((t, payload_frame(src, dst, sport, dport, data, flags=flags)))

    for port in (22, 23, 3389, 5900):                       # filtered: SYN, no answer
        add(SCANNER, SERVER_IP, 40000, port, SYN)
    for port in (21, 25, 110):                              # closed: SYN -> RST
        add(SCANNER, SERVER_IP, 40000, port, SYN)
        add(SERVER_IP, SCANNER, port, 40000, RST | ACK)
    for port in (80, 443, 8080):                            # open: SYN, SYN-ACK, scanner RST
        add(SCANNER, SERVER_IP, 40000, port, SYN)
        add(SERVER_IP, SCANNER, port, 40000, SYNACK)
        add(SCANNER, SERVER_IP, 40000, port, RST)
    add(CLIENT_IP, SERVER_IP, 51000, 443, SYN)              # a real connection: kept
    add(SERVER_IP, CLIENT_IP, 443, 51000, SYNACK)
    for _ in range(2):
        add(CLIENT_IP, SERVER_IP, 51000, 443, ACK, b"\x00" * 80)
        add(SERVER_IP, CLIENT_IP, 443, 51000, ACK, b"\x00" * 400)
    add(CLIENT_IP, SERVER_IP, 51999, 443, ACK)              # stray ACK
    return write_pcap(tmp_path / "scan.pcap", pk)


def test_a_syn_scan_is_counted_by_shape_not_classified(tmp_path):
    reasm = Reassembler()
    flows = list(reasm.run(PcapSource(_scan_capture(tmp_path))))
    assert len(flows) == 1 and flows[0].n_packets == 6       # only the real connection
    d = reasm.short_flows.as_dict()
    assert d["by_shape"] == {"syn_only": 4, "syn_rst": 3, "half_open": 3, "tcp_payload": 0,
                             "tcp_other": 1, "udp": 0}
    assert d["dropped_short"] == 11 and d["scan_like"] == 10
    assert d["scan_initiators"] == 1 and d["scan_targets"] == 1 and d["scan_ports_distinct"] == 10
    assert SCANNER not in str(d) and SERVER_IP not in str(d)  # counts only, never addresses


def test_tally_accumulates_and_merges():
    a, b = ShortFlowTally(), ShortFlowTally()
    a.add("syn_only", "1.1.1.1", "2.2.2.2", 22)
    b.add("syn_only", "1.1.1.1", "3.3.3.3", 22)
    b.add("udp", "4.4.4.4", "2.2.2.2", 53)
    a.merge(b)
    d = a.as_dict()
    assert d["dropped_short"] == 3 and d["scan_like"] == 2
    assert d["scan_initiators"] == 1 and d["scan_targets"] == 2 and d["scan_top_ports"] == [[22, 2]]
    assert "2 scan-like" in a.summary_line() and "top 22 (2)" in a.summary_line()
    assert ShortFlowTally().summary_line().endswith("none dropped")


def test_shape_rules():
    assert shape_of("udp", False, False, False, 1) == "udp"
    assert shape_of("tcp", True, False, False, 0) == "syn_only"
    assert shape_of("tcp", True, False, True, 0) == "syn_rst"
    assert shape_of("tcp", True, True, True, 0) == "half_open"
    assert shape_of("tcp", True, True, False, 2) == "tcp_payload"
    assert shape_of("tcp", False, False, True, 0) == "tcp_other"


def test_triage_markdown_reports_the_dropped_flows():
    from countdown.cli import _short_markdown

    t = ShortFlowTally()
    for port in (22, 23):
        t.add("syn_only", "1.1.1.1", "2.2.2.2", port)
    md = _short_markdown(t.as_dict())
    assert "2 look like scan probes" in md and "| SYN only, no reply | 2 |" in md
