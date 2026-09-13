"""Phase 1 tasks 1.1-1.3: the Source interface, file sources, and the live stub."""

from __future__ import annotations

import pytest

from countdown.flows.decode import ParseStats
from countdown.sources import DirectorySource, LiveSource, PcapSource, RawPacket, Source


def test_pcap_source_yields_raw_packets(simple_pcap):
    packets = list(PcapSource(simple_pcap))
    assert len(packets) == 11
    assert all(isinstance(p, RawPacket) for p in packets)
    assert packets[0].linktype == 1  # DLT_EN10MB
    assert packets[0].source == str(simple_pcap)
    assert packets[0].ts <= packets[-1].ts


def test_pcap_source_reports_linktype_without_iterating(simple_pcap):
    assert PcapSource(simple_pcap).linktype == 1


def test_pcap_source_counts_packets_read(simple_pcap):
    stats = ParseStats()
    list(PcapSource(simple_pcap, stats=stats))
    assert stats.packets_read == 11


def test_pcap_source_is_a_context_manager(simple_pcap):
    with PcapSource(simple_pcap) as src:
        assert sum(1 for _ in src) == 11


def test_pcap_source_is_re_iterable(simple_pcap):
    """The reassembler may be run twice over the same source in tests and scripts."""
    src = PcapSource(simple_pcap)
    assert len(list(src)) == len(list(src)) == 11


def test_directory_source_chains_captures(tmp_path, simple_pcap, two_flow_pcap):
    src = DirectorySource(simple_pcap.parent)
    files = src.files()
    assert simple_pcap in files and two_flow_pcap in files
    packets = list(src)
    assert len(packets) == sum(len(list(PcapSource(f))) for f in files)


def test_directory_source_tags_each_packet_with_its_capture(simple_pcap, two_flow_pcap):
    """Flows must stay attributable to a file, and must not merge across captures."""
    sources = {p.source for p in DirectorySource(simple_pcap.parent)}
    assert sources == {str(simple_pcap), str(two_flow_pcap)}


def test_directory_source_declares_multi_capture(simple_pcap):
    assert DirectorySource(simple_pcap.parent).multi_capture is True
    assert PcapSource(simple_pcap).multi_capture is False


def test_directory_source_skips_unreadable_captures(tmp_path, simple_pcap):
    """One corrupt file must not kill a walk over a 46k-file dataset."""
    import shutil

    d = tmp_path / "mixed"
    d.mkdir()
    shutil.copy(simple_pcap, d / "good.pcap")
    (d / "broken.pcap").write_bytes(b"not a capture at all")
    src = DirectorySource(d)
    packets = list(src)
    assert len(packets) == 11
    assert src.failed == [str(d / "broken.pcap")]


def test_directory_source_respects_patterns(tmp_path, simple_pcap):
    import shutil

    d = tmp_path / "patterns"
    d.mkdir()
    shutil.copy(simple_pcap, d / "keep.pcap")
    shutil.copy(simple_pcap, d / "skip.pcapng")
    assert len(DirectorySource(d, patterns="**/*.pcap").files()) == 1
    assert len(DirectorySource(d).files()) == 2


def test_live_source_is_an_importable_stub():
    """DoD 1.3: importable, typed, and raises rather than pretending to capture."""
    src = LiveSource("eth0", bpf="tcp port 443")
    assert isinstance(src, Source)
    assert (src.iface, src.bpf, src.snaplen) == ("eth0", "tcp port 443", 65535)
    with pytest.raises(NotImplementedError, match="Phase 6"):
        list(src)
    src.close()  # must not raise
