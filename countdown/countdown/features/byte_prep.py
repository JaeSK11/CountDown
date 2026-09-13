"""Pcap -> per-packet header/payload byte arrays for the TFE-GNN baseline.

Phase 0 deliberately does not retain raw bytes (``Flow.raw_bytes`` is a reserved slot,
and keeping bytes for 90 GB of pcaps is exactly what the flow_stats/packet_seq reduction
avoids).  Byte-level members need them, so this module is a separate, byte-retaining pass
that never touches the Phase-0 reassembler.

It reproduces the preprocessing in Zhang et al. (WWW'23) Sec. 4.1.2:

* bidirectional flows keyed on the canonical 5-tuple (what they get from SplitCap);
* the Ethernet header, both IP addresses and both port numbers removed;
* packets carrying no payload dropped;
* flows that end up empty, or that exceed 10 000 packets, dropped;
* for ISCX-Tor only, each flow cut into 60-second non-overlapping blocks, because that
  dataset has too few flows otherwise;
* each sample padded/truncated to 50 packets x (40 header, 150 payload) bytes.

Two deviations from the authors' code, both deliberate, both because the code contradicts
the paper's stated intent:

**Addressing really is removed.**  Their ``remove()`` slices ``p[:12]`` and ``p[20:][4:]``
from an array that still carries the 14-byte Ethernet header, so at that offset it takes
out MAC and IP-header bytes and leaves the addresses and ports intact.  A classifier that
can read server IPs is not classifying traffic, so we strip what Sec. 4.1.2 says to strip.

**The header and payload streams stay aligned.**  Their filter drops payload-less packets
from the payload stream but not from the header stream (headers are never empty), so the
two lists can end up different lengths and packet *i*'s header can be paired with packet
*j*'s payload.  We filter both on payload presence, which is what "we first remove the
ones without payload" describes.

Both deviations are recorded in :data:`DEVIATIONS` so a reproduction gap can be chased
against them rather than guessed at.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from countdown.config import get_logger
from countdown.features.traffic_graph import (
    ANOMALOUS_FLOW_THRESHOLD,
    BYTE_PAD_TRUNC_LENGTH,
    FLOW_PAD_TRUNC_LENGTH,
    HEADER_BYTE_PAD_TRUNC_LENGTH,
    pad_truncate_part,
)

log = get_logger(__name__)

#: Documented, intentional departures from the authors' reference implementation.
DEVIATIONS: tuple[str, ...] = (
    "addressing-stripped: IP addresses and ports are actually removed (their remove() "
    "applies the slice at the wrong offset and leaves them in the bytes)",
    "streams-aligned: payload-less packets are dropped from header and payload together "
    "(their filter can desynchronise the two)",
)

#: Tor segmentation window, seconds (Sec. 4.1.2).
TOR_SEGMENT_SECONDS = 60.0


@dataclass(slots=True)
class ByteSample:
    """One training sample: a padded packet-byte matrix pair.

    ``header`` is ``(50, 40)`` and ``payload`` is ``(50, 150)``, both int16 because the
    PAD token (256) does not fit in a byte.  ``n_packets`` is the count *before* padding,
    kept for diagnostics and for the optional masked readout -- the faithful model does
    not use it, since PAD packets are part of the representation.
    """

    header: np.ndarray
    payload: np.ndarray
    n_packets: int
    label_fields: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrepStats:
    """Why packets and flows were dropped.  Reported, never silently discarded."""

    packets_read: int = 0
    packets_no_ip: int = 0
    packets_not_tcp_udp: int = 0
    packets_no_payload: int = 0
    packets_kept: int = 0
    flows_seen: int = 0
    flows_empty: int = 0
    flows_overlong: int = 0
    samples_emitted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in self.__slots__}


def _split_packet(
    buf: bytes, linktype: int, strip_addressing: bool = True
) -> tuple[bytes, bytes] | None:
    """``(header, payload)`` or ``None`` if unusable.

    With ``strip_addressing`` (the default) the header keeps the IP header minus its
    address pair and the transport header minus its port pair -- what Sec. 4.1.2 says to
    do.  With it False the reference's actual byte layout is reproduced instead: their
    ``remove()`` slices ``p[:12]`` and ``p[24:]`` from an array that still begins with
    the 14-byte Ethernet header, which keeps the two MAC addresses and -- crucially --
    leaves the source/destination IPs and ports inside the first 40 header bytes.

    That second mode exists to test a specific hypothesis, not for use.  See
    :data:`DEVIATIONS`.
    """
    import dpkt

    from countdown.flows.decode import decode_l3

    ip, l2_len = decode_l3(buf, linktype)
    if ip is None:
        return None

    raw = buf[l2_len:]  # slice the wire bytes rather than re-pack via dpkt
    if isinstance(ip, dpkt.ip.IP):
        ip_hdr_len = ip.hl * 4
        addr_lo, addr_hi = 12, 20  # IPv4 src/dst
    elif isinstance(ip, dpkt.ip6.IP6):
        ip_hdr_len = 40
        addr_lo, addr_hi = 8, 40  # IPv6 src/dst
    else:
        return None
    if ip_hdr_len <= 0 or len(raw) < ip_hdr_len:
        return None

    transport = ip.data
    if isinstance(transport, dpkt.tcp.TCP):
        trans_hdr_len = transport.off * 4
    elif isinstance(transport, dpkt.udp.UDP):
        trans_hdr_len = 8
    else:
        return None

    ip_hdr = raw[:ip_hdr_len]
    rest = raw[ip_hdr_len:]
    if trans_hdr_len <= 0 or len(rest) < trans_hdr_len:
        return None

    trans_hdr = rest[:trans_hdr_len]
    payload = bytes(getattr(transport, "data", b"") or b"")

    if strip_addressing:
        header = ip_hdr[:addr_lo] + ip_hdr[addr_hi:] + trans_hdr[4:]
    else:
        # Reproduce the reference layout exactly: whole frame, then keep [:12] + [24:].
        frame_header = buf[: len(buf) - len(payload)]
        header = frame_header[:12] + frame_header[24:]
    return header, payload


def _canonical_key(ip, transport) -> tuple:
    """Order-independent 5-tuple so both directions land in one flow (SplitCap-like)."""
    import dpkt

    from countdown.flows.decode import ip_str

    proto = "tcp" if isinstance(transport, dpkt.tcp.TCP) else "udp"
    a = (ip_str(ip.src), int(transport.sport))
    b = (ip_str(ip.dst), int(transport.dport))
    lo, hi = (a, b) if a <= b else (b, a)
    return lo[0], lo[1], hi[0], hi[1], proto


def _pack_sample(
    packets: Sequence[tuple[bytes, bytes]],
    label_fields: dict[str, Any],
    meta: dict[str, Any],
) -> ByteSample:
    """Pad/truncate a packet list into the fixed ``(50, 40)`` / ``(50, 150)`` matrices."""
    kept = list(packets[:FLOW_PAD_TRUNC_LENGTH])
    header = np.stack(
        [pad_truncate_part(h, HEADER_BYTE_PAD_TRUNC_LENGTH) for h, _ in kept]
        or [pad_truncate_part(b"", HEADER_BYTE_PAD_TRUNC_LENGTH)]
    )
    payload = np.stack(
        [pad_truncate_part(p, BYTE_PAD_TRUNC_LENGTH) for _, p in kept]
        or [pad_truncate_part(b"", BYTE_PAD_TRUNC_LENGTH)]
    )

    # Pad the flow itself out to 50 all-PAD packets.  These are not masked away later:
    # a PAD packet's graph is a real graph, and "this flow was short" is information.
    if header.shape[0] < FLOW_PAD_TRUNC_LENGTH:
        missing = FLOW_PAD_TRUNC_LENGTH - header.shape[0]
        header = np.concatenate(
            [header, np.stack([pad_truncate_part(b"", HEADER_BYTE_PAD_TRUNC_LENGTH)] * missing)]
        )
        payload = np.concatenate(
            [payload, np.stack([pad_truncate_part(b"", BYTE_PAD_TRUNC_LENGTH)] * missing)]
        )

    return ByteSample(
        header=header,
        payload=payload,
        n_packets=len(kept),
        label_fields=dict(label_fields),
        meta=dict(meta),
    )


def _segment_by_time(
    packets: list[tuple[bytes, bytes]], times: list[float]
) -> Iterator[list[tuple[bytes, bytes]]]:
    """Cut a flow into 60-second non-overlapping blocks, measured from its first packet.

    Empty blocks are skipped rather than emitted, matching the reference.
    """
    if not packets:
        return
    t0 = times[0]
    span = times[-1] - t0
    n_blocks = 1 if span <= TOR_SEGMENT_SECONDS else int(span // TOR_SEGMENT_SECONDS) + 1
    rel = np.asarray(times) - t0
    for b in range(n_blocks):
        lo, hi = b * TOR_SEGMENT_SECONDS, (b + 1) * TOR_SEGMENT_SECONDS
        idx = np.nonzero((rel >= lo) & (rel < hi))[0]
        if idx.size:
            yield [packets[i] for i in idx]


def extract_byte_samples(
    path: str | Path,
    label_fields: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    segment_seconds: float | None = None,
    stats: PrepStats | None = None,
    max_flows: int = 200_000,
    strip_addressing: bool = True,
) -> list[ByteSample]:
    """Turn one capture into TFE-GNN samples.

    Parameters
    ----------
    segment_seconds
        ``None`` (default) emits one sample per flow -- the ISCX-VPN/nonVPN setting.
        Pass ``60.0`` for ISCX-Tor/nonTor, where the paper cuts each flow into
        non-overlapping blocks to make up for the scarcity of flows.
    """
    from countdown.flows.decode import decode_l3
    from countdown.sources.pcap import PcapSource

    path = Path(path)
    stats = stats if stats is not None else PrepStats()
    base_meta = dict(meta or {})
    base_meta.setdefault("source_file", str(path))
    labels = dict(label_fields or {})

    flows: OrderedDict[tuple, tuple[list[tuple[bytes, bytes]], list[float]]] = OrderedDict()

    for pkt in PcapSource(path):
        stats.packets_read += 1
        if len(flows) >= max_flows:
            break
        try:
            ip, _ = decode_l3(pkt.data, pkt.linktype)
        except Exception:
            stats.packets_no_ip += 1
            continue
        if ip is None:
            stats.packets_no_ip += 1
            continue

        try:
            split = _split_packet(pkt.data, pkt.linktype, strip_addressing)
        except Exception:
            stats.packets_not_tcp_udp += 1
            continue
        if split is None:
            stats.packets_not_tcp_udp += 1
            continue

        header, payload = split
        # "we first remove the ones without payload" -- dropped from *both* streams so
        # header i and payload i keep describing the same packet.
        if not payload:
            stats.packets_no_payload += 1
            continue

        try:
            key = _canonical_key(ip, ip.data)
        except Exception:
            stats.packets_not_tcp_udp += 1
            continue

        bucket = flows.setdefault(key, ([], []))
        bucket[0].append((header, payload))
        bucket[1].append(float(pkt.ts))
        stats.packets_kept += 1

    samples: list[ByteSample] = []
    for key, (packets, times) in flows.items():
        stats.flows_seen += 1
        if not packets:
            stats.flows_empty += 1
            continue

        flow_meta = dict(base_meta)
        flow_meta["five_tuple"] = key

        blocks = list(_segment_by_time(packets, times)) if segment_seconds else [packets]
        for i, block in enumerate(blocks):
            if not block:
                continue
            # The overlong check applies to the emitted *sample*, not the parent flow.
            # Order matters enormously on Tor: all traffic rides a handful of long-lived
            # TLS connections to the guard, so a capture is often one flow of far more
            # than 10 000 packets.  Checking before segmentation discards the entire
            # capture; the reference slices into 60s blocks first and checks each block,
            # which is what keeps ISCX-Tor usable at all.  For the unsegmented case the
            # block *is* the flow, so this is identical to checking up front.
            if len(block) > ANOMALOUS_FLOW_THRESHOLD:
                stats.flows_overlong += 1
                continue
            m = dict(flow_meta)
            if segment_seconds:
                m["segment_index"] = i
            samples.append(_pack_sample(block, labels, m))
            stats.samples_emitted += 1

    return samples
