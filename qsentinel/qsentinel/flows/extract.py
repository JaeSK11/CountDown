"""pcap/pcapng -> ``list[Flow]``: the batch entry point used by the dataset loaders.

Phase 1 re-implemented this as a **thin wrapper over ``Reassembler``** so that batch and
streaming share one code path (task 1.8).  All the assembly logic now lives in
``flows/reassembly.py``; what remains here is the batch ergonomics the loaders rely on
(take a path, return a list) plus the Phase 0 import surface.

Design notes carried over from Phase 0 -- all still enforced, now inside the reassembler:

* **Format sniffing by magic bytes, not file extension.**  The PostQuantumTLS captures are
  pcapng files named ``*.pcap``, so trusting the suffix silently loses that whole dataset.
* **Direction** is canonicalised so ``+1`` is always client->server; the client is the SYN
  sender, else the non-service-port endpoint, else the first sender.
* **Packet size** is ``L2 header length + IP total length``, not ``len(buf)``, which keeps
  sizes correct for the snaplen-truncated captures in these datasets.
* Malformed / non-IP / non-TCP-UDP packets are counted in ``ParseStats`` and skipped, never
  silently dropped without a trace.

The wrapper runs the reassembler in ``phase0_compat`` mode: no FIN/RST teardown, no
periodic sweep, flows sorted by start time with Phase 0's ``flow_id`` numbering.  That is
what keeps the cached ``(X, y)`` matrices byte-identical across this refactor -- see
``tests/test_phase0_regression.py``, which pins 29 captures against a pre-refactor golden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qsentinel.config import get_logger
from qsentinel.flows.decode import (
    DLT_EN10MB,
    DLT_IEEE802_11,
    DLT_LINUX_SLL,
    DLT_LINUX_SLL2,
    DLT_NULL,
    DLT_RAW,
    DLT_RAW_ALT,
    DLT_RAW_BSD,
    PCAP_MAGICS,
    PCAPNG_MAGIC,
    TCP_ACK,
    TCP_SYN,
    ParseStats,
    _decode_l3,
    _ip_str,
    _wire_size,
    iter_ip_packets,
    open_capture,
)
from qsentinel.flows.reassembly import (
    SERVICE_PORTS,
    Reassembler,
    _is_service_port,
    canonical_key,
)
from qsentinel.schema import Flow

log = get_logger(__name__)


class FlowExtractor:
    """Assemble flows from a capture file.

    Parameters come from ``config.flow`` (see ``configs/features.yaml``)::

        flow_timeout        seconds of idleness that closes a flow (default 64)
        min_packets         drop flows with fewer packets (default 4)
        max_flows_per_pcap  guard against pathological captures (default 200000)

    and, added in Phase 1, from ``config.reassembly``::

        handshake_window    bytes of each direction's opening to reassemble (default 8192)
        capture_handshake   set False to skip head reassembly entirely
    """

    def __init__(
        self,
        flow_timeout: float = 64.0,
        min_packets: int = 4,
        max_flows_per_pcap: int = 200_000,
        handshake_window: int = 8192,
        capture_handshake: bool = True,
        context_hint: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.flow_timeout = float(flow_timeout)
        self.min_packets = int(min_packets)
        self.max_flows_per_pcap = int(max_flows_per_pcap)
        self.handshake_window = int(handshake_window)
        self.capture_handshake = bool(capture_handshake)
        self.context_hint = context_hint

    @classmethod
    def from_config(cls, config) -> "FlowExtractor":
        params: dict[str, Any] = {}
        if config is not None:
            params.update(dict(getattr(config, "flow", {}) or {}))
            params.update(dict(getattr(config, "reassembly", {}) or {}))
        return cls(**params)

    def reassembler(self) -> Reassembler:
        """The Phase-0-compatible reassembler this extractor drives."""
        return Reassembler(
            flow_timeout=self.flow_timeout,
            min_packets=self.min_packets,
            max_flows_per_pcap=self.max_flows_per_pcap,
            handshake_window=self.handshake_window,
            capture_handshake=self.capture_handshake,
            context_hint=self.context_hint,
            phase0_compat=True,
        )

    # -- main entry point --------------------------------------------------------------
    def extract(
        self,
        path: str | Path,
        dataset: str = "",
        label: str = "",
        label_fields: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        stats: ParseStats | None = None,
    ) -> list[Flow]:
        path = Path(path)
        stats = stats if stats is not None else ParseStats()
        base_meta = dict(meta or {})
        base_meta.setdefault("source_file", str(path))

        # Imported here, not at module scope: `sources.pcap` needs `flows.decode`, and a
        # top-level import would make `import qsentinel.sources` and `import
        # qsentinel.flows` resolve differently depending on which came first.
        from qsentinel.sources.pcap import PcapSource

        source = PcapSource(path, stats=stats)
        return list(
            self.reassembler().run(
                source,
                dataset=dataset,
                label=label,
                label_fields=dict(label_fields or {}),
                meta=base_meta,
                stats=stats,
            )
        )

    # -- helpers -----------------------------------------------------------------------
    @staticmethod
    def _canonical_key(sip: str, sport: int, dip: str, dport: int, proto: str) -> tuple:
        """Order-independent key so both directions land in the same flow."""
        return canonical_key(sip, sport, dip, dport, proto)


def extract_flows(path: str | Path, config=None, **kwargs: Any) -> list[Flow]:
    """Convenience wrapper: build a ``FlowExtractor`` from config and run it."""
    params: dict[str, Any] = {}
    if config is not None:
        params.update(dict(getattr(config, "flow", {}) or {}))
        params.update(dict(getattr(config, "reassembly", {}) or {}))
    return FlowExtractor(**params).extract(path, **kwargs)


__all__ = [
    "FlowExtractor", "ParseStats", "extract_flows", "open_capture", "iter_ip_packets",
    "SERVICE_PORTS", "canonical_key",
]
