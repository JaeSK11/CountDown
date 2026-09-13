"""Live capture source -- interface only in Phase 1, implemented in Phase 6.

Defined now so that Phase 6 is a drop-in: the reassembler, the handshake capture and the
crypto parsers are all written against ``Source`` and never learn whether their packets
came off a disk or a wire.  The one property that matters to them is already fixed here:
a live source is unbounded and never signals end-of-capture, so flows must be closed by
the expiry rules (idle / FIN-RST / max-duration), never by "the file ran out".
"""

from __future__ import annotations

from typing import Iterator

from qsentinel.sources.base import RawPacket, Source


class LiveSource(Source):
    """Capture from a network interface.

    Parameters
    ----------
    iface:
        Interface name, e.g. ``"eth0"``.
    bpf:
        Optional BPF filter applied in the kernel, e.g. ``"tcp port 443"``.
    snaplen:
        Bytes captured per frame.  The default is generous enough to hold a full TLS
        ClientHello/ServerHello record, which is the whole point of capturing at all.
    promiscuous:
        Put the interface into promiscuous mode.

    Notes
    -----
    Phase 6 will back this with ``scapy``'s ``AsyncSniffer`` (already a dependency) or
    ``libpcap`` via ``pcapy``/``pypcap``.  Until then every entry point raises
    ``NotImplementedError`` so the stub can never be mistaken for a working capture.
    """

    multi_capture = False

    def __init__(
        self,
        iface: str,
        bpf: str | None = None,
        snaplen: int = 65535,
        promiscuous: bool = True,
        timeout_ms: int = 100,
    ) -> None:
        self.iface = iface
        self.bpf = bpf
        self.snaplen = int(snaplen)
        self.promiscuous = bool(promiscuous)
        self.timeout_ms = int(timeout_ms)

    def __iter__(self) -> Iterator[RawPacket]:
        raise NotImplementedError(
            "LiveSource is a Phase 1 interface stub; live capture lands in Phase 6. "
            "Use PcapSource or DirectorySource for now."
        )

    def close(self) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LiveSource(iface={self.iface!r}, bpf={self.bpf!r})"
