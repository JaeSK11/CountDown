"""The ``Source`` interface: where packets come from.

Phase 1 ships ``PcapSource`` and ``DirectorySource``; Phase 6 adds ``LiveSource`` for a
live NIC.  Everything downstream (``Reassembler``, the batch extractor, the dataset
loaders) consumes this interface only, so the live path is a drop-in with no rewrite.

A ``Source`` yields ``RawPacket`` -- an undecoded frame plus the link type needed to
decode it.  Decoding is deliberately *not* the source's job: a live capture and a pcap
differ in where frames come from, not in how they are parsed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True)
class RawPacket:
    """One undecoded frame.

    ``linktype`` is the libpcap DLT of *this* frame -- carried per packet rather than per
    source because ``DirectorySource`` chains captures that may not share a link type.
    ``source`` names the originating capture, which becomes the flow's ``source_file`` and
    keeps flows from different pcaps from being merged on a colliding 5-tuple.
    """

    ts: float
    linktype: int
    data: bytes
    source: str = ""


class Source(ABC):
    """Abstract packet source."""

    #: True when the source concatenates several captures, so the reassembler must flush
    #: its flow table at each ``RawPacket.source`` change instead of carrying state over.
    multi_capture: bool = False

    @abstractmethod
    def __iter__(self) -> Iterator[RawPacket]:
        """Yield frames in capture order."""

    def close(self) -> None:
        """Release any held file handles / capture handles.  Idempotent."""

    # -- context-manager sugar ---------------------------------------------------------
    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
