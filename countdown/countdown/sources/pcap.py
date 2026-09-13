"""File-backed sources: one capture (``PcapSource``) or a whole dataset folder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from countdown.config import get_logger
from countdown.flows.decode import ParseStats, open_capture
from countdown.sources.base import RawPacket, Source

log = get_logger(__name__)


class PcapSource(Source):
    """Stream one pcap/pcapng file.

    Format is decided from magic bytes, not the file suffix -- the PostQuantumTLS
    captures are pcapng named ``*.pcap``.
    """

    def __init__(self, path: str | Path, stats: ParseStats | None = None) -> None:
        self.path = Path(path)
        self.stats = stats if stats is not None else ParseStats()
        self._fh = None
        self._linktype: int | None = None

    @property
    def linktype(self) -> int:
        """The capture's DLT.  Opens the file if it is not open yet."""
        if self._linktype is None:
            reader, fh = open_capture(self.path)
            self._linktype = int(reader.datalink())
            fh.close()
        return self._linktype

    def __iter__(self) -> Iterator[RawPacket]:
        reader, fh = open_capture(self.path)
        self._fh = fh
        try:
            linktype = int(reader.datalink())
            self._linktype = linktype
            name = str(self.path)
            for ts, buf in reader:
                self.stats.packets_read += 1
                yield RawPacket(float(ts), linktype, bytes(buf), name)
        finally:
            fh.close()
            self._fh = None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PcapSource({self.path.name!r})"


class DirectorySource(Source):
    """Chain every capture under a directory into one packet stream.

    Used to run a whole dataset folder through the reassembler in one pass.  Each frame
    carries its originating file in ``RawPacket.source`` so flows stay attributable (and
    so per-file labels survive), and ``multi_capture`` tells the reassembler to flush its
    flow table between files rather than merging colliding 5-tuples across captures.

    A capture that fails to open is logged and skipped -- one corrupt file must not kill
    a 46k-file dataset walk.
    """

    multi_capture = True

    def __init__(
        self,
        root: str | Path,
        patterns: Sequence[str] | str = ("**/*.pcap", "**/*.pcapng"),
        recursive: bool = True,
        stats: ParseStats | None = None,
    ) -> None:
        self.root = Path(root)
        if isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = list(patterns)
        self.recursive = recursive
        self.stats = stats if stats is not None else ParseStats()
        self.failed: list[str] = []

    def files(self) -> list[Path]:
        """Every matching capture, de-duplicated, in sorted order."""
        seen: dict[Path, None] = {}
        for pattern in self.patterns:
            if not self.recursive:
                pattern = pattern.replace("**/", "")
            for p in sorted(self.root.glob(pattern)):
                if p.is_file():
                    seen[p] = None
        return list(seen)

    def __iter__(self) -> Iterator[RawPacket]:
        for path in self.files():
            try:
                yield from PcapSource(path, stats=self.stats)
            except Exception as exc:
                self.failed.append(str(path))
                log.warning("skipping unreadable capture %s: %s: %s",
                            path, type(exc).__name__, exc)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DirectorySource({str(self.root)!r}, patterns={self.patterns})"
