"""PostQuantumTLS: 90 Android app captures, one file per package.

The files are pcapng despite their ``.pcap`` suffix -- the reader sniffs magic bytes, so no
special handling is needed here.  Phase 0 only needs these to load as flows; their real job
is Phase 2, where the handshakes provide PQC/KEM ground truth.
"""

from __future__ import annotations

from qsentinel.data.base import DatasetLoader, Item, register_loader


@register_loader("postquantumtls")
class PostQuantumTLSLoader(DatasetLoader):
    def discover(self) -> list[Item]:
        items = []
        for path in self._glob():
            package = path.stem
            items.append(
                Item(
                    path=path,
                    label_fields={"app": package, "is_vpn": False, "is_tor": False,
                                  "tunnel_type": "none"},
                    meta={"source_file": str(path), "package": package},
                )
            )
        return items
