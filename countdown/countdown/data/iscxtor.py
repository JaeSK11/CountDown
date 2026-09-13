"""ISCXTor2016: multi-flow pcaps split across ``Tor/`` and ``NonTor/`` directories.

95 captures (51 Tor + 44 NonTor).  Filenames mix explicit category prefixes
(``VOIP_``, ``FILE-TRANSFER_``, ...) with free-form names (``torGoogle``,
``Workstation_Thunderbird_Imap``), so the rule list tries prefixes first and content
keywords second.
"""

from __future__ import annotations

from countdown.data.base import DatasetLoader, Item, register_loader
from countdown.schema import TunnelType


@register_loader("iscxtor")
class ISCXTorLoader(DatasetLoader):
    def discover(self) -> list[Item]:
        rules = self.spec["label_rules"]
        tor_dir = self.spec.get("tor_dir", "Tor")
        items: list[Item] = []
        self.unmapped = []

        for path in self._glob():
            stem = path.stem.lower()
            traffic_type = self.map_traffic_type(stem, rules)
            if traffic_type is None:
                self.unmapped.append(path.name)
                continue
            rel = path.relative_to(self.root)
            is_tor = rel.parts[0] == tor_dir
            items.append(
                Item(
                    path=path,
                    label_fields={
                        "traffic_type": traffic_type,
                        "is_tor": is_tor,
                        "is_vpn": False,
                        "tunnel_type": (TunnelType.TOR if is_tor else TunnelType.NONE).value,
                        "app": stem,
                    },
                    meta={"source_file": str(path), "stem": path.stem,
                          "tor_dir": rel.parts[0]},
                )
            )
        self.report_unmapped()
        return items
