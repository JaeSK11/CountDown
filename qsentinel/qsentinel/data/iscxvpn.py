"""ISCXVPN2016: multi-flow pcaps, labels encoded in the filename.

140 captures (102 ``.pcap`` + 38 ``.pcapng``).  A ``vpn_`` prefix marks tunnelled traffic;
the rest of the stem names the application/activity, which the ordered regex rules in
``configs/datasets.yaml`` map onto the unified ``traffic_type`` taxonomy.
"""

from __future__ import annotations

from qsentinel.data.base import DatasetLoader, Item, register_loader
from qsentinel.schema import TunnelType


@register_loader("iscxvpn")
class ISCXVPNLoader(DatasetLoader):
    def discover(self) -> list[Item]:
        rules = self.spec["label_rules"]
        vpn_prefix = self.spec.get("vpn_prefix", "vpn_")
        items: list[Item] = []
        self.unmapped = []

        for path in self._glob():
            stem = path.stem.lower()
            is_vpn = stem.startswith(vpn_prefix)
            core = stem[len(vpn_prefix):] if is_vpn else stem
            traffic_type = self.map_traffic_type(core, rules)
            if traffic_type is None:
                self.unmapped.append(path.name)
                continue
            items.append(
                Item(
                    path=path,
                    label_fields={
                        "traffic_type": traffic_type,
                        "is_vpn": is_vpn,
                        "is_tor": False,
                        "tunnel_type": (TunnelType.VPN if is_vpn else TunnelType.NONE).value,
                        "app": core,
                    },
                    meta={"source_file": str(path), "stem": path.stem},
                )
            )
        self.report_unmapped()
        return items
