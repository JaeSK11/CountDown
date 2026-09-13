"""CSTNET-TLS1.3: 120 app directories of single-session pcaps (~491 each, ~46k total).

Ingest modes (``ingest_mode`` in ``configs/datasets.yaml``):

``raw``     parse the pcaps into flows -- consistent with every other dataset and the
            representation Phase 2 needs for handshake/KEM parsing.  Implemented here.
``etbert``  reuse the vendored ``extracted/flow_500`` / ``packet_5000`` ET-BERT tensors
            without re-parsing.  A Phase-4 hook; raises for now.

Because the corpus is large, ``filters.max_pcaps_per_app`` and ``filters.top_k_apps``
subsample at discovery time so that iteration stays fast.
"""

from __future__ import annotations

from countdown.data.base import DatasetLoader, Item, register_loader


@register_loader("cstnet")
class CSTNETLoader(DatasetLoader):
    def discover(self) -> list[Item]:
        mode = self.spec.get("ingest_mode", "raw")
        if mode != "raw":
            raise NotImplementedError(
                f"cstnet ingest_mode={mode!r} is a Phase-4 hook; only 'raw' exists in Phase 0"
            )

        base = self.root / self.spec.get("raw_subdir", "extracted/cstnet-tls 1.3")
        if not base.exists():
            raise FileNotFoundError(f"CSTNET raw directory not found: {base}")

        filters = self.spec.get("filters") or {}
        max_per_app = filters.get("max_pcaps_per_app")
        top_k = filters.get("top_k_apps")

        by_app: dict[str, list] = {}
        for path in self._glob(base):
            by_app.setdefault(path.parent.name, []).append(path)

        apps = sorted(by_app)
        if top_k:
            apps = sorted(apps, key=lambda a: (-len(by_app[a]), a))[:top_k]

        items: list[Item] = []
        for app in sorted(apps):
            paths = sorted(by_app[app])
            if max_per_app:
                paths = paths[:max_per_app]
            for path in paths:
                items.append(
                    Item(
                        path=path,
                        label_fields={"app": app, "is_vpn": False, "is_tor": False,
                                      "tunnel_type": "none"},
                        meta={"source_file": str(path), "app": app},
                    )
                )
        return items
