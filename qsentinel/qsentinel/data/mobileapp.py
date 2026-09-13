"""MobileAppActivity-Sensors2022: 368 frame-level Wi-Fi CSVs (no pcaps, no 5-tuple).

Each CSV is one capture of one in-app activity, laid out as
``D-OutCsv/<App>/<Activity>/*.csv``: 8 apps, 92 App/Activity classes.

Direction
---------
These are 802.11 frames, so there is no IP 5-tuple to orient.  In a 3-address frame an
*uplink* frame carries ``wlan.ta == wlan.sa`` (the station transmits its own traffic), while a
*downlink* frame is relayed by the AP and has ``wlan.ta != wlan.sa``.  The client MAC is
therefore the address that dominates the ``sa == ta`` frames, inferred per capture so that a
different device in some capture still resolves correctly.  This is measured, not assumed:
across the corpus exactly two MACs appear, and the rule separates them cleanly.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from qsentinel.config import get_logger
from qsentinel.data.base import DatasetLoader, Item, register_loader
from qsentinel.schema import BACKWARD, FORWARD, FiveTuple, Flow

log = get_logger(__name__)


@register_loader("mobileapp")
class MobileAppLoader(DatasetLoader):
    def discover(self) -> list[Item]:
        base = self.root / self.spec.get("csv_subdir", "D-OutCsv")
        if not base.exists():
            raise FileNotFoundError(f"MobileApp CSV directory not found: {base}")

        items: list[Item] = []
        for path in self._glob(base):
            rel = path.relative_to(base)
            if len(rel.parts) < 3:
                continue
            app, activity = rel.parts[0], rel.parts[1]
            items.append(
                Item(
                    path=path,
                    label_fields={
                        "app": app,
                        "activity": f"{app}/{activity}",
                        "is_vpn": False,
                        "is_tor": False,
                        "tunnel_type": "none",
                    },
                    meta={"source_file": str(path), "app": app, "activity": activity},
                )
            )
        return items

    # CSVs are small and numerous; a plain loop beats process-pool overhead here.
    def build_flows(self, items: Sequence[Item], workers: int | None = None) -> list[Flow]:
        cols = self.spec.get("columns", {})
        c_ts = cols.get("ts", "frame.time_relative")
        c_size = cols.get("size", "frame.len")
        c_src = cols.get("src", "wlan.sa")
        c_dst = cols.get("dst", "wlan.da")
        c_ta = cols.get("ta", "wlan.ta")
        min_packets = int(self.config.flow.get("min_packets", 4))

        flows: list[Flow] = []
        for item in items:
            try:
                df = pd.read_csv(
                    item.path,
                    usecols=lambda c: c in {c_ts, c_size, c_src, c_dst, c_ta},
                    dtype=str,
                )
            except Exception as exc:
                log.warning("failed to read %s: %s", item.path, exc)
                continue

            self.stats.packets_read += len(df)
            df = df.dropna(subset=[c_ts, c_size, c_src])
            if df.empty:
                continue

            ts = pd.to_numeric(df[c_ts], errors="coerce").to_numpy(dtype=np.float64)
            size = pd.to_numeric(df[c_size], errors="coerce").to_numpy(dtype=np.float64)
            sa = df[c_src].to_numpy()
            ta = df[c_ta].to_numpy() if c_ta in df else sa
            da = df[c_dst].to_numpy() if c_dst in df else sa

            ok = np.isfinite(ts) & np.isfinite(size)
            ts, size, sa, ta, da = ts[ok], size[ok], sa[ok], ta[ok], da[ok]
            if size.size < min_packets:
                self.stats.flows_dropped_short += 1
                continue

            client = self._infer_client_mac(sa, ta)
            directions = np.where(sa == client, FORWARD, BACKWARD).astype(np.int8)
            peer_candidates = [m for m in set(sa.tolist()) | set(da.tolist()) if m != client]
            peer = peer_candidates[0] if peer_candidates else ""

            self.stats.packets_used += int(size.size)
            self.stats.flows_built += 1
            flows.append(
                Flow(
                    flow_id=f"{item.path.parent.parent.name}/{item.path.parent.name}/{item.path.stem}",
                    five_tuple=FiveTuple(client, 0, peer, 0, "wlan"),
                    timestamps=ts,
                    sizes=size.astype(np.int32),
                    directions=directions,
                    dataset=self.name,
                    label=str(item.label_fields.get(self.default_target, "")),
                    label_fields=dict(item.label_fields),
                    meta={**item.meta, "client_mac": client},
                )
            )
        flows.sort(key=lambda f: f.flow_id)
        log.info("[%s] built %d flows from %d CSVs", self.name, len(flows), len(items))
        return flows

    @staticmethod
    def _infer_client_mac(sa: np.ndarray, ta: np.ndarray) -> str:
        """The station is whichever MAC dominates the self-transmitted (sa == ta) frames."""
        self_tx = collections.Counter(sa[sa == ta].tolist())
        if self_tx:
            return self_tx.most_common(1)[0][0]
        return collections.Counter(sa.tolist()).most_common(1)[0][0]
