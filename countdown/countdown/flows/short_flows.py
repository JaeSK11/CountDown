"""What the ``min_packets`` filter throws away, counted by shape (Phase 10, section 2a).

Flows shorter than ``min_packets`` (4) never reach a model.  On an internet-facing host most of
them are scans: a SYN scan is one to three packets per target, and scanners can be 35-70 % of
daily connection attempts (K. Webster, "There is no such thing as a benign internet scanner",
2025).  Dropping them silently would hide that, so each dropped flow is tallied by shape:

``syn_only``      SYN from the initiator, nothing back (filtered port, or a SYN scan's probe)
``syn_rst``       SYN answered by RST, no SYN-ACK (closed port)
``half_open``     SYN / SYN-ACK and then nothing, or a reset -- no payload (SYN or connect scan)
``tcp_payload``   short TCP flow that did carry payload (a tiny real exchange)
``tcp_other``     no SYN, no payload: stray ACKs/RSTs, typically the tail of an idle-split flow
``udp``           short UDP flow (DNS, NTP, a UDP probe -- not separable without payload rules)

The first three are the *scan-like* ones.  Only counts leave this object -- distinct
initiators and responders, and the destination ports most probed -- never an address.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SHAPES = ("syn_only", "syn_rst", "half_open", "tcp_payload", "tcp_other", "udp")
SCAN_LIKE = ("syn_only", "syn_rst", "half_open")


def shape_of(proto: str, syn: bool, synack: bool, rst: bool, payload_pkts: int) -> str:
    if proto != "tcp":
        return "udp"
    if payload_pkts:
        return "tcp_payload"
    if synack:
        return "half_open"
    if syn:
        return "syn_rst" if rst else "syn_only"
    return "tcp_other"


@dataclass
class ShortFlowTally:
    shapes: Counter = field(default_factory=Counter)
    _initiators: set = field(default_factory=set)
    _responders: set = field(default_factory=set)
    _ports: Counter = field(default_factory=Counter)   # destination ports of scan-like flows

    def add(self, shape: str, initiator: str, responder: str, dst_port: int) -> None:
        self.shapes[shape] += 1
        if shape in SCAN_LIKE:
            self._initiators.add(initiator)
            self._responders.add(responder)
            self._ports[int(dst_port)] += 1

    def merge(self, other: "ShortFlowTally") -> None:
        self.shapes.update(other.shapes)
        self._initiators |= other._initiators
        self._responders |= other._responders
        self._ports.update(other._ports)

    @property
    def total(self) -> int:
        return sum(self.shapes.values())

    @property
    def scan_like(self) -> int:
        return sum(self.shapes[s] for s in SCAN_LIKE)

    def as_dict(self, top_ports: int = 10) -> dict[str, Any]:
        return {"dropped_short": self.total, "scan_like": self.scan_like,
                "by_shape": {s: self.shapes[s] for s in SHAPES},
                "scan_initiators": len(self._initiators),
                "scan_targets": len(self._responders),
                "scan_ports_distinct": len(self._ports),
                "scan_top_ports": [[p, n] for p, n in self._ports.most_common(top_ports)]}

    def summary_line(self) -> str:
        if not self.total:
            return "short flows   none dropped"
        d = self.as_dict(top_ports=5)
        s = d["by_shape"]
        line = (f"short flows   {d['dropped_short']} dropped before classification, "
                f"{d['scan_like']} scan-like (SYN-only {s['syn_only']}, SYN->RST {s['syn_rst']}, "
                f"half-open {s['half_open']}); other TCP {s['tcp_payload'] + s['tcp_other']}, "
                f"UDP {s['udp']}")
        if d["scan_like"]:
            ports = ", ".join(f"{p} ({n})" for p, n in d["scan_top_ports"])
            line += (f"\n              scan-like: {d['scan_initiators']} initiator(s) -> "
                     f"{d['scan_targets']} target(s), {d['scan_ports_distinct']} port(s); top {ports}")
        return line
