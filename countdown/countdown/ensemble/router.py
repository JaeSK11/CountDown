"""Rule-based router: Stage-1 context -> the one tailored ensemble that applies.

``ENSEMBLE-MAP.md``: a flow gets **one** applicable ensemble, not all three.  The decision is
made from deterministic Stage-1 signals, so it needs no training and can be read:

* ``l7_hint`` (``flows/handshake.py``) -- ``tls`` / ``quic`` go to **E2** (app-ID on direct
  TLS); a tunnel or Tor hint goes to **E1** (traffic type); anything unspecific (``tcp``,
  ``udp``, ``opaque``, missing) also goes to E1, the always-available fallback.
* a capture-level ``context`` the wire cannot show -- a Wi-Fi / mobile-app capture goes to
  **E3** (in-app activity).  Context wins over the hint, because "this capture is a phone"
  is a fact about the collection, not a guess from one flow's first bytes.

The crypto verdict (Phase 2) runs beside this, not through it; stopping on a PQC verdict is
Phase 6's pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

DEFAULT_HINT_RULES: dict[str, str] = {
    "tls": "E2", "quic": "E2",
    "tor": "E1", "vpn": "E1",
    "tunnel_openvpn": "E1", "tunnel_ike": "E1", "tunnel_wireguard": "E1",
    "tcp": "E1", "udp": "E1", "opaque": "E1",
}
DEFAULT_CONTEXT_RULES: dict[str, str] = {"mobile": "E3", "wifi": "E3", "mobile_app": "E3"}
FALLBACK = "E1"


class Router:
    def __init__(self, hint_rules: Mapping[str, str] | None = None,
                 context_rules: Mapping[str, str] | None = None,
                 fallback: str = FALLBACK) -> None:
        self.hint_rules = dict(DEFAULT_HINT_RULES if hint_rules is None else hint_rules)
        self.context_rules = dict(DEFAULT_CONTEXT_RULES if context_rules is None else context_rules)
        self.fallback = fallback

    def route(self, l7_hint: str | None = None, context: str | None = None) -> str:
        if context is not None and context in self.context_rules:
            return self.context_rules[context]
        if l7_hint is None:
            return self.fallback
        return self.hint_rules.get(l7_hint, self.fallback)

    def route_flow(self, flow: Any, context: str | None = None) -> str:
        return self.route(getattr(flow, "l7_hint", None), context)

    def route_flows(self, flows: Iterable[Any], context: str | None = None) -> list[str]:
        return [self.route_flow(f, context) for f in flows]

    def report(self, flows: Iterable[Any], expected: str | None = None,
               context: str | None = None) -> dict[str, Any]:
        """Where a dataset's flows go, by hint, and the share reaching ``expected``."""
        flows = list(flows)
        routes = self.route_flows(flows, context)
        by_hint: dict[str, Counter] = {}
        for f, r in zip(flows, routes):
            by_hint.setdefault(str(getattr(f, "l7_hint", None)), Counter())[r] += 1
        out: dict[str, Any] = {
            "n": len(routes), "routes": dict(Counter(routes)),
            "by_hint": {h: dict(c) for h, c in sorted(by_hint.items())},
        }
        if expected is not None and routes:
            out["expected"] = expected
            out["share_expected"] = sum(r == expected for r in routes) / len(routes)
        return out
