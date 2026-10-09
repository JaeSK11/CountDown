"""Phase 5.2: the rule-based router sends each Stage-1 context to its ensemble."""

from __future__ import annotations

from types import SimpleNamespace

from countdown.ensemble.router import Router


def test_hints_route_to_the_expected_expert_and_unknowns_fall_back():
    r = Router()
    assert r.route("tls") == "E2" and r.route("quic") == "E2"
    for h in ("tor", "vpn", "tunnel_openvpn", "tunnel_ike", "tunnel_wireguard"):
        assert r.route(h) == "E1"
    for h in ("tcp", "udp", "opaque", None, "something-new"):
        assert r.route(h) == "E1"                        # the always-available fallback
    assert r.route("tls", context="mobile") == "E3"      # capture context beats the hint


def test_report_counts_routes_per_hint():
    flows = [SimpleNamespace(l7_hint=h) for h in ("tls", "tls", "tor", None)]
    rep = Router().report(flows, expected="E1")
    assert rep["routes"] == {"E2": 2, "E1": 2}
    assert rep["by_hint"]["tls"] == {"E2": 2}
    assert rep["share_expected"] == 0.5
