"""Phase 8: the census script's QUIC profile -- trace parsing, transport-aware join, and the
QUIC-vs-TCP comparison.  No network: synthetic s_client output and pipeline records."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pqc_census.py"
_spec = importlib.util.spec_from_file_location("pqc_census", _PATH)
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)


def _trace(server_group: str | None, hrr: str | None = None) -> str:
    """The shape of ``s_client -quic -trace``: a ClientHello offering ML-KEM + x25519, then
    (optionally) a HelloRetryRequest, then the ServerHello."""
    out = ["CONNECTED(00000003)", "Sent TLS Record", "Header:",
           "    ClientHello, Length=1504",
           "        extension_type=key_share(51), length=1222",
           "            NamedGroup: X25519MLKEM768 (4588)",
           "            NamedGroup: ecdh_x25519 (29)"]
    for g in [hrr, server_group]:
        if g is None:
            continue
        out += ["Received TLS Record", "Header:", "  Version = TLS 1.2 (0x303)",
                "    ServerHello, Length=90",
                "      extensions, length = 46",
                "        extension_type=key_share(51), length=36",
                f"            NamedGroup: {g} (29)",
                "        extension_type=supported_versions(43), length=2"]
    out += ["New, TLSv1.3, Cipher is TLS_AES_256_GCM_SHA384", "Protocol: QUICv1",
            "SSL-Session:", "    Protocol  : TLSv1.3", "    Cipher    : TLS_AES_256_GCM_SHA384"]
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("group, want", [("X25519MLKEM768", "X25519MLKEM768"),
                                         ("ecdh_x25519", "x25519"),
                                         ("secp256r1", "secp256r1")])
def test_trace_server_group(group, want):
    assert census._group(_trace(group)) == want


def test_trace_ignores_client_offer_without_server_hello():
    assert census._trace_server_group(_trace(None)) is None


def test_trace_takes_final_server_hello_after_hrr():
    assert census._group(_trace("secp256r1", hrr="secp384r1")) == "secp256r1"


def test_tcp_report_still_wins_over_trace():
    text = "Negotiated TLS1.3 group: X25519MLKEM768\n" + _trace("ecdh_x25519")
    assert census._group(text) == "X25519MLKEM768"


def _probe(site, profile, t, ok=True, group="X25519MLKEM768"):
    return {"site": site, "host": site, "profile": profile, "ok": ok, "protocol": "1.3",
            "cipher": "TLS_AES_128_GCM_SHA256", "group": group if ok else None,
            "leaf_key": None, "t_start": t, "t_end": t + 1.0}


def _rec(sni, proto, start, group, safe):
    return {"sni": sni, "proto": proto, "start": start, "kem_group": group,
            "quantum_safe": safe, "pqc_verdict": "hybrid" if safe else "classical"}


def test_join_matches_transport():
    """A QUIC probe must not take the TCP record of the same host (or vice versa), even
    when both fall inside its window."""
    p_tcp, p_quic = _probe("a.com", "tls13", 10.0), _probe("a.com", "quic", 10.0)
    r_tcp = _rec("a.com", "tcp", 10.2, "x25519", False)
    r_udp = _rec("a.com", "udp", 10.1, "X25519MLKEM768", True)
    census.join([p_tcp, p_quic], [r_udp, r_tcp])
    assert p_tcp["record"] is r_tcp
    assert p_quic["record"] is r_udp


def test_join_no_udp_record_means_none():
    p = _probe("a.com", "quic", 10.0)
    census.join([p], [_rec("a.com", "tcp", 10.1, "x25519", False)])
    assert p["record"] is None


def test_compare_quic_tcp():
    probes = []
    recs = []
    t = 0.0
    # (site, tcp_safe, quic answered?, quic_safe)
    for site, tcp_safe, quic_ok, quic_safe in [("both.com", True, True, True),
                                               ("tcponly.com", True, True, False),
                                               ("noquic.com", False, False, None),
                                               ("neither.com", False, True, False)]:
        g_t = "X25519MLKEM768" if tcp_safe else "x25519"
        probes.append(_probe(site, "tls13", t, group=g_t))
        recs.append(_rec(site, "tcp", t + 0.1, g_t, tcp_safe))
        t += 5
        g_q = "X25519MLKEM768" if quic_safe else "x25519"
        probes.append(_probe(site, "quic", t, ok=quic_ok, group=g_q))
        if quic_ok:
            recs.append(_rec(site, "udp", t + 0.1, g_q, quic_safe))
        t += 5
    census.join(probes, recs)
    res, md = census.compare_quic_tcp(probes)
    assert res["tcp_ok"] == 4 and res["quic_ok"] == 3 and res["both_captured"] == 3
    assert (res["safe_tcp_and_quic"], res["safe_tcp_only"], res["safe_quic_only"],
            res["safe_neither"]) == (1, 1, 0, 1)
    assert res["quantum_safe_share_tcp_on_common"] == pytest.approx(2 / 3, abs=1e-4)
    assert res["quantum_safe_share_quic_on_common"] == pytest.approx(1 / 3, abs=1e-4)
    assert [d["site"] for d in res["group_differs"]] == ["tcponly.com"]
    assert res["quic_share_of_tcp_sites"] == 0.75

    out, text = census.summarise(probes, {"sites": 4, "tranco_list": "t", "date": "d",
                                          "openssl": "3.5", "records_captured": len(recs)})
    assert set(out["profiles"]) == {"tls13", "quic"}
    assert "## QUIC vs TCP" in text and "tcponly.com" in text


def test_unread_server_reply_is_not_counted_wrong():
    """yandex.ru over QUIC before the Retry fix: the client saw x25519, the record (basis
    "offered") carries the client's first offer.  That is "not read", not "wrong"."""
    p = _probe("y.ru", "quic", 0.0, group="x25519")
    r = _rec("y.ru", "udp", 0.1, "X25519MLKEM768", False)
    r["pqc_basis"] = "offered"
    census.join([p], [r])
    out, text = census.summarise([p], {"sites": 1, "tranco_list": "t", "date": "d",
                                       "openssl": "3.5", "records_captured": 1})
    g = out["profiles"]["quic"]["passive_group_vs_client"]
    assert (g["match"], g["wrong"], g["server_reply_not_read"]) == (0, 0, 1)
    assert "server reply not read" in text
