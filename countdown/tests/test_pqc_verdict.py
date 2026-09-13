"""Phase 2, tasks 2.6-2.7 -- the verdict engine and flow enrichment.

The unit tests below are mostly about the distinctions the engine exists to preserve.  The
easy part of this phase is mapping a codepoint to a family; the part that decides whether
the tool can be trusted is never letting "we could not see the handshake" turn into "this
traffic is not protected".  So most of what is asserted here is about the *absence* of a
claim: tunnelled flows, truncated windows and client-only captures each produce a
specifically shaped non-answer.

The three anchor cases from PHASE-2.md sec. 8 are exercised at the end against real
captures, and skip cleanly when the corpus is not present.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from countdown.crypto import enrich_flow, parse_handshake, pqc_verdict
from countdown.crypto.pqc_verdict import (
    BASIS_NEGOTIATED,
    BASIS_NONE,
    BASIS_OFFERED,
    LABEL_CLASSICAL,
    LABEL_HYBRID,
    LABEL_NOT_OBSERVABLE,
    LABEL_PQC,
    LABEL_UNKNOWN,
    VERDICT_LABELS,
)
from countdown.crypto.tls_parser import Handshake
from countdown.schema import FiveTuple, Flow

REPO = Path(__file__).resolve().parent.parent


def hs(**kw) -> Handshake:
    """A Handshake that looks like it came from a real capture unless told otherwise."""
    base = dict(had_bytes=True, saw_server_hello=True, parse_ok=True, tls_version="1.3")
    base.update(kw)
    return Handshake(**base)


# -- the four verdict families ------------------------------------------------------------

@pytest.mark.parametrize("group,label", [
    (0x001D, LABEL_CLASSICAL),   # x25519
    (0x0017, LABEL_CLASSICAL),   # secp256r1
    (0x0100, LABEL_CLASSICAL),   # ffdhe2048
    (0x11EC, LABEL_HYBRID),      # X25519MLKEM768
    (0x6399, LABEL_HYBRID),      # X25519Kyber768Draft00
    (0x0201, LABEL_PQC),         # MLKEM768
])
def test_negotiated_group_drives_the_label(group, label):
    v = pqc_verdict(hs(negotiated_group=group))
    assert v.label == label
    assert v.basis == BASIS_NEGOTIATED
    assert v.group == group
    assert v.confidence == 1.0
    assert v.evidence["ext"] == "key_share"
    assert v.evidence["codepoint"] == f"0x{group:04X}"


def test_every_label_is_declared():
    for group in (0x001D, 0x11EC, 0x0201):
        assert pqc_verdict(hs(negotiated_group=group)).label in VERDICT_LABELS


def test_quantum_resistance_requires_a_negotiated_basis():
    assert pqc_verdict(hs(negotiated_group=0x11EC)).is_quantum_resistant
    assert not pqc_verdict(hs(negotiated_group=0x001D)).is_quantum_resistant
    # Offered-only, even for a PQC group, is not a protection claim.
    offered = pqc_verdict(hs(saw_server_hello=False, saw_client_hello=True,
                             offered_groups=[0x11EC]))
    assert offered.label == LABEL_HYBRID
    assert not offered.is_quantum_resistant


def test_at_risk_is_only_confirmed_classical():
    assert pqc_verdict(hs(negotiated_group=0x001D)).is_at_risk
    assert not pqc_verdict(hs(negotiated_group=0x11EC)).is_at_risk
    assert not pqc_verdict(hs(l7_hint="tor")).is_at_risk


# -- not_observable: statements about the capture, not the cryptography -----------------------

@pytest.mark.parametrize("hint", ["tor", "vpn", "tunnel_openvpn", "tunnel_ike", "tunnel_wireguard"])
def test_tunnelled_flows_make_no_pqc_claim(hint):
    """Invariant carried from Phase 1: only the outer handshake is visible."""
    v = pqc_verdict(hs(l7_hint=hint, negotiated_group=0x001D))
    assert v.label == LABEL_NOT_OBSERVABLE
    assert v.basis == BASIS_NONE
    assert v.confidence == 0.0
    assert v.group is None
    assert hint in v.reason


def test_no_bytes_is_not_observable():
    v = pqc_verdict(Handshake(had_bytes=False))
    assert v.label == LABEL_NOT_OBSERVABLE


def test_records_but_no_hello_is_not_observable():
    """A mid-session capture: the exchange happened before the window opened."""
    v = pqc_verdict(Handshake(had_bytes=True, saw_client_hello=False, saw_server_hello=False))
    assert v.label == LABEL_NOT_OBSERVABLE
    assert v.label != LABEL_CLASSICAL


def test_incomplete_window_with_no_group_is_not_observable():
    """Phase 1 flagged the window as untrustworthy -- a parse failure there is a gap in the
    capture, not a finding that the flow is unprotected."""
    v = pqc_verdict(hs(negotiated_group=None, handshake_incomplete=True))
    assert v.label == LABEL_NOT_OBSERVABLE


def test_ech_without_a_server_hello_is_not_observable():
    v = pqc_verdict(hs(saw_server_hello=False, saw_client_hello=True, ech=True))
    assert v.label == LABEL_NOT_OBSERVABLE
    assert "Encrypted ClientHello" in v.reason


def test_ech_with_a_server_hello_still_reads_the_selection():
    """The outer ServerHello's key_share is the real selection even when the inner CH is
    encrypted -- ECH hides the client's request, not the server's answer."""
    v = pqc_verdict(hs(ech=True, negotiated_group=0x11EC))
    assert v.label == LABEL_HYBRID
    assert v.basis == BASIS_NEGOTIATED


# -- unknown vs classical ------------------------------------------------------------------------

def test_unrecognised_codepoint_is_unknown_not_classical():
    """A codepoint IANA assigns tomorrow must not be reported as breakable today."""
    v = pqc_verdict(hs(negotiated_group=0x1234))
    assert v.label == LABEL_UNKNOWN
    assert v.label != LABEL_CLASSICAL
    assert v.confidence == 0.0
    assert "not in the registry" in v.reason


def test_hello_with_no_group_at_all_is_unknown():
    v = pqc_verdict(hs(negotiated_group=None, truncated=False, handshake_incomplete=False))
    assert v.label == LABEL_UNKNOWN


# -- offered basis is explicitly weaker --------------------------------------------------------------

def test_client_only_capture_uses_offered_basis():
    v = pqc_verdict(hs(saw_server_hello=False, saw_client_hello=True,
                       offered_groups=[0x001D, 0x11EC, 0x0017]))
    assert v.label == LABEL_HYBRID       # the ceiling of what the client would accept
    assert v.basis == BASIS_OFFERED
    assert v.confidence < 1.0
    assert "ceiling" in v.reason


def test_offered_basis_prefers_real_key_shares_over_the_wish_list():
    """A key_share is computation the client actually spent; supported_groups is a list."""
    v = pqc_verdict(hs(saw_server_hello=False, saw_client_hello=True,
                       offered_groups=[0x11EC, 0x001D],
                       client_key_share_groups=[0x001D]))
    assert v.basis == BASIS_OFFERED
    assert v.group == 0x001D
    assert v.evidence["ext"] == "key_share"


def test_truncation_lowers_confidence_without_changing_the_label():
    full = pqc_verdict(hs(negotiated_group=0x11EC))
    cut = pqc_verdict(hs(negotiated_group=0x11EC, truncated=True))
    assert full.label == cut.label == LABEL_HYBRID
    assert cut.confidence < full.confidence


# -- HelloRetryRequest and TLS 1.2 -------------------------------------------------------------------

def test_hello_retry_request_only_is_a_server_selection():
    v = pqc_verdict(hs(hrr_selected_group=0x0017, saw_hello_retry_request=True))
    assert v.label == LABEL_CLASSICAL
    assert v.basis == BASIS_NEGOTIATED
    assert v.confidence < 1.0            # the completing ServerHello was not seen
    assert v.evidence["hello_retry_request"] is True


def test_tls12_server_key_exchange_curve():
    v = pqc_verdict(hs(tls_version="1.2", server_kex_group=0x0017, kex_family="ECDHE"))
    assert v.label == LABEL_CLASSICAL
    assert v.basis == BASIS_NEGOTIATED
    assert v.evidence["ext"] == "server_key_exchange"


@pytest.mark.parametrize("family", ["RSA", "DHE"])
def test_tls12_without_a_named_group_is_still_classical(family):
    """RSA key transport and finite-field DH have no PQC option: the *suite* settles it."""
    v = pqc_verdict(hs(tls_version="1.2", kex_family=family, cipher_suite=0x009C))
    assert v.label == LABEL_CLASSICAL
    assert v.basis == BASIS_NEGOTIATED
    assert v.group is None


# -- evidence is an audit trail -------------------------------------------------------------------------

def test_evidence_lets_a_verdict_be_rederived_by_hand():
    v = pqc_verdict(hs(negotiated_group=0x11EC, cipher_suite=0x1301,
                       sni="example.com", alpn=["h2"], sig_algs=[0x0403, 0x0904]))
    ev = v.evidence
    assert ev["codepoint"] == "0x11EC"
    assert ev["ext"] == "key_share"
    assert ev["tls_version"] == "1.3"
    assert ev["sni"] == "example.com"
    assert ev["algo"] == "X25519+ML-KEM-768"
    assert ev["nist_level"] == 3
    # PQC signature advertisement is recorded as context, never as a verdict input.
    assert ev["signature_algorithms"]["advertises_pqc_signatures"] is True
    assert ev["signature_algorithms"]["observable"] is False


def test_verdict_never_raises_on_a_degenerate_handshake():
    for h in (Handshake(), hs(negotiated_group=None), hs(negotiated_group=0),
              hs(offered_groups=[]), Handshake(had_bytes=True, saw_client_hello=True)):
        assert pqc_verdict(h).label in VERDICT_LABELS


# -- task 2.7: flow enrichment ------------------------------------------------------------------------------

def _flow(client: bytes = b"", server: bytes = b"", hint: str = "tls") -> Flow:
    return Flow(
        flow_id="t1",
        five_tuple=FiveTuple("10.0.0.1", 40000, "93.184.216.34", 443, "tcp"),
        timestamps=np.array([0.0, 0.1]), sizes=np.array([100, 200], dtype=np.int32),
        directions=np.array([1, -1], dtype=np.int8),
        l7_hint=hint, handshake_client_bytes=client or None,
        handshake_server_bytes=server or None,
    )


def test_enrich_flow_writes_the_reserved_schema_fields():
    from tests.test_tls_parser import client_hello, server_hello
    flow = _flow(client_hello(groups=(0x11EC, 0x001D)), server_hello(group=0x11EC, suite=0x1302))
    v = enrich_flow(flow)

    assert flow.pqc_verdict == LABEL_HYBRID == v.label
    assert flow.kem_group == "X25519MLKEM768"
    assert flow.tls_version == "1.3"
    assert flow.cipher_suite == "TLS_AES_256_GCM_SHA384"

    crypto = flow.meta["crypto"]
    assert crypto["basis"] == BASIS_NEGOTIATED
    assert crypto["confidence"] == 1.0
    assert crypto["offered_groups"] == ["X25519MLKEM768", "x25519"]
    assert crypto["transport"] == "tls"


def test_enrich_flow_on_a_tunnel_makes_no_claim():
    flow = _flow(b"\x00" * 100, b"\x00" * 100, hint="tor")
    enrich_flow(flow)
    assert flow.pqc_verdict == LABEL_NOT_OBSERVABLE
    assert flow.kem_group is None


def test_enrich_flow_on_an_empty_flow():
    flow = _flow()
    enrich_flow(flow)
    assert flow.pqc_verdict == LABEL_NOT_OBSERVABLE
    assert flow.tls_version is None


# -- the three anchor cases from PHASE-2.md sec. 8 ---------------------------------------------------------

def _first_flow(pcap: str, **kw):
    from countdown.flows import Reassembler
    from countdown.sources import PcapSource
    for f in Reassembler(**kw).run(PcapSource(pcap)):
        yield f


def _one(pattern: str, **kw):
    hits = sorted(glob.glob(str(REPO / pattern)))
    if not hits:
        pytest.skip(f"corpus not present: {pattern}")
    return hits[0], kw


@pytest.mark.slow
def test_anchor_cstnet_flow_is_classical():
    """PHASE-2.md sec. 8: a direct TLS 1.3 flow from CSTNET classifies as classical."""
    path, kw = _one("data/CSTNET-TLS1.3/extracted/cstnet-tls 1.3/*/1.pcap")
    for flow in _first_flow(path, **kw):
        if flow.l7_hint != "tls":
            continue
        h = parse_handshake(flow)
        assert h.parse_ok
        assert h.tls_version == "1.3"
        assert h.negotiated_group is not None
        assert pqc_verdict(h).label == LABEL_CLASSICAL
        return
    pytest.skip("no TLS flow in the sampled capture")


@pytest.mark.slow
def test_anchor_self_generated_cloudflare_flow_is_hybrid():
    """PHASE-2.md sec. 8: the self-generated capture yields hybrid X25519MLKEM768.

    Produced by scripts/gen_pqc_pcaps.sh; the ServerHello is genuine Cloudflare output.
    """
    path, kw = _one("data/_pqc_synth/hybrid_x25519mlkem768.pcap")
    for flow in _first_flow(path, **kw):
        v = pqc_verdict(parse_handshake(flow))
        assert v.label == LABEL_HYBRID
        assert v.group_name == "X25519MLKEM768"
        assert v.basis == BASIS_NEGOTIATED
        assert v.is_quantum_resistant
        return
    pytest.fail("no flow recovered from the generated capture")


@pytest.mark.slow
def test_anchor_tor_flow_is_not_observable():
    """PHASE-2.md sec. 8: a tunnelled flow makes no PQC claim."""
    path, kw = _one("data/ISCXTor2016/Tor/*.pcap")
    seen = 0
    for flow in _first_flow(path, context_hint="tor"):
        assert pqc_verdict(parse_handshake(flow)).label == LABEL_NOT_OBSERVABLE
        seen += 1
        if seen >= 20:
            break
    assert seen, "no flows in the Tor capture"


@pytest.mark.slow
def test_anchor_offered_pqc_but_negotiated_classical():
    """The case that separates a real verdict engine from one that reads the ClientHello.

    The client offered X25519MLKEM768; the server chose x25519.  Keying off the offer would
    report this connection as protected when it is not.
    """
    path, kw = _one("data/_pqc_synth/downgrade_offered_pqc.pcap")
    for flow in _first_flow(path):
        h = parse_handshake(flow)
        assert 0x11EC in h.offered_groups          # the client did offer it
        v = pqc_verdict(h)
        assert v.label == LABEL_CLASSICAL          # the server did not take it
        assert v.basis == BASIS_NEGOTIATED
        assert not v.is_quantum_resistant
        return
    pytest.fail("no flow recovered from the generated capture")
