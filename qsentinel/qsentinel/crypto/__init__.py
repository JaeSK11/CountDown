"""Crypto detection and quantum-resistance verdict -- pipeline stages 2 and 3 (Phase 2).

    >>> from qsentinel.crypto import parse_handshake, pqc_verdict
    >>> hs = parse_handshake(flow)
    >>> pqc_verdict(hs).label
    'classical'

Stage 2 (``parse_handshake``) turns the bytes Phase 1 reassembled into cryptographic
parameters; stage 3 (``pqc_verdict``) classifies them.  Both are deterministic: the
negotiated key-exchange group is in the cleartext ServerHello, so this is parsing and a
table lookup, never inference.  The ML pipeline starts at stage 4 (Phase 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qsentinel.crypto import kem_registry, sigalg_registry
from qsentinel.crypto.kem_registry import KemGroup
from qsentinel.crypto.pqc_verdict import (
    BASIS_NEGOTIATED,
    BASIS_NONE,
    BASIS_OFFERED,
    LABEL_CLASSICAL,
    LABEL_HYBRID,
    LABEL_NOT_OBSERVABLE,
    LABEL_PQC,
    LABEL_UNKNOWN,
    TUNNELED_HINTS,
    VERDICT_LABELS,
    Verdict,
    pqc_verdict,
)
from qsentinel.crypto.tls_parser import Handshake, parse_tls_handshake

if TYPE_CHECKING:  # pragma: no cover
    from qsentinel.schema import Flow

__all__ = [
    "parse_handshake", "parse_tls_handshake", "pqc_verdict", "enrich_flow",
    "Handshake", "Verdict", "KemGroup", "kem_registry", "sigalg_registry",
    "VERDICT_LABELS", "TUNNELED_HINTS",
    "LABEL_CLASSICAL", "LABEL_HYBRID", "LABEL_PQC", "LABEL_NOT_OBSERVABLE", "LABEL_UNKNOWN",
    "BASIS_NEGOTIATED", "BASIS_OFFERED", "BASIS_NONE",
]


def parse_handshake(flow: "Flow") -> Handshake:
    """Parse a flow's handshake window into cryptographic parameters (stage 2).

    Dispatches on transport: QUIC carries its ClientHello inside encrypted-but-not-secret
    Initial packets and needs key derivation before any TLS parsing can happen, whereas
    TCP hands us the TLS record layer directly.

    The flow's capture context (``l7_hint``, ``handshake_incomplete``) is stamped onto the
    result so that ``pqc_verdict`` can tell "no PQC" apart from "could not see".
    """
    client = flow.handshake_client_bytes or b""
    server = flow.handshake_server_bytes or b""

    if _is_quic(flow, client, server):
        from qsentinel.crypto.quic_parser import parse_quic_handshake  # lazy: heavy crypto dep
        hs = parse_quic_handshake(client, server)
    else:
        hs = parse_tls_handshake(client, server)

    hs.l7_hint = flow.l7_hint
    hs.handshake_incomplete = bool(flow.handshake_incomplete)
    hs.had_bytes = bool(client or server)
    return hs


def _is_quic(flow: "Flow", client: bytes, server: bytes) -> bool:
    """QUIC iff Phase 1 said so, or the window itself opens on a QUIC long header.

    The wire check is the fallback for flows whose hint was overridden by dataset context
    (e.g. a ``vpn`` context hint on a capture that is really plain QUIC).
    """
    if flow.l7_hint == "quic":
        return True
    if flow.l4_proto != "udp":
        return False
    from qsentinel.flows.handshake import quic_long_header
    return quic_long_header(client)[0] or quic_long_header(server)[0]


def enrich_flow(flow: "Flow") -> Verdict:
    """Run stages 2-3 and write the crypto fields onto ``flow`` in place (task 2.7).

    Populates the four fields ``schema.Flow`` reserved for Phase 2 -- ``tls_version``,
    ``cipher_suite``, ``kem_group``, ``pqc_verdict`` -- and puts the fuller detail
    (basis, confidence, offered groups, SNI, ALPN, evidence) under ``flow.meta['crypto']``
    so the flow schema stays stable while the evidence stays auditable.
    """
    hs = parse_handshake(flow)
    verdict = pqc_verdict(hs)

    flow.tls_version = hs.tls_version
    flow.cipher_suite = hs.cipher_suite_name
    flow.kem_group = verdict.group_name
    flow.pqc_verdict = verdict.label

    flow.meta["crypto"] = {
        "basis": verdict.basis,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "group_codepoint": verdict.group,
        "negotiated_group": hs.negotiated_group_name,
        "offered_groups": [kem_registry.name_of(g) for g in hs.offered_groups],
        "client_key_share_groups": [kem_registry.name_of(g) for g in hs.client_key_share_groups],
        "sni": hs.sni,
        "alpn": hs.alpn,
        "ech": hs.ech,
        "transport": hs.transport,
        "side": hs.side,
        "parse_ok": hs.parse_ok,
        "truncated": hs.truncated,
        "kex_family": hs.kex_family,
        "evidence": verdict.evidence,
    }
    return verdict
