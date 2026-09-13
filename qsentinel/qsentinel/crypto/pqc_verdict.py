"""Quantum-resistance verdict engine (Phase 2, task 2.6) -- pipeline stage 3.

Takes a parsed ``Handshake`` and answers one question: **if an adversary recorded this
flow today and had a quantum computer tomorrow, could they decrypt it?**

The answer is a lookup, not a prediction, because TLS 1.3 puts the negotiated key-exchange
group in the cleartext ServerHello.  The engine's real work is not classification -- it is
being honest about *what kind of evidence* it had:

``negotiated``  the server's selection.  Authoritative: this is the exchange that happened.
``offered``     only the client's ClientHello survived.  A ceiling, not a fact -- a client
                that offers X25519MLKEM768 still ends up on x25519 if the server declines.

and about the three ways to have no answer at all, which are deliberately kept distinct:

``not_observable``  a tunnel hides the inner handshake, or there was no handshake to read.
                    This is a statement about the *capture*, not about the cryptography.
``unknown``         a handshake was there and we could not parse a group out of it.
``classical``       we read the group and it is breakable.

Collapsing any of those into ``classical`` would manufacture false negatives -- reporting
"this traffic is at risk" about a flow we simply could not see -- which is the failure mode
that would make the whole tool untrustworthy.  So the rule throughout is: **absence of
evidence is never evidence of a classical exchange.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qsentinel.config import get_logger
from qsentinel.crypto import kem_registry, sigalg_registry
from qsentinel.crypto.kem_registry import CLASSICAL, HYBRID, PQC, UNKNOWN
from qsentinel.crypto.tls_parser import Handshake

log = get_logger(__name__)

# -- labels -----------------------------------------------------------------------------

LABEL_CLASSICAL = CLASSICAL
LABEL_HYBRID = HYBRID
LABEL_PQC = PQC
LABEL_UNKNOWN = UNKNOWN
LABEL_NOT_OBSERVABLE = "not_observable"

VERDICT_LABELS: tuple[str, ...] = (
    LABEL_CLASSICAL, LABEL_HYBRID, LABEL_PQC, LABEL_NOT_OBSERVABLE, LABEL_UNKNOWN,
)

BASIS_NEGOTIATED = "negotiated"
BASIS_OFFERED = "offered"
BASIS_NONE = "none"

#: ``l7_hint`` values where only an *outer* handshake is visible.  The inner TLS session --
#: the one whose quantum resistance we would want to judge -- is encapsulated, so any
#: verdict we produced would describe the tunnel, not the traffic inside it.
#: Invariant carried from Phase 1; these flows go straight to Stage 4 with no PQC claim.
TUNNELED_HINTS = frozenset({
    "tor", "vpn", "tunnel_openvpn", "tunnel_ike", "tunnel_wireguard",
})

# -- confidence ---------------------------------------------------------------------------
# Deliberately coarse and hand-set: these are evidence grades, not probabilities, and
# inventing a calibrated number for a deterministic parse would be false precision.
CONF_NEGOTIATED = 1.0        # server's own selection, read from a complete ServerHello
CONF_NEGOTIATED_TRUNCATED = 0.9   # ...but the window was cut; group read before the cut
CONF_HRR = 0.85              # HelloRetryRequest only: server demanded a group, exchange unconfirmed
CONF_TLS12_SKE = 0.95        # TLS 1.2 ServerKeyExchange named_curve
CONF_OFFERED = 0.5           # client's willingness; the server may well have declined
CONF_OFFERED_TRUNCATED = 0.4
CONF_NONE = 0.0


@dataclass(slots=True)
class Verdict:
    """The Stage 3 output for one flow."""

    label: str
    basis: str = BASIS_NONE
    group: int | None = None
    group_name: str | None = None
    confidence: float = CONF_NONE
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def is_quantum_resistant(self) -> bool:
        """True only for an actually-negotiated hybrid/PQC exchange.

        An ``offered`` basis never counts: the client's willingness is not the server's
        selection, and claiming protection the handshake may not have used would be the
        one lie this tool cannot afford.
        """
        return self.label in (LABEL_HYBRID, LABEL_PQC) and self.basis == BASIS_NEGOTIATED

    @property
    def is_at_risk(self) -> bool:
        """True when the flow is confirmed harvest-now-decrypt-later exposed."""
        return self.label == LABEL_CLASSICAL and self.basis == BASIS_NEGOTIATED

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        g = self.group_name or "-"
        return f"{self.label}[{self.basis}] {g} conf={self.confidence:.2f}"


def _evidence(hs: Handshake, **extra: Any) -> dict[str, Any]:
    """The audit trail: enough to re-derive the verdict by hand from the capture."""
    ev: dict[str, Any] = {
        "tls_version": hs.tls_version,
        "cipher_suite": hs.cipher_suite_name,
        "side": hs.side,
        "parse_ok": hs.parse_ok,
        "truncated": hs.truncated,
        "ech": hs.ech,
        "transport": hs.transport,
    }
    if hs.sni:
        ev["sni"] = hs.sni
    if hs.alpn:
        ev["alpn"] = hs.alpn
    ev.update(extra)
    return ev


def _classified(group: int, hs: Handshake, basis: str, confidence: float,
                reason: str, **extra: Any) -> Verdict:
    """Build a verdict from a concrete group codepoint."""
    # Callers may name a different source extension (ServerKeyExchange, HRR key_share).
    extra = {"ext": "key_share", "codepoint": f"0x{group:04X}", **extra}
    family = kem_registry.classify(group)
    if family == kem_registry.GREASE:
        # Cannot happen -- the parser strips GREASE before it reaches here -- but a GREASE
        # value surviving to a verdict would be a silent lie, so refuse it explicitly.
        return Verdict(
            label=LABEL_UNKNOWN, basis=basis, group=group,
            group_name=kem_registry.name_of(group), confidence=CONF_NONE,
            reason="selected group is a GREASE codepoint; not a real exchange",
            evidence=_evidence(hs, **extra),
        )
    entry = kem_registry.lookup(group)
    ev = _evidence(hs, **extra)
    if entry is not None:
        ev["algo"] = entry.algo
        ev["nist_level"] = entry.nist_level
        ev["codepoint_status"] = entry.status
    if hs.sig_algs:
        ev["signature_algorithms"] = sigalg_registry.summarise(hs.sig_algs)
    return Verdict(
        label=family if family != kem_registry.UNKNOWN else LABEL_UNKNOWN,
        basis=basis,
        group=group,
        group_name=kem_registry.name_of(group),
        confidence=confidence if family != kem_registry.UNKNOWN else CONF_NONE,
        reason=reason if family != kem_registry.UNKNOWN
        else f"group 0x{group:04X} is not in the registry; classification withheld",
        evidence=ev,
    )


def pqc_verdict(hs: Handshake) -> Verdict:
    """Classify one parsed handshake.  Never raises; every path yields a labelled verdict."""

    # -- 1. tunnelled: only the outer handshake exists -------------------------------------
    if hs.l7_hint in TUNNELED_HINTS:
        return Verdict(
            label=LABEL_NOT_OBSERVABLE, basis=BASIS_NONE, confidence=CONF_NONE,
            reason=f"tunnelled transport ({hs.l7_hint}); inner handshake is encapsulated",
            evidence=_evidence(hs, l7_hint=hs.l7_hint, tunnelled=True),
        )

    # -- 2. nothing to read ----------------------------------------------------------------
    if not hs.had_bytes:
        return Verdict(
            label=LABEL_NOT_OBSERVABLE, basis=BASIS_NONE, confidence=CONF_NONE,
            reason="no handshake bytes captured for this flow",
            evidence=_evidence(hs, l7_hint=hs.l7_hint),
        )
    if not (hs.saw_client_hello or hs.saw_server_hello):
        # Records were present but no Hello: a mid-session capture, or not TLS at all.
        # Either way the key exchange happened outside the window -- unseen, not classical.
        return Verdict(
            label=LABEL_NOT_OBSERVABLE, basis=BASIS_NONE, confidence=CONF_NONE,
            reason="no ClientHello or ServerHello in the captured window "
                   "(capture began mid-session, or the flow is not TLS)",
            evidence=_evidence(hs, l7_hint=hs.l7_hint, errors=hs.errors[:3]),
        )

    # -- 3. negotiated group: the authoritative answer -------------------------------------
    if hs.negotiated_group is not None:
        conf = CONF_NEGOTIATED_TRUNCATED if hs.truncated else CONF_NEGOTIATED
        return _classified(
            hs.negotiated_group, hs, BASIS_NEGOTIATED, conf,
            "server selected this group in the ServerHello key_share extension",
        )

    # TLS 1.2: the curve is named in the ServerKeyExchange instead.
    if hs.server_kex_group is not None:
        return _classified(
            hs.server_kex_group, hs, BASIS_NEGOTIATED, CONF_TLS12_SKE,
            "server named this curve in the TLS 1.2 ServerKeyExchange",
            ext="server_key_exchange",
        )

    # A HelloRetryRequest is a server *selection* even though the exchange has not completed
    # yet -- the client either uses that group or the connection fails.
    if hs.hrr_selected_group is not None:
        return _classified(
            hs.hrr_selected_group, hs, BASIS_NEGOTIATED, CONF_HRR,
            "server demanded this group in a HelloRetryRequest; "
            "the completing ServerHello was not captured",
            ext="key_share(hello_retry_request)", hello_retry_request=True,
        )

    # -- 4. TLS 1.2 without a readable curve ------------------------------------------------
    # A non-ECDHE 1.2 suite has no named group at all: RSA key transport and static DH are
    # classical by construction, and that is a fact about the suite, not a missing field.
    if hs.saw_server_hello and hs.kex_family in ("RSA", "DHE") and hs.cipher_suite is not None:
        return Verdict(
            label=LABEL_CLASSICAL, basis=BASIS_NEGOTIATED, group=None,
            group_name=None, confidence=CONF_TLS12_SKE,
            reason=f"TLS 1.2 {hs.kex_family} key exchange; no post-quantum option exists "
                   f"for this cipher suite",
            evidence=_evidence(hs, ext="cipher_suite", kex_family=hs.kex_family),
        )

    # -- 5. client side only: offered groups, explicitly weaker --------------------------
    # Prefer the groups the client generated real key shares for over the full
    # supported_groups list: a key_share is what the client actually spent computation on
    # and is the group most likely to be selected.
    offered = hs.client_key_share_groups or hs.offered_groups
    if offered:
        best = kem_registry.best_group(offered)
        if best is not None:
            conf = CONF_OFFERED_TRUNCATED if hs.truncated else CONF_OFFERED
            source = ("key_share" if hs.client_key_share_groups else "supported_groups")
            return _classified(
                best, hs, BASIS_OFFERED, conf,
                f"no ServerHello captured; best group the client offered in {source}. "
                "The server may have selected a weaker group -- this is a ceiling, not "
                "the exchange that happened",
                ext=source,
                offered_groups=[kem_registry.name_of(g) for g in offered],
            )

    # -- 6. a Hello, but no group anywhere in it -----------------------------------------
    if hs.ech and not hs.saw_server_hello:
        return Verdict(
            label=LABEL_NOT_OBSERVABLE, basis=BASIS_NONE, confidence=CONF_NONE,
            reason="Encrypted ClientHello: the real groups are in the encrypted inner "
                   "ClientHello and no ServerHello was captured",
            evidence=_evidence(hs, ech=True),
        )
    if hs.handshake_incomplete or hs.truncated:
        # Phase 1 told us this window cannot be trusted as the start of the stream, or the
        # record ran past it.  A group may well have been negotiated in the bytes we never
        # saw, so this is a gap in the capture -- not a finding about the cryptography.
        return Verdict(
            label=LABEL_NOT_OBSERVABLE, basis=BASIS_NONE, confidence=CONF_NONE,
            reason="handshake window is incomplete or truncated; the key exchange was "
                   "not inside the captured bytes",
            evidence=_evidence(hs, handshake_incomplete=hs.handshake_incomplete,
                               errors=hs.errors[:3]),
        )
    return Verdict(
        label=LABEL_UNKNOWN, basis=BASIS_NONE, confidence=CONF_NONE,
        reason="handshake parsed but no key-exchange group could be recovered",
        evidence=_evidence(hs, errors=hs.errors[:3]),
    )
