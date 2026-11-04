"""Certificate-chain analysis from a passively captured handshake (Phase 7, task 7.1).

Before TLS 1.3 the server's ``Certificate`` message is plaintext, so a passive observer can read
the chain the server actually presented: each certificate's public-key algorithm (what an
adversary must break to impersonate the subject -- or, under TLS 1.2 RSA key transport, to
decrypt every recorded session) and the algorithm the issuer signed it with.  In TLS 1.3 and
QUIC the chain is encrypted and this module never runs; the verdict says ``not_observable``.

Why a hand-rolled DER walker rather than ``cryptography.x509``: the window is 8 KB and the
last certificate of a chain is routinely cut, and a library rejects a truncated or unfamiliar
certificate outright (an ML-DSA key it does not implement included).  We need the two
AlgorithmIdentifiers, which sit in the first few hundred bytes, and an honest ``truncated``
flag for the rest.  Nothing here verifies a signature -- the observer has no trust anchor and
no reason to; the question is only *which* algorithms are in use.

As with the KEM verdict, absence of evidence is never classical: an OID not in
``data/x509_algs.yaml`` is ``unknown``, and a chain that was not captured is
``not_observable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from countdown.config import get_logger
from countdown.crypto.kem_registry import PQC, UNKNOWN

log = get_logger(__name__)

DATA_PATH = Path(__file__).resolve().parent / "data" / "x509_algs.yaml"

CLASSICAL = "classical"
NOT_OBSERVABLE = "not_observable"


# -- registry ---------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class X509Alg:
    oid: str
    name: str
    kind: str            # key | signature | both
    family: str          # classical | pqc
    algo: str
    nist_level: int | None = None


@lru_cache(maxsize=1)
def _registry() -> tuple[dict[str, X509Alg], dict[str, str]]:
    with open(DATA_PATH) as fh:
        raw = yaml.safe_load(fh) or {}
    algs: dict[str, X509Alg] = {}
    for row in raw.get("algorithms", []):
        oid = str(row["oid"])
        algs[oid] = X509Alg(oid=oid, name=str(row["name"]), kind=str(row["kind"]),
                            family=str(row["family"]), algo=str(row.get("algo", row["name"])),
                            nist_level=row.get("nist_level"))
    curves = {str(k): str(v) for k, v in (raw.get("curves") or {}).items()}
    if not algs:
        raise RuntimeError(f"x509_algs.yaml at {DATA_PATH} produced an empty registry")
    return algs, curves


def lookup(oid: str | None) -> X509Alg | None:
    return _registry()[0].get(oid) if oid else None


def family_of(oid: str | None) -> str:
    entry = lookup(oid)
    return entry.family if entry is not None else UNKNOWN


def name_of(oid: str | None) -> str:
    entry = lookup(oid)
    if entry is not None:
        return entry.name
    return f"unknown({oid})" if oid else "unknown"


def curve_name(oid: str | None) -> str | None:
    return _registry()[1].get(oid, f"unknown({oid})") if oid else None


# -- DER --------------------------------------------------------------------------------------

class _Short(Exception):
    """The DER element runs past the captured bytes."""


def _tlv(buf: bytes, pos: int) -> tuple[int, int, int]:
    """``(tag, content_start, content_end)`` of the element at ``pos``.  Raises ``_Short``."""
    if pos + 2 > len(buf):
        raise _Short
    tag, first = buf[pos], buf[pos + 1]
    pos += 2
    if first < 0x80:
        length = first
    else:
        n = first & 0x7F
        if n == 0 or n > 4 or pos + n > len(buf):
            raise _Short
        length = int.from_bytes(buf[pos:pos + n], "big")
        pos += n
    if pos + length > len(buf):
        raise _Short
    return tag, pos, pos + length


def _children(buf: bytes, start: int, end: int) -> list[tuple[int, int, int]]:
    out, pos = [], start
    while pos < end:
        tag, s, e = _tlv(buf, pos)
        out.append((tag, s, e))
        pos = e
    return out


def _open(buf: bytes, pos: int) -> tuple[int, int, int, bool]:
    """Like ``_tlv`` but tolerant of a cut: ``(tag, start, end clamped to buf, was_cut)``."""
    try:
        tag, s, e = _tlv(buf, pos)
        return tag, s, e, False
    except _Short:
        if pos + 2 > len(buf):
            raise
        first = buf[pos + 1]
        n = first & 0x7F if first & 0x80 else 0
        if pos + 2 + n > len(buf):
            raise
        return buf[pos], pos + 2 + n, len(buf), True


def _children_upto_cut(buf: bytes, start: int, end: int) -> list[tuple[int, int, int]]:
    """The complete children of a possibly cut constructed element, stopping at the cut."""
    out, pos = [], start
    while pos < end:
        try:
            tag, s, e = _tlv(buf, pos)
        except _Short:
            break
        out.append((tag, s, e))
        pos = e
    return out


def decode_oid(raw: bytes) -> str:
    if not raw:
        return ""
    first = raw[0]
    parts = [min(first // 40, 2), first - 40 * min(first // 40, 2)]
    val = 0
    for b in raw[1:]:
        val = (val << 7) | (b & 0x7F)
        if not b & 0x80:
            parts.append(val)
            val = 0
    return ".".join(map(str, parts))


def _alg_id(buf: bytes, s: int, e: int) -> tuple[str | None, tuple[int, int, int] | None]:
    """AlgorithmIdentifier ``SEQUENCE { OID, params OPTIONAL }`` -> (oid, params tlv)."""
    kids = _children(buf, s, e)
    if not kids or kids[0][0] != 0x06:
        return None, None
    oid = decode_oid(buf[kids[0][1]:kids[0][2]])
    return oid, (kids[1] if len(kids) > 1 else None)


@dataclass(slots=True)
class CertInfo:
    """What one certificate says about the algorithms in use."""

    position: int                      # 0 = leaf
    key_oid: str | None = None
    key_alg: str | None = None         # registry name, e.g. rsaEncryption / ML-DSA-65
    key_family: str = UNKNOWN          # classical | pqc | unknown
    key_detail: str | None = None      # RSA-2048 / secp256r1 / ...
    key_bits: int | None = None
    sig_oid: str | None = None
    sig_alg: str | None = None
    sig_family: str = UNKNOWN
    self_signed: bool | None = None
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"position": self.position, "key": self.key_detail or self.key_alg,
                "key_family": self.key_family, "signature": self.sig_alg,
                "signature_family": self.sig_family, "self_signed": self.self_signed,
                "truncated": self.truncated}


def _rsa_bits(bitstring: bytes) -> int | None:
    """Modulus size from an RSAPublicKey inside a SubjectPublicKeyInfo BIT STRING."""
    try:
        tag, s, e = _tlv(bitstring, 1)  # skip the unused-bits byte
        if tag != 0x30:
            return None
        kids = _children(bitstring, s, e)
        if not kids or kids[0][0] != 0x02:
            return None
        mod = bitstring[kids[0][1]:kids[0][2]].lstrip(b"\x00")
        return len(mod) * 8 if mod else None
    except _Short:
        return None


def _parse_spki(der: bytes, spki: tuple[int, int, int], info: CertInfo) -> None:
    spki_kids = _children(der, spki[1], spki[2])
    if not spki_kids or spki_kids[0][0] != 0x30:
        return
    info.key_oid, params = _alg_id(der, spki_kids[0][1], spki_kids[0][2])
    key_bits = der[spki_kids[1][1]:spki_kids[1][2]] if len(spki_kids) > 1 else b""
    entry = lookup(info.key_oid)
    if entry is not None and entry.algo == "RSA":
        info.key_bits = _rsa_bits(key_bits)
        info.key_detail = f"RSA-{info.key_bits}" if info.key_bits else "RSA"
    elif entry is not None and entry.algo == "EC" and params and params[0] == 0x06:
        info.key_detail = curve_name(decode_oid(der[params[1]:params[2]]))
    elif entry is not None:
        info.key_detail = entry.name


def parse_certificate(der: bytes, position: int = 0) -> CertInfo:
    """The algorithm fields of one DER certificate.  Never raises; a cut certificate yields
    whatever was read before the cut with ``truncated=True``."""
    info = CertInfo(position=position)
    try:
        # Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signatureValue }
        tag, s, e, cut = _open(der, 0)
        if tag != 0x30:
            return info
        tag, ts, te, cut_tbs = _open(der, s)                  # tbsCertificate
        if tag != 0x30:
            return info
    except _Short:
        info.truncated = True
        return info
    info.truncated = cut or cut_tbs
    kids = _children_upto_cut(der, ts, te)
    # tbsCertificate: [0] version?, serial, signature, issuer, validity, subject, spki, ...
    i = 1 if kids and kids[0][0] == 0xA0 else 0
    try:
        if len(kids) > i + 1:
            info.sig_oid = _alg_id(der, kids[i + 1][1], kids[i + 1][2])[0]
        if len(kids) > i + 4:
            issuer, subject = kids[i + 2], kids[i + 4]
            info.self_signed = der[issuer[1]:issuer[2]] == der[subject[1]:subject[2]]
        if len(kids) > i + 5:
            _parse_spki(der, kids[i + 5], info)
    except (_Short, IndexError, ValueError):
        info.truncated = True
    info.key_alg = name_of(info.key_oid) if info.key_oid else None
    info.key_family = family_of(info.key_oid) if info.key_oid else UNKNOWN
    info.sig_alg = name_of(info.sig_oid) if info.sig_oid else None
    info.sig_family = family_of(info.sig_oid) if info.sig_oid else UNKNOWN
    return info


# -- the TLS Certificate message --------------------------------------------------------------

def parse_certificate_message(body: bytes, complete: bool = True) -> tuple[list[CertInfo], bool]:
    """TLS 1.2 ``Certificate``: ``opaque certificate_list<0..2^24-1>`` of ``<1..2^24-1>`` DER
    entries, leaf first.  Returns ``(certs, truncated)``; a cut final entry is parsed as far
    as it goes."""
    certs: list[CertInfo] = []
    truncated = not complete
    if len(body) < 3:
        return certs, True
    total = int.from_bytes(body[:3], "big")
    end = min(3 + total, len(body))
    if 3 + total > len(body):
        truncated = True
    pos = 3
    while pos + 3 <= end:
        n = int.from_bytes(body[pos:pos + 3], "big")
        der = body[pos + 3:pos + 3 + n]
        if n == 0 or not der or der[0] != 0x30:
            # Not a DER SEQUENCE: this is not a certificate list.  A resumed TLS 1.2 session
            # sends its *encrypted* Finished in a 0x16 record after ChangeCipherSpec, and those
            # bytes can pass for a message header; reading on would invent a chain.
            break
        info = parse_certificate(der, position=len(certs))
        if len(der) < n:
            info.truncated = True
            truncated = True
        certs.append(info)
        pos += 3 + n
    return certs, truncated


# -- the authentication verdict ---------------------------------------------------------------

@dataclass(slots=True)
class AuthVerdict:
    """Is the *authentication* of this flow post-quantum?  Kept apart from the KEM verdict:
    a recorded session is decrypted by breaking the key exchange, so authentication never
    moves ``pqc_verdict`` -- except that under RSA key transport the leaf key *is* the key
    exchange (``rsa_key_transport``)."""

    label: str = NOT_OBSERVABLE        # classical | pqc | unknown | not_observable
    basis: str = "none"                # certificate | server_key_exchange | none
    leaf_key: str | None = None
    leaf_key_bits: int | None = None
    leaf_signature: str | None = None
    chain: list[dict[str, Any]] = field(default_factory=list)
    chain_truncated: bool = False
    ske_signature: str | None = None   # TLS 1.2 ServerKeyExchange SignatureScheme
    rsa_key_transport: bool = False
    client_auth_requested: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "basis": self.basis, "leaf_key": self.leaf_key,
                "leaf_key_bits": self.leaf_key_bits, "leaf_signature": self.leaf_signature,
                "chain": self.chain, "chain_truncated": self.chain_truncated,
                "ske_signature": self.ske_signature, "rsa_key_transport": self.rsa_key_transport,
                "client_auth_requested": self.client_auth_requested, "reason": self.reason}


def auth_verdict(hs: Any) -> AuthVerdict:
    """Authentication verdict for a parsed ``Handshake``."""
    from countdown.crypto import sigalg_registry

    v = AuthVerdict(client_auth_requested=bool(getattr(hs, "cert_request_sig_algs", None)))
    if hs.ske_sig_scheme is not None:
        v.ske_signature = sigalg_registry.name_of(hs.ske_sig_scheme)
    v.rsa_key_transport = hs.kex_family == "RSA"
    certs: list[CertInfo] = list(hs.server_certs)
    if certs:
        leaf = certs[0]
        v.basis = "certificate"
        v.leaf_key, v.leaf_key_bits, v.leaf_signature = (leaf.key_detail or leaf.key_alg,
                                                         leaf.key_bits, leaf.sig_alg)
        v.chain = [c.as_dict() for c in certs]
        v.chain_truncated = bool(hs.server_certs_truncated)
        v.label = leaf.key_family
        if leaf.key_family == UNKNOWN:
            v.reason = (f"leaf key algorithm {leaf.key_alg or 'unreadable'} is not in the "
                        "registry; classification withheld")
        elif leaf.key_family == PQC:
            v.reason = f"server certificate key is {leaf.key_alg} (post-quantum)"
        else:
            v.reason = (f"server certificate key is {v.leaf_key}; a quantum adversary could "
                        "forge this server's identity")
        return v
    if v.ske_signature is not None:
        v.basis = "server_key_exchange"
        v.label = sigalg_registry.classify(hs.ske_sig_scheme)
        if v.label not in (CLASSICAL, PQC):
            v.label = UNKNOWN
        v.reason = f"TLS 1.2 ServerKeyExchange signed with {v.ske_signature}; chain not captured"
        return v
    if hs.tls_version == "1.3" or hs.transport == "quic":
        v.reason = "TLS 1.3 / QUIC encrypt the Certificate and CertificateVerify messages"
    else:
        v.reason = "no server Certificate in the captured window"
    return v
