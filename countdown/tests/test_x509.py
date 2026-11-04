"""Phase 7, task 7.1 -- certificate-chain analysis from TLS 1.2 handshakes.

Classical certificates come from ``cryptography`` so the DER is exactly what a CA emits; the
post-quantum ones (which ``cryptography`` cannot sign) are assembled with a tiny DER encoder,
since the parser only needs the AlgorithmIdentifiers to be right, not the signature.
"""

from __future__ import annotations

import datetime as dt
import struct

import pytest
from cryptography import x509 as cx509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509.oid import NameOID

from countdown.crypto import enrich_flow
from countdown.crypto.tls_parser import parse_tls_handshake
from countdown.crypto.x509 import (
    auth_verdict,
    decode_oid,
    parse_certificate,
    parse_certificate_message,
)
from tests.test_tls_parser import client_hello, msg, rec, server_hello, vec

# -- certificates -----------------------------------------------------------------------------


def _name(cn: str) -> cx509.Name:
    return cx509.Name([cx509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_cert(key, subject="leaf.example", issuer_key=None, issuer="leaf.example",
              hash_alg=hashes.SHA256()) -> bytes:
    now = dt.datetime(2026, 1, 1)
    signer = issuer_key or key
    b = (cx509.CertificateBuilder().subject_name(_name(subject)).issuer_name(_name(issuer))
         .public_key(key.public_key()).serial_number(7)
         .not_valid_before(now).not_valid_after(now + dt.timedelta(days=90)))
    alg = None if isinstance(signer, ed25519.Ed25519PrivateKey) else hash_alg
    return b.sign(signer, alg).public_bytes(Encoding.DER)


@pytest.fixture(scope="module")
def rsa_ca():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


# -- a minimal DER encoder for the post-quantum certificates ---------------------------------

def _der(tag: int, body: bytes) -> bytes:
    n = len(body)
    if n < 0x80:
        return bytes([tag, n]) + body
    ln = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([tag, 0x80 | len(ln)]) + ln + body


def _oid(dotted: str) -> bytes:
    parts = [int(p) for p in dotted.split(".")]
    out = bytearray([40 * parts[0] + parts[1]])
    for p in parts[2:]:
        chunk = [p & 0x7F]
        p >>= 7
        while p:
            chunk.append(0x80 | (p & 0x7F))
            p >>= 7
        out += bytes(reversed(chunk))
    return _der(0x06, bytes(out))


def _seq(*kids: bytes) -> bytes:
    return _der(0x30, b"".join(kids))


def pq_cert(oid: str, key_len: int = 1952, subject=b"pq.example", issuer=b"pq.example") -> bytes:
    alg = _seq(_oid(oid))
    name = lambda cn: _seq(_der(0x31, _seq(_oid("2.5.4.3"), _der(0x0C, cn))))  # noqa: E731
    tbs = _seq(_der(0xA0, _der(0x02, b"\x02")), _der(0x02, b"\x07"), alg, name(issuer),
               _seq(_der(0x17, b"260101000000Z"), _der(0x17, b"270101000000Z")), name(subject),
               _seq(alg, _der(0x03, b"\x00" + b"\x42" * key_len)))
    return _seq(tbs, alg, _der(0x03, b"\x00" + b"\x99" * 3309))


ML_DSA_65 = "2.16.840.1.101.3.4.3.18"


# -- TLS 1.2 messages ---------------------------------------------------------------------------

def certificate_msg(*ders: bytes) -> bytes:
    entries = b"".join(vec(d, 3) for d in ders)
    return msg(0x0B, vec(entries, 3))


def ske_ecdhe(group=0x0017, scheme=0x0804) -> bytes:
    body = b"\x03" + struct.pack(">H", group) + vec(b"\x04" + b"\x55" * 64, 1)
    return msg(0x0C, body + struct.pack(">H", scheme) + vec(b"\x77" * 256, 2))


def server_flight_12(*ders: bytes, suite=0xC02F, ske=True) -> bytes:
    sh = server_hello(suite=suite, tls13=False, key_share=False)
    rest = certificate_msg(*ders) + (ske_ecdhe() if ske else b"") + msg(0x0E, b"")
    return sh + rec(rest)


# -- DER / certificate parsing ----------------------------------------------------------------

def test_decode_oid():
    assert decode_oid(bytes.fromhex("2a864886f70d010101")) == "1.2.840.113549.1.1.1"
    assert decode_oid(_oid(ML_DSA_65)[2:]) == ML_DSA_65


def test_rsa_leaf(rsa_ca):
    info = parse_certificate(make_cert(rsa_ca))
    assert info.key_detail == "RSA-2048" and info.key_bits == 2048
    assert info.key_family == "classical"
    assert info.sig_alg == "sha256WithRSAEncryption" and info.sig_family == "classical"
    assert info.self_signed is True and not info.truncated


def test_ec_leaf_signed_by_rsa_ca(rsa_ca):
    leaf = ec.generate_private_key(ec.SECP384R1())
    info = parse_certificate(make_cert(leaf, issuer_key=rsa_ca, issuer="ca.example",
                                       hash_alg=hashes.SHA384()))
    assert info.key_detail == "secp384r1"
    assert info.sig_alg == "sha384WithRSAEncryption"
    assert info.self_signed is False


def test_ed25519_leaf():
    info = parse_certificate(make_cert(ed25519.Ed25519PrivateKey.generate()))
    assert info.key_alg == "Ed25519" and info.sig_alg == "Ed25519"


def test_ml_dsa_leaf_is_post_quantum():
    info = parse_certificate(pq_cert(ML_DSA_65))
    assert info.key_alg == "ML-DSA-65" and info.key_family == "pqc"
    assert info.sig_alg == "ML-DSA-65" and info.sig_family == "pqc"


def test_unregistered_oid_is_unknown_not_classical():
    info = parse_certificate(pq_cert("1.3.6.1.4.1.99999.1"))
    assert info.key_family == "unknown"
    assert info.key_alg == "unknown(1.3.6.1.4.1.99999.1)"


def test_cut_certificate_keeps_what_precedes_the_cut(rsa_ca):
    der = make_cert(rsa_ca)
    info = parse_certificate(der[:len(der) - 200])     # cut inside the signature value
    assert info.truncated and info.key_detail == "RSA-2048"
    info = parse_certificate(der[:40])                 # cut before the key
    assert info.truncated and info.key_alg is None and info.key_family == "unknown"


def test_garbage_is_not_a_chain():
    certs, _ = parse_certificate_message(b"\x00\x00\x18" + b"\x00\x00\x15" + b"\x87" * 21)
    assert certs == []


# -- through the TLS parser -------------------------------------------------------------------

def test_tls12_chain_and_ske_signature(rsa_ca):
    leaf = ec.generate_private_key(ec.SECP256R1())
    chain = (make_cert(leaf, issuer_key=rsa_ca, issuer="ca.example"),
             make_cert(rsa_ca, subject="ca.example", issuer="ca.example"))
    hs = parse_tls_handshake(None, server_flight_12(*chain))
    assert [c.key_detail for c in hs.server_certs] == ["secp256r1", "RSA-2048"]
    assert hs.server_kex_group == 0x0017 and hs.ske_sig_scheme == 0x0804
    v = auth_verdict(hs)
    assert (v.label, v.basis, v.leaf_key) == ("classical", "certificate", "secp256r1")
    assert v.ske_signature == "rsa_pss_rsae_sha256"
    assert not v.rsa_key_transport and not v.chain_truncated


def test_tls12_pq_chain():
    hs = parse_tls_handshake(None, server_flight_12(pq_cert(ML_DSA_65)))
    v = auth_verdict(hs)
    assert v.label == "pqc" and v.leaf_key == "ML-DSA-65"


def test_cut_chain_is_flagged(rsa_ca):
    flight = server_flight_12(make_cert(rsa_ca), make_cert(rsa_ca), ske=False)
    hs = parse_tls_handshake(None, flight[:len(flight) - 600])
    v = auth_verdict(hs)
    assert v.label == "classical" and v.chain_truncated


def test_rsa_key_transport_is_flagged(rsa_ca):
    hs = parse_tls_handshake(None, server_flight_12(make_cert(rsa_ca), suite=0x009C, ske=False))
    v = auth_verdict(hs)
    assert v.rsa_key_transport and v.leaf_key == "RSA-2048"


def test_certificate_request_means_client_auth(rsa_ca):
    creq = msg(0x0D, vec(b"\x01\x40", 1) + vec(struct.pack(">HH", 0x0401, 0x0403), 2) + vec(b"", 2))
    sh = server_hello(suite=0xC02F, tls13=False, key_share=False)
    hs = parse_tls_handshake(None, sh + rec(certificate_msg(make_cert(rsa_ca)) + creq))
    assert hs.cert_request_sig_algs == [0x0401, 0x0403]
    assert auth_verdict(hs).client_auth_requested


def test_encrypted_finished_after_resumption_is_not_a_chain():
    # Abbreviated handshake: ServerHello, CCS, then the encrypted Finished in a 0x16 record.
    sh = server_hello(suite=0xC02F, tls13=False, key_share=False)
    fake = bytes([0x0B]) + (28).to_bytes(3, "big") + bytes.fromhex("874b8a51355eb704") + b"\x11" * 20
    hs = parse_tls_handshake(None, sh + rec(b"\x01", ctype=0x14) + rec(fake))
    assert hs.server_certs == []
    assert auth_verdict(hs).label == "not_observable"


def test_tls13_authentication_is_not_observable():
    hs = parse_tls_handshake(client_hello(), server_hello(group=0x11EC))
    v = auth_verdict(hs)
    assert v.label == "not_observable" and v.basis == "none"
    assert "TLS 1.3" in v.reason


def test_enrich_flow_carries_auth(rsa_ca):
    import numpy as np

    from countdown.schema import FiveTuple, Flow

    f = Flow(flow_id="t", five_tuple=FiveTuple("10.0.0.1", "10.0.0.2", 50000, 443, "tcp"),
             timestamps=np.array([0.0]), sizes=np.array([100]), directions=np.array([1]),
             handshake_client_bytes=None,
             handshake_server_bytes=server_flight_12(make_cert(rsa_ca)))
    enrich_flow(f)
    assert f.meta["crypto"]["auth"]["label"] == "classical"
    assert f.meta["crypto"]["auth"]["leaf_key"] == "RSA-2048"
