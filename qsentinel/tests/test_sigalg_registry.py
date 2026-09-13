"""Phase 2, task 2.4 -- the signature-scheme registry.

Small on purpose.  Signatures are a *context* field, not a verdict input: TLS 1.3 encrypts
CertificateVerify, so what we can see is the client's ``signature_algorithms`` list, which
states what it would accept rather than what was used.  These tests mostly pin that
distinction in place so a later change cannot quietly promote an advertisement into a
protection claim.
"""

from __future__ import annotations

import pytest

from qsentinel.crypto import sigalg_registry as reg


@pytest.mark.parametrize("codepoint,name,family", [
    (0x0403, "ecdsa_secp256r1_sha256", "classical"),
    (0x0804, "rsa_pss_rsae_sha256", "classical"),
    (0x0807, "ed25519", "classical"),
    (0x0401, "rsa_pkcs1_sha256", "classical"),
    (0x0904, "mldsa44", "pqc"),
    (0x0905, "mldsa65", "pqc"),
    (0x0906, "mldsa87", "pqc"),
    (0x0911, "slhdsa_sha2_128s", "pqc"),
    (0x091C, "slhdsa_shake_256f", "pqc"),
])
def test_known_schemes(codepoint, name, family):
    assert reg.classify(codepoint) == family
    assert reg.name_of(codepoint) == name


def test_post_quantum_flag():
    assert reg.is_post_quantum(0x0905)      # ML-DSA-65
    assert reg.is_post_quantum(0x0913)      # SLH-DSA
    assert not reg.is_post_quantum(0x0403)  # ECDSA
    assert not reg.is_post_quantum(0x1234)  # unknown
    assert not reg.is_post_quantum(0x0A0A)  # GREASE


def test_unknown_is_not_classical():
    assert reg.classify(0x1234) == "unknown"
    assert reg.classify(0x1234) != "classical"


def test_grease_is_filtered():
    assert reg.classify(0x0A0A) == "grease"
    assert reg.pq_schemes([0x0A0A, 0x0905, 0xFAFA]) == [0x0905]
    assert not [cp for cp in reg.all_schemes() if reg.classify(cp) == "grease"]


def test_summarise_marks_signatures_unobservable():
    """The whole point of the module: an advertisement is not evidence of use."""
    out = reg.summarise([0x0403, 0x0905, 0x0A0A])
    assert out["advertises_pqc_signatures"] is True
    assert out["pq_offered"] == ["mldsa65"]
    assert out["observable"] is False           # TLS 1.3 encrypts CertificateVerify
    assert "GREASE(0x0A0A)" not in out["offered"]


def test_summarise_with_no_pq():
    out = reg.summarise([0x0403, 0x0804])
    assert out["advertises_pqc_signatures"] is False
    assert out["pq_offered"] == []


def test_table_is_self_consistent():
    table = reg.all_schemes()
    assert len(table) > 30
    for cp, entry in table.items():
        assert entry.codepoint == cp
        assert entry.family in ("classical", "pqc")
        if entry.family == "pqc":
            assert entry.nist_level in (1, 2, 3, 5)


def test_bad_input_does_not_raise():
    for bad in (None, "ed25519", -1, 1 << 40):
        assert reg.classify(bad) in ("unknown", "grease")
        assert reg.lookup(bad) is None
