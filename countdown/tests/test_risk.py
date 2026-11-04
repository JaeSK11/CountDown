"""Phase 7, task 7.2 -- harvest-now-decrypt-later score and mitigation codes."""

from __future__ import annotations

import pytest

from countdown.risk import (
    MITIGATIONS,
    UNASSESSED,
    assess,
    level_of,
    load_risk_config,
    summarise,
    volume_factor,
)


def rec(verdict="classical", basis="negotiated", tls="1.3", nbytes=10_000_000, listed=False,
        cipher="TLS_AES_128_GCM_SHA256", l7="tls"):
    return {"pqc_verdict": verdict, "pqc_basis": basis, "tls_version": tls, "bytes": nbytes,
            "pqc_listed": listed, "cipher": cipher, "l7": l7}


def test_quantum_safe_scores_zero_and_needs_nothing():
    out = assess(rec("hybrid"), tier="migrate_first")
    assert out == {"hndl_score": 0.0, "hndl_level": "none", "mitigations": []}


def test_unseen_handshake_is_unassessed_not_scored():
    out = assess(rec("not_observable", "none", tls=None, l7="tor"))
    assert out["hndl_score"] is None and out["hndl_level"] == UNASSESSED
    assert out["mitigations"] == ["assess_tunnel"]
    assert assess(rec("not_observable", "none", tls=None, l7="tcp"))["mitigations"] == ["capture_handshake"]
    assert assess(rec("unknown", "none"))["hndl_score"] is None


def test_score_orders_by_tier_and_volume():
    first = assess(rec(), tier="migrate_first")["hndl_score"]
    low = assess(rec(), tier="low_priority")["hndl_score"]
    small = assess(rec(nbytes=500), tier="migrate_first")["hndl_score"]
    assert first == 100.0 and low == 30.0
    assert 50.0 <= small < first
    assert assess(rec(), tier="migrate_first")["hndl_level"] == "high"


def test_server_side_fix_when_client_already_offers_pq():
    assert assess(rec(listed=True))["mitigations"] == ["enable_hybrid_kem"]
    assert assess(rec(listed=False))["mitigations"] == ["enable_hybrid_kem_both"]


def test_tls12_rsa_key_transport_weak_key_and_cipher():
    crypto = {"auth": {"label": "classical", "rsa_key_transport": True, "leaf_key": "RSA-1024",
                       "leaf_key_bits": 1024}}
    r = rec(tls="1.2", cipher="TLS_RSA_WITH_3DES_EDE_CBC_SHA", nbytes=10_000)
    out = assess(r, crypto, tier="migrate_next")
    assert out["mitigations"] == ["upgrade_tls13", "disable_rsa_kex", "replace_weak_cipher",
                                  "replace_weak_rsa_key", "plan_pq_certificates"]
    plain = assess(r, {"auth": {"label": "classical"}}, tier="migrate_next")["hndl_score"]
    assert out["hndl_score"] == pytest.approx(plain * 1.25, abs=0.1)


def test_offered_only_flows():
    # pqc_listed unknown (None) or True: a HelloRetryRequest could still pick a PQ group
    assert assess(rec(basis="offered", listed=None))["mitigations"] == ["confirm_server_choice"]
    assert assess(rec(basis="offered", listed=True))["hndl_score"] == pytest.approx(0.4 * 0.6 * 100, abs=0.1)
    # pqc_listed False on an offered classical verdict = a certain classical ceiling
    ceiling = assess(rec(basis="offered", listed=False))
    assert ceiling["mitigations"] == ["update_client"]
    assert ceiling["hndl_score"] == pytest.approx(0.9 * 0.6 * 100, abs=0.1)


def test_every_code_has_advice_and_config_is_ordered():
    cfg = load_risk_config()
    assert level_of(None, cfg) == UNASSESSED and level_of(0.0, cfg) == "none"
    assert level_of(60, cfg) == "high" and level_of(59.9, cfg) == "medium"
    assert volume_factor(1, cfg) == cfg["volume"]["floor"]
    assert volume_factor(10**9, cfg) == 1.0
    assert all(MITIGATIONS[c] for c in MITIGATIONS)


def test_summary_counts_bytes_per_change():
    recs = [rec(listed=True) | assess(rec(listed=True), tier="migrate_first"),
            rec("hybrid", nbytes=30_000_000) | assess(rec("hybrid"))]
    s = summarise(recs)
    assert s["levels"]["high"]["flows"] == 1 and s["levels"]["none"]["bytes_share"] == 0.75
    assert s["mitigations"]["enable_hybrid_kem"]["bytes_share"] == 0.25
