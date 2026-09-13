"""Phase 2, task 2.1 -- the named-group registry.

The registry is the entire basis of the verdict, so these tests are less about coverage
than about the two errors that would matter: classifying something as quantum-resistant
when it is not, and letting GREASE reach a verdict.
"""

from __future__ import annotations

import pytest

from qsentinel.crypto import kem_registry as reg


# -- the codepoints PHASE-2.md names explicitly -------------------------------------------

@pytest.mark.parametrize("codepoint,name,family", [
    (0x001D, "x25519", reg.CLASSICAL),
    (0x001E, "x448", reg.CLASSICAL),
    (0x0017, "secp256r1", reg.CLASSICAL),
    (0x0018, "secp384r1", reg.CLASSICAL),
    (0x0019, "secp521r1", reg.CLASSICAL),
    (0x0100, "ffdhe2048", reg.CLASSICAL),
    (0x0104, "ffdhe8192", reg.CLASSICAL),
    (0x11EC, "X25519MLKEM768", reg.HYBRID),
    (0x11EB, "SecP256r1MLKEM768", reg.HYBRID),
    (0x11ED, "SecP384r1MLKEM1024", reg.HYBRID),
    (0x6399, "X25519Kyber768Draft00", reg.HYBRID),
    (0x639A, "SecP256r1Kyber768Draft00", reg.HYBRID),
    (0x0200, "MLKEM512", reg.PQC),
    (0x0201, "MLKEM768", reg.PQC),
    (0x0202, "MLKEM1024", reg.PQC),
])
def test_known_codepoints(codepoint, name, family):
    assert reg.classify(codepoint) == family
    assert reg.name_of(codepoint) == name
    assert reg.lookup(codepoint).codepoint == codepoint


def test_classify_the_headline_case():
    """The one assertion PHASE-2.md's DoD spells out."""
    assert reg.classify(0x11EC) == reg.HYBRID


def test_quantum_resistance_is_exactly_hybrid_and_pqc():
    assert reg.is_quantum_resistant(0x11EC)      # hybrid
    assert reg.is_quantum_resistant(0x0201)      # pure PQC
    assert not reg.is_quantum_resistant(0x001D)  # x25519
    assert not reg.is_quantum_resistant(0x1234)  # unknown
    assert not reg.is_quantum_resistant(0x0A0A)  # GREASE


# -- GREASE (RFC 8701) ----------------------------------------------------------------------

GREASE_VALUES = [0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
                 0x8A8A, 0x9A9A, 0xAAAA, 0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA]


@pytest.mark.parametrize("value", GREASE_VALUES)
def test_all_sixteen_grease_values(value):
    assert reg.is_grease(value)
    assert reg.classify(value) == reg.GREASE
    assert not reg.is_quantum_resistant(value)
    assert reg.lookup(value) is None


@pytest.mark.parametrize("value", [0x001D, 0x11EC, 0x0A0B, 0x1A2A, 0x0AA0, 0x0000, 0xFFFF])
def test_non_grease_is_not_flagged(value):
    assert not reg.is_grease(value)


def test_strip_grease_preserves_order():
    got = reg.strip_grease([0x0A0A, 0x11EC, 0xFAFA, 0x001D, 0x2A2A, 0x0017])
    assert got == [0x11EC, 0x001D, 0x0017]


def test_no_grease_codepoint_is_in_the_table():
    """A GREASE value in the YAML would make GREASE classifiable -- the loader must drop it."""
    assert not [cp for cp in reg.all_groups() if reg.is_grease(cp)]


# -- unknowns are never silently downgraded to classical ------------------------------------

@pytest.mark.parametrize("value", [0x1234, 0x0999, 0xABCD, 0x11FF, 0x0203])
def test_unknown_codepoints_stay_unknown(value):
    assert reg.classify(value) == reg.UNKNOWN
    assert reg.classify(value) != reg.CLASSICAL
    assert reg.lookup(value) is None
    assert "unknown" in reg.name_of(value)


def test_bad_input_does_not_raise():
    for bad in (None, "x25519", -1, 1 << 40, 3.5):
        assert reg.classify(bad) in (reg.UNKNOWN, reg.GREASE)
        assert reg.lookup(bad) is None
        assert not reg.is_quantum_resistant(bad)


# -- best_group, used only for the offered-basis fallback -------------------------------------

def test_best_group_prefers_pqc_then_hybrid_then_classical():
    assert reg.best_group([0x001D, 0x11EC, 0x0017]) == 0x11EC     # hybrid beats classical
    assert reg.best_group([0x001D, 0x11EC, 0x0201]) == 0x0201     # pure PQC beats hybrid
    assert reg.best_group([0x001D, 0x0017]) == 0x001D             # first classical wins
    assert reg.best_group([0x0A0A, 0x001D]) == 0x001D             # GREASE ignored
    assert reg.best_group([]) is None
    assert reg.best_group([0x0A0A, 0xFAFA]) is None               # nothing but GREASE


def test_best_group_ties_break_on_client_order():
    assert reg.best_group([0x0017, 0x001D]) == 0x0017
    assert reg.best_group([0x001D, 0x0017]) == 0x001D


# -- table integrity ---------------------------------------------------------------------------

def test_table_loads_and_is_self_consistent():
    table = reg.all_groups()
    assert len(table) > 40
    for cp, entry in table.items():
        assert entry.codepoint == cp
        assert entry.family in (reg.CLASSICAL, reg.HYBRID, reg.PQC)
        assert 0 <= cp <= 0xFFFF
        assert entry.name
        # Every quantum-resistant entry must carry a NIST level; a hybrid without one
        # would report protection we cannot characterise.
        if entry.family in (reg.HYBRID, reg.PQC):
            assert entry.nist_level in (1, 2, 3, 5), (entry.name, entry.nist_level)
        else:
            assert entry.nist_level is None


def test_version_string_is_present():
    assert reg.table_version() != "unknown"
