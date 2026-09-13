"""Named-group codepoint -> quantum-resistance classification (Phase 2, task 2.1).

The whole PQC verdict reduces to a lookup in this table: a TLS 1.3 flow is
quantum-resistant iff the group the server *selected* has family ``hybrid`` or ``pqc``.
There is no inference and no model here -- the codepoint is on the wire in cleartext.

Three things this module refuses to do, all of them deliberate:

* **Never guess.**  An unrecognised codepoint returns ``UNKNOWN``, never ``CLASSICAL``.
  "We do not know" and "we know it is breakable" are different findings, and collapsing
  them would silently invent classical negatives every time IANA assigns a codepoint.
* **Never classify GREASE.**  RFC 8701 reserves 16 values that clients inject purely to
  keep middleboxes honest.  They are not offers, so they are filtered before any counting.
* **Never crash.**  Malformed input, absurd codepoints and unknown families all resolve to
  ``UNKNOWN``; the table is data (``data/kem_groups.yaml``) so that extending it as PQC
  standardisation churns is a data edit, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

from qsentinel.config import get_logger

log = get_logger(__name__)

DATA_PATH = Path(__file__).resolve().parent / "data" / "kem_groups.yaml"

# -- families ---------------------------------------------------------------------------

CLASSICAL = "classical"
HYBRID = "hybrid"
PQC = "pqc"
UNKNOWN = "unknown"
GREASE = "grease"

#: Families whose break requires a cryptographically-relevant quantum computer to *not*
#: exist... i.e. the ones that survive one.
QUANTUM_RESISTANT_FAMILIES = frozenset({HYBRID, PQC})

_VALID_FAMILIES = frozenset({CLASSICAL, HYBRID, PQC})


# -- GREASE (RFC 8701) ------------------------------------------------------------------

def is_grease(codepoint: int) -> bool:
    """True for the 16 RFC 8701 GREASE values ``0x0A0A``, ``0x1A1A`` ... ``0xFAFA``.

    Clients advertise these as deliberate nonsense to detect middleboxes that fail on
    unknown codepoints.  They are not real offers: counting them would pollute every
    offered-group statistic, and classifying one would be meaningless.
    """
    if not isinstance(codepoint, int) or not 0 <= codepoint <= 0xFFFF:
        return False
    # Both bytes must be of the form 0x?A -- i.e. low nibble A, and the two bytes equal.
    hi, lo = codepoint >> 8, codepoint & 0xFF
    return hi == lo and (hi & 0x0F) == 0x0A


# -- table ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KemGroup:
    """One row of the named-group table."""

    codepoint: int
    name: str
    family: str  # classical | hybrid | pqc
    algo: str
    nist_level: int | None
    status: str  # standard | draft | obsolete

    @property
    def is_quantum_resistant(self) -> bool:
        return self.family in QUANTUM_RESISTANT_FAMILIES

    @property
    def hex(self) -> str:
        return f"0x{self.codepoint:04X}"

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.name}({self.hex}, {self.family})"


@lru_cache(maxsize=1)
def _table() -> dict[int, KemGroup]:
    """Parse ``kem_groups.yaml`` once.  Bad rows are dropped with a warning, never fatal."""
    with open(DATA_PATH) as fh:
        raw = yaml.safe_load(fh) or {}

    table: dict[int, KemGroup] = {}
    for row in raw.get("groups", []):
        try:
            cp = int(row["codepoint"])
            family = str(row["family"]).lower()
            name = str(row["name"])
        except (KeyError, TypeError, ValueError):
            log.warning("kem_groups.yaml: skipping malformed row %r", row)
            continue
        if family not in _VALID_FAMILIES:
            log.warning("kem_groups.yaml: %s has unknown family %r -- skipped", name, family)
            continue
        if not 0 <= cp <= 0xFFFF:
            log.warning("kem_groups.yaml: %s codepoint %d out of range -- skipped", name, cp)
            continue
        if is_grease(cp):
            # A GREASE codepoint in the table would make GREASE classifiable.  Refuse it.
            log.warning("kem_groups.yaml: %s uses GREASE codepoint 0x%04X -- skipped", name, cp)
            continue
        if cp in table:
            log.warning("kem_groups.yaml: duplicate codepoint 0x%04X (%s vs %s)",
                        cp, table[cp].name, name)
            continue
        level = row.get("nist_level")
        table[cp] = KemGroup(
            codepoint=cp,
            name=name,
            family=family,
            algo=str(row.get("algo", name)),
            nist_level=int(level) if level is not None else None,
            status=str(row.get("status", "standard")),
        )
    if not table:
        raise RuntimeError(f"kem_groups.yaml at {DATA_PATH} produced an empty registry")
    return table


@lru_cache(maxsize=1)
def table_version() -> str:
    """The ``version`` string of the YAML table, recorded in validation reports."""
    with open(DATA_PATH) as fh:
        return str((yaml.safe_load(fh) or {}).get("version", "unknown"))


# -- lookup API -------------------------------------------------------------------------

def lookup(codepoint: int) -> KemGroup | None:
    """The registry row for ``codepoint``, or ``None`` if unlisted (incl. GREASE)."""
    if not isinstance(codepoint, int):
        return None
    return _table().get(codepoint)


def classify(codepoint: int) -> str:
    """``classical`` | ``hybrid`` | ``pqc`` | ``grease`` | ``unknown``.

    Unknown is a real answer, not a failure: PQC codepoints are still being assigned, and
    reporting an unassigned one as ``classical`` would fabricate a negative.
    """
    if is_grease(codepoint):
        return GREASE
    entry = lookup(codepoint)
    return entry.family if entry is not None else UNKNOWN


def name_of(codepoint: int) -> str:
    """Human-readable group name; falls back to ``unknown(0x....)`` for unlisted values."""
    if is_grease(codepoint):
        return f"GREASE(0x{codepoint:04X})"
    entry = lookup(codepoint)
    if entry is not None:
        return entry.name
    return f"unknown(0x{codepoint:04X})" if isinstance(codepoint, int) else "unknown"


def is_quantum_resistant(codepoint: int) -> bool:
    """True only for hybrid/pure-PQC groups.  Unknown and GREASE are both False."""
    entry = lookup(codepoint)
    return entry is not None and entry.is_quantum_resistant


def strip_grease(codepoints: Iterable[int]) -> list[int]:
    """Drop RFC 8701 GREASE values, preserving order.

    Applied to supported_groups, cipher_suites and signature_algorithms alike -- GREASE
    appears in all three, and leaving it in any one of them corrupts that statistic.
    """
    return [cp for cp in codepoints if not is_grease(cp)]


def best_group(codepoints: Iterable[int]) -> int | None:
    """The most quantum-resistant group in an *offered* list.

    Ranking is ``pqc`` > ``hybrid`` > ``classical`` > ``unknown``, ties broken by the order
    the client listed them (which is the client's own preference order).  Used only for
    the ``basis=offered`` fallback, where no server selection exists to key off.

    Note this reports the *ceiling* of what the client was willing to do -- deliberately
    optimistic, which is precisely why an offered-basis verdict carries lower confidence.
    """
    rank = {PQC: 3, HYBRID: 2, CLASSICAL: 1, UNKNOWN: 0}
    best, best_rank = None, -1
    for cp in strip_grease(codepoints):
        r = rank.get(classify(cp), 0)
        if r > best_rank:
            best, best_rank = cp, r
    return best


def all_groups() -> dict[int, KemGroup]:
    """The full table, keyed by codepoint (read-only copy)."""
    return dict(_table())
