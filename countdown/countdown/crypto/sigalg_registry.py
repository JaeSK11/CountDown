"""Signature-scheme codepoint -> classification (Phase 2, task 2.4).

Deliberately a *secondary* signal.  In TLS 1.3 the Certificate and CertificateVerify
messages are encrypted under the handshake traffic keys, so a passive observer never sees
which signature algorithm the server actually used.  What is visible is the ClientHello's
``signature_algorithms`` extension -- a statement of what the client would *accept*.

That distinction is the whole reason the PQC verdict keys off the KEM and not the
signature.  A client advertising ``mldsa65`` tells us the ecosystem is moving; it tells us
nothing about whether this connection was authenticated post-quantumly.  And it does not
matter for harvest-now-decrypt-later at all: a recorded session is decrypted by breaking
the *key exchange*, not the signature, because the signature only ever proved liveness at
handshake time.  We surface it as context and never let it move the verdict.

The table mirrors ``kem_registry`` in shape so both read the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

from countdown.config import get_logger
from countdown.crypto.kem_registry import GREASE, PQC, UNKNOWN, is_grease, strip_grease

log = get_logger(__name__)

DATA_PATH = Path(__file__).resolve().parent / "data" / "sig_schemes.yaml"

CLASSICAL = "classical"
_VALID_FAMILIES = frozenset({CLASSICAL, PQC})


@dataclass(frozen=True, slots=True)
class SigScheme:
    codepoint: int
    name: str
    family: str  # classical | pqc
    algo: str
    nist_level: int | None
    status: str

    @property
    def is_post_quantum(self) -> bool:
        return self.family == PQC

    @property
    def hex(self) -> str:
        return f"0x{self.codepoint:04X}"

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.name}({self.hex}, {self.family})"


@lru_cache(maxsize=1)
def _table() -> dict[int, SigScheme]:
    with open(DATA_PATH) as fh:
        raw = yaml.safe_load(fh) or {}
    table: dict[int, SigScheme] = {}
    for row in raw.get("schemes", []):
        try:
            cp = int(row["codepoint"])
            family = str(row["family"]).lower()
            name = str(row["name"])
        except (KeyError, TypeError, ValueError):
            log.warning("sig_schemes.yaml: skipping malformed row %r", row)
            continue
        if family not in _VALID_FAMILIES or is_grease(cp) or not 0 <= cp <= 0xFFFF:
            log.warning("sig_schemes.yaml: skipping %s (0x%04X, family=%s)", name, cp, family)
            continue
        level = row.get("nist_level")
        table[cp] = SigScheme(
            codepoint=cp, name=name, family=family, algo=str(row.get("algo", name)),
            nist_level=int(level) if level is not None else None,
            status=str(row.get("status", "standard")),
        )
    if not table:
        raise RuntimeError(f"sig_schemes.yaml at {DATA_PATH} produced an empty registry")
    return table


@lru_cache(maxsize=1)
def table_version() -> str:
    with open(DATA_PATH) as fh:
        return str((yaml.safe_load(fh) or {}).get("version", "unknown"))


def lookup(codepoint: int) -> SigScheme | None:
    if not isinstance(codepoint, int):
        return None
    return _table().get(codepoint)


def classify(codepoint: int) -> str:
    """``classical`` | ``pqc`` | ``grease`` | ``unknown``."""
    if is_grease(codepoint):
        return GREASE
    entry = lookup(codepoint)
    return entry.family if entry is not None else UNKNOWN


def name_of(codepoint: int) -> str:
    if is_grease(codepoint):
        return f"GREASE(0x{codepoint:04X})"
    entry = lookup(codepoint)
    if entry is not None:
        return entry.name
    return f"unknown(0x{codepoint:04X})" if isinstance(codepoint, int) else "unknown"


def is_post_quantum(codepoint: int) -> bool:
    entry = lookup(codepoint)
    return entry is not None and entry.is_post_quantum


def pq_schemes(codepoints: Iterable[int]) -> list[int]:
    """The post-quantum entries of an offered signature_algorithms list, GREASE removed."""
    return [cp for cp in strip_grease(codepoints) if is_post_quantum(cp)]


def summarise(codepoints: Iterable[int]) -> dict[str, object]:
    """Context block for a verdict: what the client was willing to accept.

    ``advertises_pqc_signatures`` is exactly that -- an advertisement.  It is never
    evidence that this handshake *was* signed post-quantumly, which is unobservable.
    """
    clean = strip_grease(codepoints)
    pq = [cp for cp in clean if is_post_quantum(cp)]
    return {
        "offered": [name_of(cp) for cp in clean],
        "pq_offered": [name_of(cp) for cp in pq],
        "advertises_pqc_signatures": bool(pq),
        "observable": False,  # TLS 1.3 encrypts CertificateVerify; see module docstring
    }


def all_schemes() -> dict[int, SigScheme]:
    return dict(_table())
