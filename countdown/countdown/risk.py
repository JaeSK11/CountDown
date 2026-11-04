"""Harvest-now-decrypt-later risk score and mitigation guidance per flow (Phase 7, task 7.2).

The PQC verdict says *whether* a recording of a flow would decrypt once a quantum computer
exists; the triage tables say *how much* traffic is in that position.  This module turns both
into the thing an operator acts on: a 0-100 score per flow and the concrete change that
removes the exposure.

    score = 100 * exposure * sensitivity * volume * key_transport      (clamped to 100)

with every weight in ``configs/hndl_risk.yaml`` (see there for the meaning of each factor).
Sensitivity comes from the readiness tier of the flow's *predicted* traffic type, so a score
inherits the classifier's error -- the measured class-mix error is printed with every triage
report.  Two rules carry over from the verdict engine:

* a flow whose handshake was not seen is **unassessed** (``score None``), never scored as
  exposed or as safe;
* authentication (the certificate chain) never moves the score: a recorded session is opened
  by breaking the key exchange.  A classical certificate gets its own advice, because it
  matters for *active* impersonation once a quantum computer exists, not for recordings --
  except under RSA key transport, where the certificate key is the key exchange.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from countdown.eval.triage import _bucket

DEFAULT_RISK_PATH = Path(__file__).resolve().parents[1] / "configs" / "hndl_risk.yaml"

UNASSESSED = "unassessed"

#: Mitigation codes -> the advice printed for them.  Codes are what records carry (stable,
#: countable); the text is for people.
MITIGATIONS: dict[str, str] = {
    "enable_hybrid_kem": "Enable a hybrid post-quantum group (X25519MLKEM768) on the server; "
                         "the client already offers one, so this is a server-side change only.",
    "enable_hybrid_kem_both": "Enable X25519MLKEM768 on both ends (e.g. OpenSSL >= 3.5, recent "
                              "BoringSSL / Go / rustls); neither side negotiated a post-quantum group.",
    "upgrade_tls13": "Upgrade to TLS 1.3: post-quantum key exchange is defined only for TLS 1.3.",
    "update_client": "The client offered no post-quantum group: update the client or its TLS library.",
    "disable_rsa_kex": "Disable RSA key-transport cipher suites: without forward secrecy, breaking "
                       "the one certificate key decrypts every recorded session that used it.",
    "replace_weak_cipher": "Replace the broken symmetric cipher (3DES / RC4 / NULL / EXPORT) now; "
                           "it is weak even without a quantum computer.",
    "replace_weak_rsa_key": "The certificate's RSA key is under 2048 bits: replace it now.",
    "plan_pq_certificates": "Plan post-quantum certificates (ML-DSA): a classical certificate lets a "
                            "future quantum adversary impersonate the server. This affects live "
                            "connections, not recordings.",
    "confirm_server_choice": "A post-quantum group was offered but the server reply was not captured: "
                             "capture the full handshake to confirm.",
    "assess_tunnel": "Tunnelled traffic: the inner handshake is hidden; assess the VPN/Tor endpoint's "
                     "own key exchange.",
    "capture_handshake": "No handshake in the capture: capture the start of the connection to assess it.",
}

_WEAK_CIPHER_MARKERS = ("3DES", "RC4", "_NULL_", "EXPORT", "_DES_")
_TUNNELS = frozenset({"tor", "vpn", "tunnel_openvpn", "tunnel_ike", "tunnel_wireguard"})


@lru_cache(maxsize=4)
def load_risk_config(path: str | Path = DEFAULT_RISK_PATH) -> dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    levels = cfg["levels"]
    if list(levels) != sorted(levels, key=lambda k: -levels[k]):
        raise ValueError("hndl_risk.yaml: levels must be listed highest bound first")
    return cfg


def level_of(score: float | None, cfg: dict[str, Any]) -> str:
    if score is None:
        return UNASSESSED
    for name, lo in cfg["levels"].items():
        if score >= lo:
            return name
    return "none"


def volume_factor(nbytes: int, cfg: dict[str, Any]) -> float:
    v = cfg["volume"]
    frac = min(1.0, math.log10(max(nbytes, 1)) / math.log10(v["saturate_bytes"]))
    return v["floor"] + (1.0 - v["floor"]) * frac


def assess(record: dict[str, Any], crypto: dict[str, Any] | None = None,
           tier: str | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """``{hndl_score, hndl_level, mitigations}`` for one pipeline record.

    ``crypto`` is the flow's ``meta['crypto']`` (for the certificate verdict and key exchange
    family); ``tier`` the readiness tier of the record's predicted class, if it has one.
    """
    cfg = cfg or load_risk_config()
    crypto = crypto or {}
    auth = crypto.get("auth") or {}
    bucket = _bucket(record["pqc_verdict"], record["pqc_basis"], record.get("pqc_listed"))
    codes: list[str] = []

    exposure = cfg["exposure"].get(bucket)
    score: float | None = None
    if exposure is not None:
        sens = cfg["sensitivity"].get(tier or "unrated", cfg["sensitivity"]["unrated"])
        mult = cfg["key_transport_multiplier"] if auth.get("rsa_key_transport") else 1.0
        score = min(100.0, 100.0 * exposure * sens * volume_factor(int(record.get("bytes") or 0), cfg) * mult)

    # -- what to change ------------------------------------------------------------------
    tls = record.get("tls_version")
    if bucket == "at_risk":
        if tls == "1.3":
            codes.append("enable_hybrid_kem" if record.get("pqc_listed") else "enable_hybrid_kem_both")
        else:
            codes.append("upgrade_tls13")
    elif bucket == "at_risk_ceiling":
        codes.append("update_client" if tls == "1.3" else "upgrade_tls13")
    elif bucket == "pqc_offered_undecided":
        codes.append("confirm_server_choice")
    elif bucket == "not_observable":
        codes.append("assess_tunnel" if record.get("l7") in _TUNNELS else "capture_handshake")
    if auth.get("rsa_key_transport"):
        codes.append("disable_rsa_kex")
    cipher = record.get("cipher") or ""
    if any(m in cipher for m in _WEAK_CIPHER_MARKERS):
        codes.append("replace_weak_cipher")
    bits = auth.get("leaf_key_bits")
    if bits and str(auth.get("leaf_key", "")).startswith("RSA") and bits < 2048:
        codes.append("replace_weak_rsa_key")
    if auth.get("label") == "classical":
        codes.append("plan_pq_certificates")

    return {"hndl_score": None if score is None else round(score, 1),
            "hndl_level": level_of(score, cfg), "mitigations": codes}


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score levels by flows and bytes, and how many flows / bytes each mitigation covers --
    the "what to fix first" table of ``countdown triage``."""
    from collections import Counter

    lv_f, lv_b, mit_f, mit_b = Counter(), Counter(), Counter(), Counter()
    for r in records:
        b = int(r.get("bytes") or 0)
        lv_f[r.get("hndl_level", UNASSESSED)] += 1
        lv_b[r.get("hndl_level", UNASSESSED)] += b
        for c in r.get("mitigations") or []:
            mit_f[c] += 1
            mit_b[c] += b
    tot_b = float(sum(lv_b.values())) or 1.0
    order = ["high", "medium", "low", "none", UNASSESSED]
    return {
        "levels": {k: {"flows": lv_f[k], "bytes_share": round(lv_b[k] / tot_b, 4)}
                   for k in order if lv_f[k]},
        "mitigations": {c: {"flows": mit_f[c], "bytes_share": round(mit_b[c] / tot_b, 4),
                            "advice": MITIGATIONS[c]}
                        for c, _ in sorted(mit_b.items(), key=lambda kv: (-kv[1], kv[0]))},
    }


def summary_markdown(s: dict[str, Any]) -> str:
    lines = ["#### Harvest-now-decrypt-later risk", "",
             "| level | flows | bytes share |", "| --- | --- | --- |"]
    for k, v in s["levels"].items():
        lines.append(f"| {k} | {v['flows']:,} | {v['bytes_share']:.1%} |")
    if s["mitigations"]:
        lines += ["", "#### What to change (by bytes covered)", "",
                  "| change | flows | bytes share | advice |", "| --- | --- | --- | --- |"]
        for c, v in s["mitigations"].items():
            lines.append(f"| `{c}` | {v['flows']:,} | {v['bytes_share']:.1%} | {v['advice']} |")
    return "\n".join(lines) + "\n"
