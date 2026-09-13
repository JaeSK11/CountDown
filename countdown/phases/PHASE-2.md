# Phase 2 — Crypto Detection & Quantum-Resistance Verdict (Pipeline Stages 2–3)

**Roadmap ref:** `../PLAN.md` → Phase 2. **Depends on:** Phase 1 (`Flow.handshake_*_bytes`,
`l7_hint`). **★ This is the novel core of the project.**
**Goal:** deterministically parse the handshake bytes into cryptographic parameters (TLS version,
cipher suite, **key-exchange / KEM named group**, signature algs) and classify each flow's
**quantum resistance** — `classical` / `hybrid` / `pqc` / `not_observable` / `unknown`. **No ML.**

**One-line exit criterion:** for a direct-TLS 1.3 flow, `pqc_verdict(parse_handshake(flow)).label`
returns the correct class, cross-checked against `tshark`'s
`tls.handshake.extensions_key_share_group`; a self-generated Cloudflare capture yields `hybrid`
(`X25519MLKEM768`), and CSTNET flows yield `classical`.

---

## 1. The core principle (why this is deterministic, not ML)

In TLS 1.3 the `ClientHello` and `ServerHello` are **cleartext**, and the **selected key-exchange
group** is in the ServerHello `key_share` extension. Quantum-resistance is therefore a **lookup on
that codepoint**, not a prediction:

- **Negotiated group (authoritative)** = ServerHello `key_share.group`.
- **Offered groups** = ClientHello `supported_groups` (what the client was willing to do).
- Verdict keys off **negotiated** when the ServerHello is present; falls back to offered (with lower
  confidence) if only the client side was captured.

**Signatures are mostly invisible:** in TLS 1.3 the Certificate/CertificateVerify are encrypted, so
PQC *signature* detection (ML-DSA/Falcon/SPHINCS+) is usually N/A. We key the verdict off the
**KEM**, which is exactly the harvest-now-decrypt-later–relevant parameter.

**Invariant carried from Phases 1:** tunneled flows (VPN/Tor) expose only the outer handshake →
verdict = `not_observable`. They skip to Stage 4 (ML) with no PQC claim.

---

## 2. Scope

**In scope**
- `crypto/tls_parser.py` — TLS record + handshake parser (ClientHello/ServerHello, TLS 1.2 & 1.3).
- `crypto/quic_parser.py` — QUIC Initial decode: derive Initial keys, decrypt, reassemble CRYPTO
  frames → ClientHello (QUIC v1; v2 best-effort).
- `crypto/kem_registry.py` — named-group codepoint → {classical, hybrid, pqc} + metadata.
- `crypto/sigalg_registry.py` — signature-scheme codepoints (for the rare observable cases).
- `crypto/pqc_verdict.py` — the verdict engine (offered vs negotiated, confidence, evidence).
- Validation harness: cross-check vs `tshark`; metrics on CSTNET (classical) + PostQuantumTLS +
  **self-generated PQC positives**.

**Out of scope (later)**
- Any ML / traffic-type classification → Phase 3+.
- Decrypting application data (we never need it).
- Cert-chain / X.509 PQC signature parsing beyond what's observable (kept minimal).
- ECH inner-ClientHello recovery (can't — encrypted; detect + mark, don't attempt).

---

## 3. Dependencies
- Add `cryptography` (HKDF + AES-GCM/ChaCha20-Poly1305 for QUIC Initial decryption).
- **Validation-only:** `tshark`/`pyshark` as an oracle to cross-check extracted groups (not a
  runtime dep). Optional: `oqs-provider`/`openssl` or `curl` for generating PQC handshakes.

---

## 4. Directory additions

```
countdown/crypto/
├── tls_parser.py        # record layer + CH/SH handshake, extensions
├── quic_parser.py       # QUIC Initial: key derivation, AEAD decrypt, CRYPTO reassembly
├── kem_registry.py      # named-group table (+ data/kem_groups.yaml)
├── sigalg_registry.py   # signature schemes (ML-DSA/Falcon/SPHINCS+ + classical)
├── pqc_verdict.py       # verdict engine
└── data/
    ├── kem_groups.yaml  # codepoint → {name, family, algo, nist_level, draft?}
    └── sig_schemes.yaml
scripts/
├── validate_pqc.py      # countdown vs tshark; precision/recall report
└── gen_pqc_pcaps.sh     # capture PQC handshakes (Cloudflare / openssl+oqs) → data/_pqc_synth/
tests/
├── test_tls_parser.py   # golden ClientHello/ServerHello byte vectors
├── test_kem_registry.py
└── test_pqc_verdict.py
```

---

## 5. Components

### 5.1 TLS parser (`tls_parser.py`)
Parse from `handshake_*_bytes`:
- **Record layer:** iterate TLS records (type 0x16 handshake), concatenate handshake fragments.
- **ClientHello:** legacy_version; cipher_suites list; extensions.
- **ServerHello:** selected cipher_suite; extensions.
- **Extensions to extract** (by id): `supported_versions`(43)→ real TLS version;
  `supported_groups`(10)→ offered KEMs; `key_share`(51)→ offered/**selected** group;
  `signature_algorithms`(13); `server_name`(0)=SNI (context); `application_layer_protocol_negotiation`(16);
  detect `encrypted_client_hello`(65037/0xfe0d)=ECH.
- **GREASE hygiene (RFC 8701):** drop codepoints matching `0x?A?A` (0x0A0A,0x1A1A,…,0xFAFA) from
  groups/ciphers/sig-algs — never classify them.
- **TLS 1.2 path:** key-exchange family from cipher suite (ECDHE/DHE/RSA) + ServerKeyExchange
  named_curve; PQC in 1.2 is rare but handle gracefully.
- Output `Handshake{tls_version, cipher_suite, offered_groups[], negotiated_group, sig_algs[], sni,
  alpn, ech: bool, side: client|server|both}`. Robust to truncated/incomplete buffers → partial +
  `parse_ok` flag.

**DoD:** golden-vector unit tests for a real CH and SH; extracts `negotiated_group` from a captured
ServerHello; ignores GREASE.

### 5.2 QUIC parser (`quic_parser.py`)
- Parse QUIC long header → version, DCID. **Derive Initial secrets** (HKDF-Expand-Label from the
  version-specific `initial_salt` + DCID — public, no secrets) → decrypt the Initial packet AEAD →
  parse CRYPTO frames → reassemble the TLS ClientHello → hand to `tls_parser`.
- Server Initial (ServerHello) similarly. Support QUIC v1 (`0x00000001`); v2 (`0x6b3343cf`)
  best-effort; log unknown versions.
- **DoD:** decrypts a captured QUIC v1 Initial and recovers the ClientHello `supported_groups`;
  cross-checked vs tshark `quic`/`tls` dissectors.

### 5.3 KEM registry (`kem_registry.py` + `data/kem_groups.yaml`)
Authoritative codepoint → classification (extend as IANA evolves):

| Family | Examples (codepoint) |
| --- | --- |
| **classical** | x25519 (0x001D), x448 (0x001E), secp256r1 (0x0017), secp384r1 (0x0018), secp521r1 (0x0019), ffdhe2048–8192 (0x0100–0x0104) |
| **hybrid** (classical+PQC) | X25519MLKEM768 (0x11EC), SecP256r1MLKEM768 (0x11EB), SecP384r1MLKEM1024 (0x11ED), X25519Kyber768Draft00 (0x6399), SecP256r1Kyber768Draft00 (0x639A) |
| **pqc** (pure) | MLKEM512 (0x0200), MLKEM768 (0x0201), MLKEM1024 (0x0202) |

Each entry: `{name, family, algo, nist_level, draft_or_standard}`. Unknown codepoint → `unknown`
(logged, not crashed). `is_quantum_resistant = family in {hybrid, pqc}`.

**DoD:** table unit-tested; `classify(0x11EC) == hybrid`; GREASE + unknowns handled.

### 5.4 Verdict engine (`pqc_verdict.py`)
- Input: `Handshake` (+ `l7_hint` for tunneled short-circuit).
- Logic:
  - `l7_hint` tunneled / no handshake → **`not_observable`**.
  - parse failed / only partial, no group → **`unknown`**.
  - negotiated_group present → classify it → **`classical` | `hybrid` | `pqc`**.
  - only ClientHello captured → classify **best offered** group, mark `basis=offered`, lower
    confidence (client *offered* PQC ≠ server *negotiated* PQC).
- Output `Verdict{label, basis: negotiated|offered, group, confidence, evidence:{ext, codepoint}}`.
- Enrich the `Flow`: `tls_version, cipher_suite, kem_group, pqc_verdict, offered_groups, sni, alpn`.

**DoD:** CSTNET flow → `classical`; self-gen Cloudflare flow → `hybrid`; ISCXTor flow →
`not_observable`.

---

## 6. Validation strategy (this is a measurable phase)
1. **Oracle cross-check:** run `tshark -T fields -e tls.handshake.extensions_key_share_group` on a
   sample of CSTNET/PostQuantumTLS/self-gen pcaps; assert countdown's `negotiated_group` matches on
   ≥99% of parseable flows. Catches parser bugs deterministically.
2. **Classical negatives at scale:** CSTNET-TLS1.3 (46k sessions) — expect ~all `classical`;
   measure parser success rate + any misclassifications.
3. **PQC positives:** PostQuantumTLS (real-world, may be **sparse** for 2023) **+** self-generated
   guaranteed positives via `gen_pqc_pcaps.sh` (curl to Cloudflare / Google, or openssl+oqs-provider
   → `X25519MLKEM768`, `X25519Kyber768Draft00`).
4. **Report:** PQC-detection precision/recall/F1, parser success %, per-family counts, unknown/GREASE
   rates.

---

## 7. Work breakdown & Definition of Done

| # | Task | DoD |
| --- | --- | --- |
| 2.1 | `kem_groups.yaml` + `kem_registry.py` | `classify()` unit tests; GREASE/unknown safe |
| 2.2 | `tls_parser.py` records + CH/SH + extensions | golden-vector tests; negotiated_group extracted |
| 2.3 | GREASE filtering + TLS 1.2 fallback | GREASE dropped; 1.2 key-exchange family resolved |
| 2.4 | `sigalg_registry.py` (+ observable-sig handling) | classical + PQC sig codepoints mapped |
| 2.5 | `quic_parser.py` Initial decrypt → CH | recovers groups from a QUIC v1 Initial |
| 2.6 | `pqc_verdict.py` engine (negotiated vs offered) | correct labels on the 3 anchor cases |
| 2.7 | Flow enrichment (write crypto fields) | `Flow.pqc_verdict` populated end-to-end |
| 2.8 | `gen_pqc_pcaps.sh` self-gen positives | produces ≥1 hybrid + ≥1 classical capture |
| 2.9 | `validate_pqc.py` vs tshark + metrics report | precision/recall table printed |

---

## 8. Acceptance test (end of Phase 2)

```
from countdown.sources import PcapSource
from countdown.flows import Reassembler
from countdown.crypto import parse_handshake, pqc_verdict

flows = list(Reassembler().run(PcapSource(CSTNET_PCAP)))
f  = next(x for x in flows if x.l7_hint == "tls")
hs = parse_handshake(f)
assert hs.parse_ok and hs.tls_version == "1.3" and hs.negotiated_group
assert pqc_verdict(hs).label == "classical"

# self-generated Cloudflare capture → hybrid X25519MLKEM768
assert pqc_verdict(parse_handshake(CF_FLOW)).label == "hybrid"

# tunneled → not_observable
assert pqc_verdict(parse_handshake(TOR_FLOW)).label == "not_observable"

# oracle agreement
# validate_pqc.py: countdown negotiated_group == tshark group on >=99% of parseable flows
```
**Then the pipeline can triage:** every flow gets a crypto verdict; **non-PQC flows feed Stage 4
(Phase 3 ML).**

---

## 9. Risks & open questions
- **Sparse real PQC positives** in PostQuantumTLS (2023 adoption low) → rely on self-generated
  positives for precision/recall; document the mix.
- **QUIC Initial decryption complexity:** version-specific salts + AEAD; scope to v1 first, log
  others. Retry/coalesced packets add edge cases.
- **ECH (Encrypted ClientHello):** inner CH (SNI, groups) hidden → mark `ech=True`; still read the
  ServerHello selected group if present, else `not_observable` on the client side.
- **Client-only captures:** offered≠negotiated → verdict `basis=offered`, lower confidence; be
  explicit in output and metrics.
- **Codepoint churn:** PQC codepoints are still standardizing (draft Kyber vs ML-KEM) → keep the
  registry in YAML, versioned, easy to extend; unknowns logged, never fatal.
- **GREASE:** must be filtered everywhere (groups, ciphers, sig-algs) or it pollutes offered-group
  stats.

## 10. Not doing yet (explicit)
ML/traffic-type classification (P3+), app-data decryption (never), full X.509 PQC-cert parsing,
ECH inner recovery, live capture (P6).
