# Phase 7 — Certificates, HNDL risk, E1 `file_transfer`, and a PQC census

**Pipeline:** stages 2–4 + reporting · **Deliverable:** four follow-ups from Phase 6
(`PHASE-6.md` §7, §10, §15), plus two Stage-2/3 parser fixes the census turned up.

Started 2026-11-04. Committed locally, not pushed; the model zoo is unchanged (`models-v1`).

---

## 1. Scope

| # | task | from |
| --- | --- | --- |
| 7.1 | certificate-chain and signature analysis where the handshake shows it (TLS 1.2) | §7 future work |
| 7.2 | harvest-now-decrypt-later (HNDL) risk score and mitigation advice for each flow | §7 future work |
| 7.3 | E1 `file_transfer` under-count | §10, §15 known limit |
| 7.4 | a larger capture of real post-quantum traffic, turned into a statistic | §10, §13 |

No new dependencies and no environment changes. The DER walker is hand-written, and the census
uses the `openssl` that is already in the image.

## 2. What was added

```
countdown/crypto/x509.py              # 7.1 DER walker, Certificate message, authentication verdict
countdown/crypto/data/x509_algs.yaml  # 7.1 X.509 algorithm OIDs (classical + FIPS 203/204/205)
countdown/risk.py                     # 7.2 HNDL score, mitigation codes, triage summary
configs/hndl_risk.yaml                # 7.2 the score's weights (a judgement, per customer)
countdown/ensemble/decision.py        # 7.3 per-class decision bias (fit, apply, load)
configs/decision_bias.yaml            # 7.3 E1's biases, plain numbers, tracked
scripts/fit_decision_bias.py          # 7.3 fits the bias on out-of-fold predictions
scripts/pqc_census.py                 # 7.4 passive census of the Tranco top sites
results/phase7/                       # decision_bias_E1.{json,md}, census/{census.json,.md,sites.csv}
tests/test_x509.py, test_risk.py, test_decision_bias.py   (+ regressions in existing tests)
```

Changed: `crypto/tls_parser.py` (Certificate, CertificateRequest, ServerKeyExchange signature,
two fixes in §6.3), `crypto/pqc_verdict.py` (the TLS 1.2 suite rule, §6.3),
`pipeline/pipeline.py` (decision bias, risk fields), `cli.py` (`--no-decision-bias`, new
record fields, risk tables in `triage`), `eval/triage.py` (byte-weighted fidelity, the effect of
the bias), `sources/live.py` (a `stop` event).

New record fields: `auth_verdict`, `cert_key`, `hndl_score`, `hndl_level`, `mitigations`.

## 3. 7.1 Certificate-chain analysis

Before TLS 1.3 the server's `Certificate` is plaintext (`0x16`). `x509.py` reads each
certificate's public-key algorithm and the algorithm its issuer signed it with. It does this
with a bounds-checked DER walker, not `cryptography.x509`, for two reasons: the 8 KB window
routinely cuts the last certificate, and a library rejects a cut certificate or a key type it
doesn't implement (ML-DSA, for example). The parser also reads the SignatureScheme of the TLS
1.2 `ServerKeyExchange` (the one signature a passive observer sees the server actually make)
and the client-authentication request (`CertificateRequest`).

The authentication verdict is kept beside the KEM verdict, never inside it (`meta.crypto.auth`;
`auth_verdict` / `cert_key` in records):

- `classical` / `pqc`: from the leaf key, looked up in `x509_algs.yaml`. Only NIST-assigned
  post-quantum OIDs are listed. Draft composite and Falcon OIDs are left out, so they come back
  as `unknown(<oid>)` and are never guessed.
- `not_observable` in TLS 1.3 and QUIC (the chain is encrypted), and when no chain was
  captured.
- `rsa_key_transport`: under TLS 1.2 RSA key transport, the certificate key *is* the key
  exchange. That is the one case where the certificate matters for recorded traffic, and §4
  scores it.

Guards: a certificate list whose first entry is not a DER SEQUENCE is not a chain. A resumed
1.2 session sends its *encrypted* Finished in a `0x16` record after ChangeCipherSpec, and on
ISCX those bytes passed for a `Certificate` header 15 times before the guard was added.

**On the datasets** (all classical, as expected):

| dataset | TLS 1.2 flows with a chain read | leaf keys | notes |
| --- | --- | --- | --- |
| ISCX-VPN (2016) | 1,280 (+6 from the ServerKeyExchange alone) | RSA-2048 745, secp256r1 417, **RSA-1024 108**, RSA-1280 10 | 908 leaves signed with SHA-1/RSA; 688 flows on RSA key transport |
| PostQuantumTLS (2023) | 3,331 (+13) | RSA-2048 2,754, secp256r1 506, RSA-4096 69 | 149 on RSA key transport; 33 chains cut by the window |

**On live servers** (§6): the passive leaf key matched the client's own parse of the
certificate on **756 / 756** TLS 1.2 handshakes.

## 4. 7.2 Harvest-now-decrypt-later risk score

`countdown/risk.py`, weights in `configs/hndl_risk.yaml`:

    score = 100 · exposure · sensitivity · volume · key_transport      (clamped to 100)

- **exposure** comes from the triage bucket: `at_risk` 1.0, `at_risk_ceiling` 0.9,
  `pqc_offered_undecided` 0.4, `quantum_safe` 0.
- **sensitivity** comes from the readiness tier of the *predicted* traffic type: `migrate_first`
  1.0, `migrate_next` 0.6, `low_priority` 0.3. Targets without tiers (app, activity) get 0.6.
- **volume** = 0.5 + 0.5·min(1, log₁₀ bytes / 7), so 10 MB counts as a full recording.
- **key_transport** ×1.25 under TLS 1.2 RSA key transport.

Levels: high ≥ 60, medium ≥ 30, low ≥ 1, none = 0. A flow whose handshake was not seen is
**`unassessed`** (no score), never guessed. Like `readiness_tiers.yaml`, the weights are a
judgement, not something learned from data.

Each record also gets **mitigation codes** (`risk.MITIGATIONS` holds the text):
`enable_hybrid_kem` (the client already offers ML-KEM, so the fix is server-side),
`enable_hybrid_kem_both`, `upgrade_tls13`, `update_client`, `disable_rsa_kex`,
`replace_weak_cipher`, `replace_weak_rsa_key`, `plan_pq_certificates` (live impersonation,
not recordings), `confirm_server_choice`, `assess_tunnel`, `capture_handshake`.
`countdown triage` now ends with the level table and a "what to change" table ranked by the
bytes each change would cover.

Caveat, stated in the `triage` output: sensitivity inherits the classifier's error (§5).

## 5. 7.3 E1 `file_transfer`

**What the under-count is.** On E1's out-of-fold predictions, `file_transfer` recall is 0.049
by flows but 0.64 by bytes. **71 % of `file_transfer`-labelled flows are under 5 KB.** These
are the control, DNS and background connections inside the file-transfer captures, which carry
the capture's label (labels are per capture file). The large transfers are mostly found. Flow
counts overstate the problem, and HNDL is about bytes.

**What was done.** A per-class **decision bias** (`ensemble/decision.py`): label =
argmax(log p + b). It is fit by coordinate ascent on macro-F1 over the **out-of-fold**
predictions of the train split (52,605 flows) and never sees test labels. It is not a training
change: decision D5 (no class weights) stands, and members, calibrators and the zoo's pickles
are untouched. The biases are plain numbers in `configs/decision_bias.yaml`, so `models-v1`
still verifies. `--no-decision-bias` gives the plain argmax.
Biases: file_transfer +0.59, voip +0.59, chat/video +0.34, email/p2p +0.09, audio −0.91,
browsing −1.16.

| E1, grouped test split (17,918 flows) | before | after |
| --- | --- | --- |
| macro-F1 | 0.6654 | **0.6747** |
| macro-F1, byte-weighted | 0.7548 | 0.7727 |
| `file_transfer` recall (flows / bytes) | 0.466 / 0.956 | 0.494 / 0.963 |
| out-of-fold `file_transfer` recall (flows / bytes) | 0.049 / 0.636 | 0.205 / 0.806 |

**Fidelity of the predicted breakdown** of *exposed* test flows (`results/experiments/triage.md`;
TV = total-variation distance between predicted and true mix):

| E1 (1,468 exposed test flows) | accuracy | class-mix TV (flows / bytes) | tier-mix TV (flows / bytes) | worst class by flows |
| --- | --- | --- | --- | --- |
| without bias (Phase 6) | 0.706 | 0.127 / 0.127 | 0.109 / 0.051 | file_transfer −8.0 pts |
| **with bias (deployed)** | **0.718** | **0.120 / 0.091** | 0.116 / **0.017** | voip +10.2 pts |

By bytes, the tier mix that drives migration priority is now off by 1.7 points (it was 5.1).
By flows the gain is small and the error moves: `voip` is now over-counted. The by-flows tier
mix is slightly worse (0.116 vs 0.109). Reported as it is: the bias helps the byte-weighted
decision and does not fix E1 per flow. Doing that needs either flow-level labels in place of
capture-level ones or features that separate bulk transfers from their control connections.

## 6. 7.4 Post-quantum census of popular HTTPS servers

```bash
docker compose -f docker/compose.yaml run --rm job \
    python-netraw scripts/pqc_census.py --sites ../data/_census/top-1m.csv --tranco-list 746WX --top 1000
```

One container process captures its own bridge-network traffic (`countdown live` internals,
port 443, fast mode, through the whole pipeline). It also probes each of the **Tranco top 1,000**
(list `746WX`, 2026-11-03) with one `openssl s_client` (3.5.7) handshake per profile and
closes; no HTTP request is sent. Profile `tls13` uses OpenSSL's defaults, which offer
X25519MLKEM768 first. Profile `tls12` uses `-tls1_2`, so the chain crosses the wire in
cleartext. What the client says it negotiated is the ground truth for the passive verdicts.
Output: `results/phase7/census/` (per-site public facts, no addresses). One run, from one
vantage point, on 2026-11-04.

### 6.1 Results

| | profile `tls13` | profile `tls12` |
| --- | --- | --- |
| handshakes completed / captured | 759 / 758 | 760 / 759 |
| **quantum-safe (negotiated X25519MLKEM768)** | **455 (60.0 %)**; 455 of 638 TLS 1.3 servers (71 %) | 0 (none possible) |
| other groups | x25519 216, secp256r1 61, secp521r1 8, secp384r1 5, curve unseen 13 | x25519 623, secp256r1 88, … |
| servers that answered with TLS 1.2 | 121 | 760 |
| **passive group vs client's report** | 745 match, 6 curve not captured, **0 wrong** (751) | 727 match, 25 not captured, **0 wrong** (752) |
| passive leaf key vs client's parse | — | **756 / 756** |
| leaf keys | — | RSA-2048 465, secp256r1 264, RSA-4096 15, secp384r1 9 |
| RSA key transport | — | 7 |

1. **Most of the popular web now negotiates a hybrid post-quantum exchange** with a current
   client: 60 % of responding sites, and 71 % of those that speak TLS 1.3. Every public dataset
   (§10 of Phase 6) predates this.
2. **The passive verdict was never wrong** on 1,503 handshakes with ground truth. Where it had
   no group, it said so (the curve was outside the window) and still gave the right verdict.
3. **Six sites completed the TLS 1.2 handshake but not the default one** (the default
   ClientHello carries a ~1.2 KB ML-KEM key share): chinamobile.com, manutouch.com.cn,
   shifen.com, userapi.com, visa.cn, ys7.com. This fits the known large-ClientHello
   intolerance of some servers and middleboxes, but one run from one network can't separate
   that from flakiness.
4. With HNDL scoring, the census's exposed sites get `enable_hybrid_kem` (183, the client
   already offers it) or `upgrade_tls13` (120). The census flows are bare handshakes, so their
   predicted classes, and therefore their sensitivity and score level, mean nothing. Only the
   mitigation codes do.

### 6.2 How the census was checked

The first runs disagreed with the client on 21 of 761 handshakes. Every disagreement traced to
one of three causes: the census script (it counted s_client's failed handshakes as successes,
since s_client prints a `Protocol` line either way, and it joined probes to records over too
wide a time window), or one of the two parser bugs below. After the fixes: 0 wrong.

### 6.3 Two Stage-2/3 fixes (they change Phase-6 numbers)

- **A TLS 1.2 server answering a TLS 1.3-capable client was parsed as TLS 1.3.** The
  ServerHello only replaced the ClientHello's offered version if no ClientHello had been seen.
  A 1.2 server sends no `supported_versions`, so "1.3" stuck, and the parser stopped before the
  ServerKeyExchange. Now the server's choice always wins. A ServerHello cut before its
  extensions still gets 1.3 when it names a 1.3-only suite.
- **A TLS 1.2 ECDHE suite whose ServerKeyExchange is outside the window** (a 1.2 chain over
  ~8 KB, common on large sites) fell back to the client's *offered* groups, so it could even
  come out as "hybrid, offered". The server's choice of a 1.2 suite is positive evidence: the
  hybrid ML-KEM groups exist only for TLS 1.3. So it is now `classical`, negotiated, curve
  unknown, confidence 0.9.

Effect on `results/experiments/triage.md` (regenerated). **Exposed shares are unchanged**,
because both buckets count as exposed. The evidence behind them got stronger: flows move from
`at_risk_ceiling` (client side only) to `at_risk` (negotiated).

| dataset | `at_risk` (Phase 6 → 7) | `at_risk_ceiling` (Phase 6 → 7) | TLS versions (Phase 6 → 7) |
| --- | --- | --- | --- |
| PostQuantumTLS | 10,443 → **12,997** | 2,575 → **21** | 1.3 10,540 / 1.2 2,478 → 1.3 8,558 / 1.2 **4,460** |
| ISCX-VPN | 1,554 → 2,343 | 952 → 163 | 1.0 counts rise slightly (the server's version now wins) |
| ISCX-Tor | 3,699 → 5,474 | 2,149 → 374 | 〃 |

The PostQuantumTLS row in `PHASE-6.md` §10 ("1.3 (81 %)") is corrected to **1.3 66 %, 1.2
34 %** by this.

## 7. Status

| # | task | state |
| --- | --- | --- |
| 7.1 | certificate chain / signatures | ✅ §3: datasets and 756/756 live checks |
| 7.2 | HNDL risk + mitigations | ✅ §4: in records and `countdown triage` |
| 7.3 | E1 `file_transfer` | ◐ §5: byte-weighted tier error 5.1 → 1.7 pts; per-flow error remains |
| 7.4 | PQC census | ✅ §6: 1,000 sites, 60 % quantum-safe, 0 wrong passive verdicts |

Suite **608 / 608** in Docker (575 before).

**Known limits carried forward:**
- E1's per-flow class mix (labels are per capture);
- the census is one run from one network;
- certificate analysis is TLS 1.2 only (by construction);
- the HNDL weights are a judgement;
- the earlier limits in Phase 6 §15 (E1 cross-capture transfer, E3's 92 test flows,
  refit-on-train zoo).

**Next candidates:**
- repeat the census over time or from other networks, plus a QUIC/HTTP3 profile (the census is
  TCP-only);
- flow-level relabelling of the ISCX file-transfer captures;
- ECH handling, and streaming/eBPF prefiltering (Phase 6 §7).
