# Example captures

Three single-handshake captures we generated ourselves (`scripts/gen_pqc_pcaps.sh`, Phase 2,
task 2.8) -- the only pcaps in this project that are ours to redistribute. Each is one TLS 1.3
handshake from a LAN client to a public server:

| file | what happens | verdict |
| --- | --- | --- |
| `hybrid_x25519mlkem768.pcap` | client offers X25519MLKEM768, server selects it | `hybrid` / negotiated -- quantum-safe, not classified |
| `classical_x25519.pcap` | classical x25519 only | `classical` / negotiated -- exposed, classified |
| `downgrade_offered_pqc.pcap` | client offers ML-KEM, the server picks x25519 | `classical` / negotiated -- exposed (offering PQC is not getting it) |

```bash
countdown models fetch                         # once, on a fresh clone
countdown analyze examples/*.pcap --mode fast  # compare with expected_fast.jsonl
countdown triage  examples/*.pcap --mode fast
```

`expected_fast.jsonl` is the `--mode fast --redact` output with the model zoo `models-v1` and
E1's per-class decision bias (`configs/decision_bias.yaml`, Phase 7; `--no-decision-bias` gives
the plain argmax) -- `tests/test_examples.py` checks it. Since Phase 7 each record also carries
`auth_verdict` / `cert_key` (not observable here: TLS 1.3 encrypts the certificate) and a
harvest-now-decrypt-later score with mitigation codes: the downgrade capture gets
`enable_hybrid_kem` (the client already offers ML-KEM, so the fix is server-side), the
classical one `enable_hybrid_kem_both`. Since Phase 10 all three are `traffic_class: "handshake_only"`
(`route_reason: "handshake_only"`, no expert consulted): each is a handshake with no
application data after it, which no traffic expert was trained on -- before, E1 named the two
exposed ones `voip` and `chat` at 0.40 and 0.65 confidence. The PQC verdicts and risk scores
are unchanged.
