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

`expected_fast.jsonl` is the `--mode fast --redact` output with the model zoo `models-v1`
(`tests/test_examples.py` checks it). The class labels of the two exposed flows come from the
traffic-type expert after the E2 -> E1 fallback; a synthetic single handshake is not traffic
any expert was trained on, so read them as a demonstration of the routing, not as a result.
