# Phase 6 — End-to-End Pipeline, Eval Framework & Live Capture

**Roadmap ref:** `../PLAN.md` → Phase 6. **Depends on:** all prior phases.
**Goal:** assemble Stages 1→4 into one runnable pipeline that realizes the thesis —
**pull packets → detect crypto → is it quantum-resistant? → if not, classify the encrypted traffic
type** — plus a reproducible **research eval framework** and **live capture**.

**One-line exit criterion:** `countdown analyze <pcap>` emits per-flow verdicts
`{five_tuple, l7, tls_version, cipher, kem_group, pqc_verdict, traffic_class?, confidence}`, and
`countdown experiments run --all` regenerates every results table (PQC-triage %, per-target
macro-F1, ensemble ablation, cross-dataset) deterministically.

---

## 1. Scope
**In scope**
- `pipeline/pipeline.py` — wire Source → Reassembler (P1) → crypto/PQC verdict (P2) → **branch**:
  PQC/hybrid → mark quantum-safe, stop; classical/not-observable → **ensemble classify** (P5).
- `cli.py` — `countdown {analyze, live, train, eval, experiments}` console entry.
- `sources/live.py` — implement the P1 `LiveSource` stub (real-time capture).
- `eval/experiments.py` + `experiments/` configs — cross-dataset runs, results tables, **PQC-triage
  statistics** (the headline research output), reproducible + seeded.
- `eval/report_paper.py` — emit LaTeX/markdown tables + figures.
- Packaging: quickstart README, model zoo (trained artifacts), example pcaps.

**Out of scope (future work)**
- Active defenses / mitigation, cert-chain PQC-signature analysis at scale, distributed/streaming
  deployment, ECH inner recovery — listed under §7.

---

## 2. Dependencies
Runtime: prior phases only. Live: `scapy`/`pyshark` (or AF_PACKET) — needs `CAP_NET_RAW`/root.
Reporting: optional `matplotlib`/`jinja2` for tables/figures.

---

## 3. Directory additions
```
countdown/
├── pipeline/
│   └── pipeline.py         # Stages 1→4 orchestration + branch logic
├── sources/live.py         # real-time capture (implements P1 interface)
├── eval/
│   ├── experiments.py      # cross-dataset, PQC-triage stats, ablation runner
│   └── report_paper.py     # LaTeX/markdown tables + figures
├── cli.py                  # countdown {analyze,live,train,eval,experiments}
experiments/                # experiment configs (seeded, reproducible)
results/                    # generated tables/figures (gitignored)
examples/                   # sample pcaps + expected output
```

---

## 4. Components

### 4.1 End-to-end pipeline (`pipeline/pipeline.py`)
Per flow, in order:
1. **Stage 1** reassemble → flow + handshake bytes + `l7_hint`.
2. **Stage 2–3** parse handshake → `{tls_version, cipher, kem_group}` → **PQC verdict**.
3. **Branch (the thesis):**
   - `pqc` / `hybrid` → emit verdict `quantum_safe`, **skip classification** (optional: still tag
     traffic_class if configured).
   - `classical` / `not_observable` → **Stage 4 ensemble** → `traffic_class`/`app` + confidence.
4. Emit a **verdict record** (JSON/CSV row).
- Modes: `full` (all ensemble members) and `fast` (GBDT+CNN, soft-vote) for real-time.
- **DoD:** a pcap yields verdict records; branch behaviour matches config.

### 4.2 CLI (`cli.py`)
- `countdown analyze <pcap> [--out csv|json] [--mode full|fast]`
- `countdown live <iface> [--bpf ...] [--mode fast]`
- `countdown train|eval` — unify P3–P5 training/eval behind one entry.
- `countdown experiments run [--all|--name X]` — regenerate results.
- **DoD:** `countdown analyze` on an example pcap prints/writes verdicts.

### 4.3 Live capture (`sources/live.py`)
- Implements the P1 `Source` interface for a live NIC (scapy/pyshark/AF_PACKET); streaming reassembly
  with real-time (wall-clock) flow expiry + bounded memory + backpressure.
- Requires `CAP_NET_RAW`; document setcap/sudo. `fast` mode default for real-time latency.
- **DoD:** classifies live TLS flows on a loopback/replayed capture; graceful on partial flows.

### 4.4 Research eval framework (`eval/experiments.py`)
- **PQC-triage statistics** (headline): over a dataset, the % `pqc`/`hybrid`/`classical`/
  `not_observable`; and among **classical (vulnerable)** flows, the traffic-type/app breakdown — the
  core result ("what vulnerable traffic looks like").
- **Per-target results:** macro-F1 tables for traffic_type / app / activity / tunnel_type.
- **Ensemble ablation** (from P5) + **cross-dataset generalization** (train on X → test on Y).
- Seeded, config-driven; one command regenerates all tables → `results/`.
- **DoD:** `experiments run --all` reproduces identical tables across runs.

---

## 5. Work breakdown & Definition of Done
| # | Task | DoD |
| --- | --- | --- |
| 6.1 | `pipeline/pipeline.py` Stages 1→4 + branch | pcap → verdict records; branch correct |
| 6.2 | `cli.py analyze` | one-command pcap→verdicts (json/csv) |
| 6.3 | `sources/live.py` | real-time verdicts on replayed/loopback traffic |
| 6.4 | `cli.py live` (+ fast mode) | live capture classifies TLS flows |
| 6.5 | `eval/experiments.py` PQC-triage + per-target | tables generated to `results/` |
| 6.6 | cross-dataset + ablation experiments | generalization table produced |
| 6.7 | `eval/report_paper.py` | LaTeX/markdown tables + figures |
| 6.8 | packaging: README quickstart, model zoo, examples | fresh clone → `analyze` works |

---

## 6. Acceptance test (end of Phase 6)
```
# 1) end-to-end analyze
$ countdown analyze examples/cstnet_sample.pcap --out verdicts.csv
#   rows: five_tuple,l7,tls_version,cipher,kem_group,pqc_verdict,traffic_class,confidence

# 2) the thesis in action
#   - a hybrid/PQC flow  -> pqc_verdict=hybrid,   traffic_class=(skipped/optional)
#   - a classical flow   -> pqc_verdict=classical, traffic_class=streaming (conf 0.8x)
#   - a Tor flow         -> pqc_verdict=not_observable, traffic_class=<ensemble>

# 3) reproducible research tables
$ countdown experiments run --all      # -> results/{pqc_triage,per_target,ablation,cross_dataset}.*

# 4) live
$ sudo countdown live eth0 --bpf "tcp port 443" --mode fast   # streams verdicts
```
**Outcome:** a working prototype **and** a reproducible eval framework — the full research artifact.

---

## 7. Future work (beyond Phase 6)
- PQC **signature**-chain analysis where observable (TLS 1.2 / CertificateRequest).
- Mitigation guidance per vulnerable flow (harvest-now-decrypt-later risk scoring).
- ECH-aware handling as adoption grows; QUIC v2+ coverage.
- Streaming/distributed deployment; on-NIC/eBPF prefiltering for line-rate.
- Concept-drift monitoring + periodic retraining of Stage-4 models.

## 8. Risks
- **Live perf/root:** capture + reassembly + inference at rate → `fast` mode, bounded buffers,
  optional BPF prefilter; document capabilities.
- **Cross-dataset domain shift:** expect macro-F1 drops train≠test; report honestly (it's a result).
- **Reproducibility drift:** pin seeds/versions; freeze configs with each results table.
- **Privacy:** verdict records may contain IPs/SNI → add a `--redact` option for shareable output.
