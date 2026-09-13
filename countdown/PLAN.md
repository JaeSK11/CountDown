# CountDown — Phased Build Plan

**Goal:** A pipeline that (1) pulls packets, (2) detects the encryption/crypto in use,
(3) decides whether it is **quantum-resistant**, and (4) for **non-PQC** flows, applies an
**ensemble of ML models** to identify the type of encrypted traffic.

## Terminology: pipeline stages vs build phases

The runtime **pipeline has 4 stages**; the **build roadmap has 7 phases**. The ML ensemble
(the "which model for which group" work) is **Pipeline Stage 4**, built across **Phases 3–5**.

```
PIPELINE:   [1 Ingest] → [2 Crypto detect] → [3 PQC verdict] → [4 ML ensemble]
BUILD:      P0 setup · P1(stage1) · P2(stages2-3) · P3-P5(stage4) · P6 integrate/eval
```

Key invariant: **Stages 2–3 are deterministic parsing (no ML).** ML is only Stage 4.
Quantum-resistance can only be read from a directly-observable TLS/QUIC handshake; tunneled
traffic (VPN/Tor) gets verdict = "crypto not observable" and routes straight to Stage 4.

---

## Phase 0 — Project setup & data loaders
**Pipeline:** foundation · **Deliverable:** `load + featurize any dataset → (X, y)`
- Repo scaffold, Python env, deps (scapy/pyshark, dpkt, scikit-learn, lightgbm, torch, dgl/pyg).
- `data/` symlink to `../data`; unified dataset loaders (ISCXVPN2016, ISCXTor2016, CSTNET-TLS1.3,
  PostQuantumTLS, MobileAppActivity) → common flow/label schema.
- Feature extractors (start with flow-stats + packet-sequence): `features/flow_stats.py`,
  `features/packet_seq.py`.
- **Exit:** each dataset loads to a labeled feature table + a sequence tensor.

## Phase 1 — Ingest + flow reassembly  → Pipeline Stage 1
**Deliverable:** pcap → list of flows with metadata
- `sources/base.py` (Source interface), `sources/pcap.py` (PcapSource). LiveSource deferred to P6.
- `flows/reassembly.py` — 5-tuple bidirectional flow assembly; per-flow first-N packets,
  sizes/directions/timings; handshake bytes extraction.
- **Exit:** deterministic flow objects from any pcap in the datasets.

## Phase 2 — Crypto detection + PQC verdict  → Pipeline Stages 2–3  ★ novel core
**Deliverable:** per-flow crypto verdict + PQC-detection precision/recall
- `crypto/tls_parser.py` — parse ClientHello/ServerHello: TLS version, cipher suite,
  `supported_groups` + `key_share` (the KEM named group). `crypto/quic_parser.py` for TLS-in-QUIC.
- `crypto/kem_registry.py` — codepoint → {classical, hybrid, pqc}. e.g. classical `x25519`(0x1D),
  `secp256r1`(0x17); PQC/hybrid `X25519MLKEM768`(0x11EC), `X25519Kyber768Draft00`(0x6399).
- `crypto/pqc_verdict.py` — resistant / hybrid / classical-vulnerable / tunneled-unobservable.
- **Validation data:** PostQuantumTLS (real PQC handshakes) + CSTNET-TLS1.3 (classical TLS 1.3).
  *Caveat:* 2023-era PQC positives may be sparse → supplement with self-generated PQC handshakes
  (Cloudflare/Google now negotiate X25519MLKEM768; or openssl + oqs-provider) for a clean positive set.
- **Exit:** verdict on every flow; measured PQC-detection precision/recall.

## Phase 3 — ML ensemble foundation  → Pipeline Stage 4 (part 1)  ★ ensemble starts here
**Deliverable:** runnable single-model classifier + metrics
- `models/base.py` — `BaseModel` interface (`fit/predict/predict_proba/save/load`) + registry
  so models self-register into the ensemble.
- First member: `models/flow_gbdt.py` — **LightGBM** on flow stats, trained on **traffic-type**
  (ISCXVPN/Tor). (GBDT replaces the papers' Random Forest — strict upgrade for tabular.)
- `eval/metrics.py` — accuracy / macro-F1 / confusion matrix, stratified CV, class-imbalance handling.
- **Exit:** `train flow_gbdt on ISCXVPN traffic-type → report metrics`.

## Phase 4 — Grow the ensemble (build each model type)  → Pipeline Stage 4 (part 2)
**Deliverable:** multiple independently-trained base models + per-model scorecards
- Add members one type at a time, each self-contained + independently trained + calibrated:
  | Module | Model | Primary target |
  | --- | --- | --- |
  | `models/seq_cnn.py` | 1D-CNN (DF-style) | universal deep voter |
  | `models/seq_ssm.py` | TCN → Mamba | long sequences (activity, app-ID) |
  | `models/byte_transformer.py` | ET-BERT / YaTC | app-ID (CSTNET 120) |
  | `models/graph_gnn.py` | GraphSAGE / GIN | relational (app-ID) |
  | `models/flow_image_cnn.py` | 2D-CNN (FlowPic) | in-app activity |
- Domain experts (Axis A): `experts/{tls,vpn,tor,app,activity}.py`, each trained on its dataset.
- **Model→target guidance:** traffic-type & tunnel-type → GBDT (+CNN); app-ID → byte-transformer
  / GNN; in-app activity → seq-CNN / TCN / FlowPic.
- **Exit:** each model type trains + evaluates standalone with a saved artifact + scorecard.

## Phase 5 — Router + combiner (ensemble integration)  → Pipeline Stage 4 (part 3)
**Deliverable:** full ensemble + single-vs-ensemble ablation
- `ensemble/router.py` — MoE: Stage-1 protocol/tunnel detection routes to the domain expert(s).
- `ensemble/combiner.py` — **calibrate each model (temperature/Platt) → stack with a LightGBM
  meta-learner** over `predict_proba` (beats naive soft-voting).
- **Exit:** ensemble > best single model on held-out data; ablation table.

## Phase 6 — End-to-end pipeline + eval framework + live capture
**Deliverable:** paper-ready results + working prototype
- `ensemble/pipeline.py` + `cli.py`: pcap in → per-flow verdict
  `{proto, tls_ver, kem, pqc_verdict, traffic_class, confidence}` out (JSON/CSV).
- `eval/experiments.py` — cross-dataset experiments, reproducible tables, PQC-triage statistics.
- `sources/live.py` — real-time capture (scapy/pyshark).
- **Exit:** one command runs the whole pipeline; experiment tables reproduce.

---

## Dataset → stage map
| Dataset | Stage 2–3 (PQC) | Stage 4 (ML) |
| --- | --- | --- |
| PostQuantumTLS | ✓ PQC positives | — |
| CSTNET-TLS1.3 | ✓ classical TLS 1.3 | ✓ app-ID (120) |
| ISCXVPN2016 | ✗ tunneled | ✓ traffic-type / tunnel-type |
| ISCXTor2016 | ✗ tunneled | ✓ traffic-type / tunnel-type |
| MobileAppActivity | ✗ (Wi-Fi) | ✓ in-app activity (92) |

## Model choices (summary)
- **Tabular flow stats:** LightGBM/XGBoost (default everywhere; upgrade over RF).
- **Sequences:** 1D-CNN (DF) baseline → TCN / CNN+Mamba for long flows.
- **Bytes:** ET-BERT or YaTC (app-ID).
- **Graph:** GraphSAGE / GIN.
- **Image:** 2D-CNN / FlowPic.
- **Combiner:** per-model calibration → LightGBM stacking meta-learner.

## References (see ../reference/papers/)
DF, ET-BERT/CSTNET, TFE-GNN, Okonkwo CNN + WF papers, Generalized Inter-Flow, Sensors-2022,
RADCOM repo (RF + TLS first-packet-size features).
