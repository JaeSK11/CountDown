# CountDown

Detect the cryptography in use on a network flow, decide whether it is quantum-resistant,
and — for flows whose crypto is not observable — classify the encrypted traffic with an
ensemble of ML models.

```
PIPELINE:   [1 Ingest] → [2 Crypto detect] → [3 PQC verdict] → [4 ML ensemble]
BUILD:      P0 setup · P1(stage1) · P2(stages2-3) · P3-P5(stage4) · P6 integrate/eval
```

Key invariant: **stages 2–3 are deterministic parsing, no ML.** Quantum-resistance is read
from a directly-observable TLS/QUIC handshake; tunnelled traffic (VPN/Tor) gets the verdict
"crypto not observable" and routes straight to stage 4.

See [`PLAN.md`](PLAN.md) for the full roadmap and [`phases/`](phases/) for per-phase specs.

---

## Status: Phases 0–3 complete, Phase 4 in progress

Phase 0 delivers the data layer: **any dataset → `(X, y)`**.

```python
from countdown.data import load

ds = load("iscxvpn", target="traffic_type")   # parses (then caches) → FlowDataset
X  = ds.features("flow_stats")                # (N, 56)   float32 tabular
S  = ds.features("packet_seq")                # (N, 32, 2) float32 sequence tensor
y  = ds.labels()                              # (N,)      int64
ds.class_counts()                             # {"voip": 6557, "file_transfer": 4126, ...}
idx = ds.split(test_size=0.2, seed=42)        # stratified train/val/test indices
```

Crypto detection and the PQC verdict (P2) are in `countdown.crypto`:

```python
from countdown.crypto import parse_handshake, pqc_verdict, enrich_flow

hs = parse_handshake(flow)          # TLS/QUIC handshake -> crypto parameters
v  = pqc_verdict(hs)                # -> classical | hybrid | pqc | not_observable | unknown
v.basis                             # 'negotiated' (the server's choice) or 'offered'
enrich_flow(flow)                   # writes tls_version / cipher_suite / kem_group / pqc_verdict
```

Phase 3 adds the ML layer — the `BaseModel` contract, leakage-safe splits and the first
ensemble member:

```python
from countdown.data import load
from countdown.schema import LabelSpace
from countdown.eval import group_split, evaluate
from countdown.models import ModelRegistry

ds    = load("iscx_pooled", target="traffic_type")   # ISCXVPN + ISCXTor, 8 classes
space = LabelSpace.taxonomy("traffic_type")          # 8 fixed proba columns  <- ensemble seam
X, y  = ds.features("flow_stats"), ds.labels(space=space)
g     = ds.groups("source_file")                     # 235 captures
tr, te = group_split(X, y, g, test_size=0.2, seed=42)

m = ModelRegistry.create("flow_gbdt", label_space=space).fit(X[tr], y[tr])
evaluate(m, X[te], y[te], space=space)["macro_f1"]
```

```bash
.venv/bin/python scripts/train.py --model flow_gbdt --dataset iscx_pooled --target traffic_type
```

### Phase-3 results

Every split is group-aware (no capture on both sides) and no class is missing from any test
fold. Macro-F1 is the headline, not accuracy. Regenerate with `scripts/summarize_runs.py`.

| dataset | target | classes | macro-F1 | baseline | accuracy | group key |
| --- | --- | --- | --- | --- | --- | --- |
| `iscx_pooled` | traffic_type | 8 | **0.632** | 0.050 | 0.884 | `source_file` |
| `iscx_pooled` | tunnel_type | 3 | **0.898** | 0.326 | 0.995 | `source_file` |
| `cstnet` | app | 120 | **0.811** | 0.000 | 0.838 | `capture_day` |
| `mobileapp` | activity | 92 | 0.441 | 0.000 | 0.489 | `source_file` (vacuous) |

Two things these numbers are saying:

* **`tunnel_type` is why accuracy is not the headline.** It moves 0.957 → 0.995 — nearly
  nothing — while macro-F1 moves 0.326 → 0.898.
* **CSTNET's 0.811 is the floor Phase 4 has to beat**, and it is higher than a "GBDT is
  just a baseline" framing would suggest. Flow statistics alone already separate most of
  the 120 services.

These read *lower* than much of the published work on ISCX, and deliberately so: a random
flow-level split puts near-duplicate flows from one capture on both sides. See
`phases/PHASE-3.md` §5.3.

### Phase-4 status

Each ensemble member is built as a **paper baseline first**, replicated against its published
numbers under both the paper's protocol and a leak-free one, before the recommended variant is
compared to it. Live status, blockers and open decisions: `phases/PHASE-4-HANDOFF.md`.

| member | registered today | paper baseline | write-up |
| --- | --- | --- | --- |
| flow-stats | **`flow_gbdt`** (P3), `flow_c45_paper`, `flow_knn_paper`, `flow_catboost` (bake-off only) | Draper-Gil C4.5 / k-NN | `phases/phase4-models/REPLICATION-DRAPERGIL.md` |
| image | `flow_image_cnn_baseline`, **`flow_image_cnn`** (FlowPic 3ch), `flow_image_resnet18` | Okonkwo scatter-CNN | `phases/phase4-models/REPLICATION-OKONKWO.md`, `MODEL-image.md` |
| bytes | `byte_net_baseline`, **`byte_net`** (+ seq/ack randomisation) | ET-BERT (public checkpoint) | `phases/phase4-models/MODEL-bytes.md` |
| graph | `graph_gnn_baseline`, **`graph_gnn`** (GIN) | TFE-GNN | `phases/phase4-models/REPLICATION-TFEGNN.md` |
| sequence | `seq_cnn_baseline` (DF), **`seq_cnn`** (dilated-residual) | Deep Fingerprinting | `phases/phase4-models/MODEL-sequence.md` |

All three replications reach the published number only under the paper's own (leaky) protocol;
the grouped column is the bar for every recommended variant.

Bold = the recommended member. Recommended members train through `training/deep.py`; baselines
keep their paper loops; no member uses class weights. `countdown/experts/` binds them into the
three ensembles of `phases/ENSEMBLE-MAP.md` (E1 traffic type, E2 app-ID, E3 activity). The
evidence behind every scorecard is copied from the gitignored `runs/` into the tracked
[`results/`](results/INDEX.md) by `scripts/collect_results.py`.

Not yet built: calibration + stacking + router (P5), live capture (P6).

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,viz]"
.venv/bin/python -m pytest -q
```

`viz` is optional and only adds the confusion-matrix PNG; every run writes `metrics.json`
with the matrix in it either way.

The Phase-4 deep members additionally need `torch` + `torchvision` (a CUDA build that matches
the GPU — an RTX 5090 needs CUDA ≥ 12.8) and `torch-geometric` for the graph member. They are
lazy-imported and not yet declared in `pyproject.toml`; the registry works without them.

## Layout

```
configs/datasets.yaml     per-dataset roots, globs, label-mapping rules, filters
configs/features.yaml     flow-assembly + feature-extractor params, SEED
countdown/
  schema.py               Packet / Flow / FlowDataset + the label taxonomy
  config.py               YAML loading, logging, cache-key hashing
  flows/extract.py        pcap|pcapng → flows (bidirectional 5-tuple)   [shared with P1]
  crypto/tls_parser.py    TLS records + ClientHello/ServerHello + extensions   [P2]
  crypto/quic_parser.py   QUIC Initial decrypt → CRYPTO frames → ClientHello   [P2]
  crypto/kem_registry.py  named-group codepoint → classical | hybrid | pqc     [P2]
  crypto/pqc_verdict.py   the verdict engine (negotiated vs offered basis)     [P2]
  crypto/data/*.yaml      versioned codepoint tables (IANA-derived)
  data/                   DatasetLoader ABC + registry + parquet cache + 5 loaders
  data/pooled.py          load_pooled([...]) -> one FlowDataset                    [P3]
  features/               FeatureExtractor ABC + flow_stats, packet_seq
  features/timeonly.py    Draper-Gil's 23 time-only features (paper baseline)          [P4]
  features/flow_image.py  scatter / FlowPic images, 1 or 3 channels                    [P4]
  features/payload_bytes.py  ET-BERT bi-gram tokens from handshake bytes               [P4]
  features/byte_prep.py   byte-retaining pcap pass for TFE-GNN (ISCX)                  [P4]
  features/traffic_graph.py  PMI byte graphs (TFE-GNN)                                 [P4]
  data/windows.py         capture-window sample unit (Okonkwo); data/app_labels.py     [P4]
  data/etbert_corpus.py   the released ET-BERT CSTNET corpus + endpoint-leakage audit  [P4]
  flows/segment.py        flow-timeout segmentation (Draper-Gil ftm)                   [P4]
  models/base.py          BaseModel ABC + ModelRegistry (the ensemble seam)        [P3]
  models/flow_gbdt.py     LightGBM on flow_stats -- the universal baseline member  [P3]
  models/paper_baselines.py  flow_c45_paper, flow_knn_paper                        [P4]
  models/flow_image_cnn.py   flow_image_cnn_baseline (Okonkwo); flow_image_resnet.py  [P4]
  models/byte_net.py      byte_net_baseline (ET-BERT transcription)                [P4]
  models/tfe_gnn.py       TFEGNNNet (not yet registered)                           [P4]
  training/deep.py        shared torch trainer: early stop, checkpoint/resume, AMP [P4]
  eval/splits.py          group-aware splits + the vacuous-grouping guard          [P3]
  eval/metrics.py         macro-F1 headline, per-class, Dummy baseline             [P3]
  eval/report.py          reproducible runs/<id>/ directories                      [P3]
  training/train.py       load -> split -> fit -> evaluate -> persist              [P3]
configs/experiments/      one YAML per Phase-3 experiment
runs/                     per-run artifacts (gitignored)
scripts/train.py          training CLI
scripts/summarize_runs.py tabulate runs/ into the results table
scripts/summarize_datasets.py
scripts/replicate_drapergil.py, replicate_okonkwo.py, replicate_tfegnn.py, repro_etbert.py
                          paper replications (write to runs/)                        [P4]
scripts/build_tfegnn_cache.py  pcap -> TFE-GNN byte shards under cache/tfegnn/        [P4]
scripts/gen_pqc_pcaps.sh  self-generated PQC/classical captures  → data/_pqc_synth/
scripts/validate_pqc.py   parser vs tshark oracle + PQC precision/recall
tests/
cache/                    parquet flow caches, keyed by config hash (gitignored)
data -> ../data           symlink to the dataset corpus
```

## Datasets

| name | source | one sample is | default target | classes |
| --- | --- | --- | --- | --- |
| `iscxvpn` | ISCXVPN2016, 140 captures | a 5-tuple flow | `traffic_type` | 7 |
| `iscxtor` | ISCXTor2016, 95 captures | a 5-tuple flow | `traffic_type` | 8 |
| `cstnet` | CSTNET-TLS1.3, ~46k pcaps | one TLS session | `app` | 120 |
| `postquantumtls` | PostQuantumTLS, 90 captures | a 5-tuple flow | `app` | 90 |
| `mobileapp` | MobileAppActivity-Sensors2022, 368 CSVs | one capture | `activity` | 92 |

Labels are unified across datasets. `traffic_type` ∈ {browsing, email, chat,
audio_streaming, video_streaming, file_transfer, voip, p2p}; the orthogonal fields
`is_vpn`, `is_tor` and `tunnel_type` ∈ {none, vpn, tor} live alongside it, and `target=`
picks which one becomes `y`.

```bash
.venv/bin/python scripts/summarize_datasets.py            # all five
.venv/bin/python scripts/summarize_datasets.py cstnet --top-k-apps 20 --max-per-app 50
```

## Design notes

Decisions that were made against the data rather than assumed:

* **Captures are sniffed by magic bytes, not by extension.** The PostQuantumTLS files are
  pcapng named `*.pcap`; trusting the suffix loses that dataset entirely.
* **Flow direction** prefers the TCP SYN sender, falls back to the service-port rule, and
  only then to first-seen. First-seen alone inverts every capture that starts mid-session,
  which many CSTNET sessions do.
* **MobileApp direction** comes from 802.11 addressing: an uplink frame has
  `wlan.ta == wlan.sa` (the station transmits its own traffic) while a downlink frame is
  relayed by the AP. The client MAC is inferred per capture from that split.
* **Packet size** is reconstructed as `L2 header + IP total length` rather than `len(buf)`,
  so snaplen-truncated captures still yield correct sizes. Sizes are capped at the MTU at
  *feature* time — the ISCXTor captures contain ~10 kB segmentation-offload superframes.
* **Unmapped filenames are a hard error.** ISCXVPN/ISCXTor encode labels in inconsistent
  filenames; a capture matching no rule raises rather than being silently dropped.
* **Flows keep `(ts, size, direction)` only.** That is the reduction that turns 90 GB of
  pcaps into a ~100 MB cache. Raw bytes have a reserved slot for Phase 1/4 to populate.
* **Caches are keyed by a config hash.** Changing a label rule or `flow_timeout`
  invalidates them automatically instead of serving stale flows.

### One sample means something different per dataset — read this before training

Flow-vs-capture granularity varies, and on ISCXTor it skews the label distribution hard:

| | captures | flows | packets |
| --- | --- | --- | --- |
| ISCXTor `Tor/` | 51 | **697** | 15.2 M |
| ISCXTor `NonTor/` | 44 | **53,732** | 12.2 M |

Tor multiplexes every application stream over a handful of long-lived guard connections, so
most Tor captures collapse to one or two flows while a single NonTor p2p capture explodes
into ~9,600. The two sides carry comparable *packet* volume; only the flow count differs.
Consequences for Phase 3:

* `tunnel_type` / `is_tor` on `iscxtor` is a ~77:1 imbalance that is an artefact of
  granularity, not of evidence. Weight by class, or resample, or evaluate per-capture.
* Per-flow `traffic_type` on `iscxtor` is likewise dominated by browsing and p2p (25,997 and
  23,054 of 54,429) because those NonTor captures fan out the most.
* Splits should ideally be grouped by source capture — flows from one pcap share a client,
  a session and a time window, so a random flow-level split leaks across train/test.
  `FlowDataset.split()` is stratified but *not* grouped; `flow.meta["source_file"]` carries
  what a grouped splitter would need.
