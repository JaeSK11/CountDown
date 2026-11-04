# CountDown

Detect the cryptography in use on a network flow, decide whether it is quantum-resistant,
and — for flows whose crypto is not observable — classify the encrypted traffic with an
ensemble of ML models.

```
PIPELINE:   [1 Ingest] → [2 Crypto detect] → [3 PQC verdict] → [4 ML ensemble]
BUILD:      P0 setup · P1(stage1) · P2(stages2-3) · P3-P5(stage4) · P6 integrate/eval · P7 follow-ups
```

Key invariant: **stages 2–3 are deterministic parsing, no ML.** Quantum-resistance is read
from a directly-observable TLS/QUIC handshake; tunnelled traffic (VPN/Tor) gets the verdict
"crypto not observable" and routes straight to stage 4.

See [`PLAN.md`](PLAN.md) for the full roadmap and [`phases/`](phases/) for per-phase specs.

---

## Status: Phases 0–7 complete (Phase 7: `phases/PHASE-7.md`)

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

Headline scorecards (macro-F1, leak-free grouped splits; details in each `MODEL-*.md`):

| member | target | baseline | recommended | `flow_gbdt` floor |
| --- | --- | --- | --- | --- |
| graph | CSTNET app (120), 5-fold | 0.821 | **0.865** | 0.811 |
| bytes | ET-BERT packet corpus, seq/ack randomised | 0.566 | **0.857** | — |
| image | ISCX-VPN traffic (4), 5 seeds | 0.864 | **0.938** | — |
| image | MobileApp activity (92) | 0.115 | 0.264 | **0.441** |
| sequence | `iscx_pooled` traffic (8) | 0.547 | 0.616 | **0.645** |

Only the graph member beats the tabular floor on its home target; the others go into Phase 5 as
decorrelated voters.

### Phase-5 status

`countdown/ensemble/` calibrates each member (temperature), stacks or votes over leakage-safe
out-of-fold predictions, and routes each flow to one expert from its Stage-1 hint. Macro-F1 on
the grouped test split, with the combiner chosen by cross-validation inside train
(`phases/PHASE-5.md` §10):

| expert | best single member | ensemble | |
| --- | --- | --- | --- |
| E2 app-ID (CSTNET, 120) | `graph_gnn` 0.868 | GBDT + GIN, calibrated soft vote **0.890** | ✓ |
| E3 activity (MobileApp, 92) | `flow_gbdt` 0.441 | 3-member vote 0.513 | ✓ weak: 92 test rows |
| E1 traffic type (`iscx_pooled`, 8) | `flow_gbdt` 0.665 | 3-member vote 0.665 | ✗ tie — ship the GBDT |

The soft vote beat logistic and LightGBM stacking on every expert. Long jobs run in Docker
(`docker/README.md`).

### Quickstart: analyze a capture

```bash
countdown models fetch                                 # once: model zoo (private release), hash-verified
countdown analyze examples/*.pcap --mode fast          # the shipped example handshakes
countdown analyze capture.pcap --out verdicts.csv
countdown live eth0 --port 443                         # live (needs CAP_NET_RAW: see docker/README.md)
```

Each flow gets its PQC verdict; a negotiated hybrid/PQC exchange is marked quantum-safe and not
classified, everything else is routed to one expert and labelled (`phases/PHASE-6.md` §9).
`--mode fast` uses only the GBDTs; `--redact` drops IPs and SNI.

`countdown triage capture.pcap` adds the PQC-triage summary: how much traffic is quantum-safe,
exposed or unobservable (flows and bytes) and what the exposed traffic is. On the labelled
datasets **no flow is quantum-safe** and every observable flow is exposed (`PHASE-6.md` §10).

`countdown experiments run --all` regenerates every results table (`results/experiments/`)
byte for byte. One of them is a warning: the traffic-type model trained on one ISCX capture
campaign scores ~0.11 macro-F1 on the other (0.58 in-domain) — in-domain numbers here describe
these capture setups, not traffic in general (`PHASE-6.md` §11).

Paper tables and figures: `countdown experiments run paper` -> `results/paper/`. Live capture
of current HTTPS found the first real quantum-safe flows of the project: Cloudflare, Google and
Wikipedia negotiate X25519MLKEM768 today (`PHASE-6.md` §13).

**Phase 7** (`phases/PHASE-7.md`):
- Every record now carries the TLS 1.2 certificate verdict (`auth_verdict`, `cert_key`), a
  harvest-now-decrypt-later score and the mitigation codes that would remove the exposure
  (`hndl_score`, `hndl_level`, `mitigations`). `countdown triage` ranks the changes by the bytes
  they cover.
- E1 applies a per-class decision bias (`configs/decision_bias.yaml`; `--no-decision-bias` turns
  it off).
- A passive census of the Tranco top 1,000 (`scripts/pqc_census.py`) found **60 % of responding
  sites negotiate X25519MLKEM768**, with no wrong passive verdicts against the client's own
  report.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,viz]"
.venv/bin/python -m pytest -q
```

Or, with the exact package set and a GPU, in Docker (`docker/README.md`):

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm job python -m pytest -q
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
  models/tfe_gnn.py       TFEGNNNet towers (SAGE or GIN); models/graph_gnn.py registers them [P4]
  training/deep.py        shared torch trainer: early stop, checkpoint/resume, AMP [P4]
  eval/splits.py          group-aware splits + the vacuous-grouping guard          [P3]
  eval/metrics.py         macro-F1 headline, per-class, Dummy baseline             [P3]
  eval/report.py          reproducible runs/<id>/ directories                      [P3]
  training/train.py       load -> split -> fit -> evaluate -> persist              [P3]
  experts/                E1 / E2 / E3 bindings (dataset, target, members)         [P4]
  ensemble/calibration.py temperature / isotonic, ECE, CalibratedModel             [P5]
  ensemble/stacking.py    grouped out-of-fold predictions (leakage-safe)           [P5]
  ensemble/combiner.py    soft vote, logistic / LightGBM stacking                  [P5]
  ensemble/router.py      Stage-1 hint / capture context -> one expert             [P5]
  ensemble/views.py       one flow table per expert; per-member input views        [P5]
  ensemble/ensemble.py    Ensemble: fit (OOF + calibrate + combine), explain, save [P5]
  eval/ablation.py        single vs vote vs stacking x calibration table           [P5]
  pipeline/pipeline.py    pcap -> reassembly -> PQC verdict -> route -> classify   [P6]
  pipeline/artifacts.py   DeployedExpert (members + recipes + calibrators + vote)  [P6]
  cli.py, __main__.py     `countdown analyze|triage|live|models|experiments`      [P6]
  pipeline/zoo.py         model zoo: fetch from the release, SHA-256 manifest      [P6]
  sources/live.py         AF_PACKET live capture with idle heartbeats              [P6]
  eval/report_paper.py    LaTeX tables + figures -> results/paper/                 [P6]
  eval/triage.py          PQC-triage: exposure by flows/bytes + what is exposed    [P6]
  eval/cross_dataset.py   train on one capture setting, test on another            [P6]
  eval/experiments.py     `countdown experiments run` -> results/experiments/      [P6]
  eval/tiers.py           readiness tiers (configs/readiness_tiers.yaml)           [P6]
configs/experiments/      one YAML per Phase-3 experiment
runs/                     per-run artifacts (gitignored)
scripts/train.py          training CLI
scripts/summarize_runs.py tabulate runs/ into the results table
scripts/summarize_datasets.py
scripts/replicate_drapergil.py, replicate_okonkwo.py, replicate_tfegnn.py, repro_etbert.py
                          paper replications (write to runs/)                        [P4]
scripts/build_tfegnn_cache.py  pcap -> TFE-GNN byte shards under cache/tfegnn/        [P4]
scripts/run_ensemble.py   fit one expert's ensemble + ablation (runs/ensemble/)      [P5]
scripts/reparse_check.py  re-parse a dataset, diff flows against the old cache       [P5]
scripts/collect_results.py  copy scorecard evidence from runs/ into tracked results/
docker/                   training image, compose file, GPU queue runner             [P5]
scripts/export_models.py  Phase-5 runs -> models_store/ deployed experts             [P6]
scripts/check_pipeline_skew.py  pipeline vs Phase-5 predictions (must agree exactly) [P6]
scripts/run_triage.py     PQC-triage tables over the labelled datasets              [P6]
models_store/             deployed experts (gitignored; `countdown models fetch`)    [P6]
models_store.sha256       hashes of the published zoo -- what makes a download trusted [P6]
examples/                 our synthetic PQC handshake captures + expected output    [P6]
scripts/publish_models.sh publish models_store/ as a release + write the manifest   [P6]
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
