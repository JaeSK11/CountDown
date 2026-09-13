# Phase 4 — Grow the Ensemble (Pipeline Stage 4, part 2)

**Roadmap ref:** `../PLAN.md` → Phase 4. **Depends on:** Phase 3 (`BaseModel`, registry, eval
harness, `LabelSpace`), Phase 0 (features).
**Goal:** add each remaining **model type** behind the same `BaseModel` seam, and wire the **domain
experts** (Axis A). Every model type is built as **two variants — a paper baseline and our
recommended upgrade** — so the upgrade is *measured*, not assumed. Each model trains and evaluates
standalone; the ensemble combiner is Phase 5.

**One-line exit criterion:** each model type has a registered `BaseModel` (baseline + recommended),
trains via `scripts/train.py --model <name>`, and produces a **baseline-vs-recommended scorecard**
(macro-F1, #params, latency) on its target dataset — all `predict_proba` aligned to `LabelSpace`.

---

## 1. Two axes of "type"

**Axis B — representation base-learners** (each has its own plan in `phase4-models/`):

| Plan | input_type | Baseline (paper) | Recommended (our chat) | Primary target |
| --- | --- | --- | --- | --- |
| `MODEL-flow-stats.md` | flow_stats | Random Forest (RADCOM) | **LightGBM/XGBoost** *(built in P3)* | traffic/tunnel type |
| `MODEL-sequence.md` | packet_seq | **DF** 1D-CNN (Sirinam) | dilated-residual CNN → **CNN+Mamba** | traffic type / activity |
| `MODEL-bytes.md` | bytes | **ET-BERT** (CSTNET) | **YaTC** (MAE transformer) | app-ID (CSTNET 120) |
| `MODEL-graph.md` | graph | **TFE-GNN** (GCN) | **GraphSAGE / GIN** | app-ID / relational |
| `MODEL-image.md` | image | **Okonkwo** scatter-CNN (AISC'22) | **FlowPic** ResNet / ViT | activity / traffic type |

**Axis A — domain experts** (`experts/`): a `(model, dataset, target)` binding, routed by Stage-1
protocol in Phase 5. Any base learner can back an expert.

| Expert | Dataset | Target | Default backing model |
| --- | --- | --- | --- |
| `TLSExpert` | CSTNET / PostQuantumTLS | app | bytes (ET-BERT/YaTC) |
| `VPNExpert` | ISCXVPN | traffic_type | flow_stats (GBDT) + sequence |
| `TorExpert` | ISCXTor | traffic_type | flow_stats + sequence |
| `AppExpert` | CSTNET | app (120) | bytes / graph |
| `ActivityExpert` | MobileApp | activity (92) | sequence / image |

---

## 2. Scope

**In scope**
- New feature extractors: `features/payload_bytes.py`, `features/flow_image.py`,
  `features/graph.py` (flow_stats + packet_seq already exist from P0).
- A shared **deep-training harness** (`training/deep.py`): torch training loop — epochs, early
  stopping, checkpointing, GPU, class-weighted loss, seed — reused by all deep models.
- Each model type: **baseline + recommended** variants, registered, each with `predict_proba`.
- Domain expert wiring (`experts/`).
- Per-model **baseline-vs-recommended scorecards**.

**Out of scope (later)**
- Calibration, stacking, router → Phase 5 (the `predict_proba` seam is already reserved).
- End-to-end CLI / live capture → Phase 6.

---

## 3. Dependencies (added this phase)
`torch`, `torch-geometric` **or** `dgl` (graph), optional `mamba-ssm` (falls back to a pure-torch
S4/SSM block if the CUDA kernel isn't available), optional `timm` (image backbones/ViT). Byte
transformers implemented in-house (ET-BERT/YaTC-style) to avoid a heavy `transformers` dependency —
decided per `MODEL-bytes.md`.

---

## 4. Directory additions
```
countdown/
├── features/
│   ├── payload_bytes.py     # first-N bytes per packet → token/byte matrix
│   ├── flow_image.py        # scatter / FlowPic 2D histogram
│   └── graph.py             # per-flow traffic graph (nodes/edges)
├── models/
│   ├── seq_cnn.py           # DF baseline + dilated-residual/CNN-SSM  (MODEL-sequence)
│   ├── byte_net.py          # ET-BERT baseline + YaTC                 (MODEL-bytes)
│   ├── graph_gnn.py         # TFE-GNN baseline + GraphSAGE/GIN        (MODEL-graph)
│   └── flow_image_cnn.py    # Okonkwo baseline + FlowPic ResNet/ViT   (MODEL-image)
├── experts/
│   ├── base.py              # Expert = (model, dataset, target) binding
│   └── {tls,vpn,tor,app,activity}.py
└── training/
    └── deep.py              # shared torch trainer
phases/phase4-models/
├── MODEL-flow-stats.md      MODEL-sequence.md   MODEL-bytes.md
├── MODEL-graph.md           MODEL-image.md
```

---

## 5. Common contract for every Phase-4 model
- Subclass `BaseModel`; register `@register("<name>_baseline")` and `@register("<name>")`
  (recommended). Same `fit/predict/predict_proba/save/load`; `predict_proba` in `LabelSpace` order.
- Consume exactly one `input_type` (its feature extractor); deep models share `training/deep.py`.
- Evaluated with the **Phase-3 harness** (group-aware split, macro-F1 headline) so numbers are
  comparable across all models and to `flow_gbdt`.
- Each model plan ends with a **baseline-vs-recommended table** (macro-F1, #params, train/infer time).

---

## 6. Work breakdown & Definition of Done
| # | Task | DoD |
| --- | --- | --- |
| 4.1 | `training/deep.py` shared trainer | trains a toy net; early stop + checkpoint + GPU |
| 4.2 | `features/payload_bytes.py` | (N, L) byte matrix per flow |
| 4.3 | `features/flow_image.py` | FlowPic/scatter image tensor per flow |
| 4.4 | `features/graph.py` | per-flow graph object (PyG/DGL) |
| 4.5 | `MODEL-sequence` (DF + CNN-SSM) | both registered; scorecard produced |
| 4.6 | `MODEL-bytes` (ET-BERT + YaTC) | both registered; scorecard on CSTNET |
| 4.7 | `MODEL-graph` (TFE-GNN + SAGE/GIN) | both registered; scorecard |
| 4.8 | `MODEL-image` (Okonkwo + FlowPic) | both registered; scorecard |
| 4.9 | `MODEL-flow-stats` (RF baseline vs P3 GBDT) | comparison table |
| 4.10 | `experts/` bindings | each expert trains its default model on its dataset |

---

## 7. Acceptance test (end of Phase 4)
```
from countdown.models import ModelRegistry
ModelRegistry.list()   # includes: flow_gbdt, seq_cnn_baseline/seq_cnn,
                       #  byte_net_baseline/byte_net, graph_gnn_baseline/graph_gnn,
                       #  flow_image_cnn_baseline/flow_image_cnn
# each trains standalone and beats flow_gbdt on its home target where expected (app-ID, activity)
# each model plan has a baseline-vs-recommended scorecard committed under runs/
```
**Then Phase 5 calibrates every model's `predict_proba` and stacks them + adds the router.**

---

## 8. Risks
- **GPU/compute:** deep models need a GPU (you have an RTX A4000 per the WF paper's setup); keep
  batch/seq configurable; provide CPU-small configs for CI.
- **Fair comparison:** baseline vs recommended must use the **same split/features/target** — enforce
  via shared configs so the scorecard is honest.
- **Overfitting on small datasets** (ISCX): augmentation + early stopping + group-aware split.
- **Dependency weight:** GNN/transformer libs are heavy; keep each model import-isolated so a missing
  optional dep doesn't break the whole registry (lazy import + capability flags).

## 9. Per-model plans
See `phase4-models/`: each specifies the paper baseline, the recommended upgrade, shared training,
and the comparison scorecard.
