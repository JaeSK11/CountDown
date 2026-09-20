# Model: Sequence (packet size / direction sequence)

**input_type:** `packet_seq` · **feature:** `features/packet_seq.py` (Phase 0)
**Primary targets:** traffic_type (ISCXVPN/Tor), activity (MobileApp), app (CSTNET)
**Datasets:** ISCXVPN, ISCXTor, MobileAppActivity, CSTNET
**Ensemble role:** the universal deep voter — cheap, generalizes across targets.
**Input:** first `N` packets as **signed sizes** (`direction·min(size,1500)`) + IATs; `N` config
(32 for flow-type, up to 1000 for fine-grained/long).

> **Status (2026-09-19):** both variants built and registered — `seq_cnn_baseline` (DF, its
> paper loop, `features/dir_seq.py`) and `seq_cnn` (dilated-residual CNN through
> `training/deep.py`), `models/seq_cnn.py`, `tests/test_deep_members.py`. The Mamba/S4 stage is
> **deferred** (decision D1d): `mamba-ssm` needs a CUDA toolkit this machine does not have.

## Baseline (paper) — DF 1D-CNN
- **Reference:** Sirinam et al., "Deep Fingerprinting," CCS 2018 (`reference/…` WF family).
- **Architecture:** VGG-like 1D-CNN — 4 conv blocks, filters [32, 64, 128, 256]; each block =
  2× (Conv1D k=8 → BatchNorm → activation) → MaxPool(pool 8, stride 4) → Dropout. Activations: ELU
  in block 1, ReLU after. Head: FC[512,512] with dropout 0.7/0.5 → softmax.
- **Input, as built:** the paper's own — a length-5000 **direction-only** sequence from the new
  `dir_seq` extractor, not our signed sizes. Running DF on our richer `packet_seq` would not be a
  replication (`MODEL-DECISIONS.md`). Note what that means on this data: the median pooled ISCX
  flow is **9 packets** (p90 73), so >99 % of DF's input is padding there; on MobileApp the
  median capture is 101 packets.
- **Training, as published:** Adamax 2e-3, batch 128, 30 epochs, final model. Own loop (D3).
- **Register:** `@register("seq_cnn_baseline")` · `features: dir_seq` · rank 2 · ~3.9 M params.

## Recommended (our chat) — dilated-residual CNN → CNN + Mamba (SSM)
- **Step 1 — dilated residual CNN (Var-CNN-style):** ResNet-1D with dilated convolutions for a
  larger receptive field with fewer params; more data-efficient than DF. Ref: Bhat et al.,
  "Var-CNN," PETS 2019.
- **Step 2 — CNN + state-space (Mamba/S4):** dilated-residual CNN front-end (local structure) +
  bidirectional **SSM** for long-range temporal dependencies. Ref: the group's **Hybrid CNN-SSM**
  (WCCI 2026) — strong on long/interleaved flows. `mamba-ssm` kernel with a pure-torch S4 fallback.
- **Why better:** larger effective context, fewer params, better on long sequences (activity,
  multi-tab) than a plain VGG-CNN.
- **Register:** `@register("seq_cnn")` (recommended default).
- **As built (step 1 only):** k=7 stem, four residual stages with dilations 1/2/4/8 (two blocks
  each, widths 64/128/192/256), mean + max pooling, one hidden FC — **1.59 M params**. Input
  `packet_seq` with `feature_params: {n: 256, channels: 2}`: signed size / 1500 and
  log-scaled inter-arrival time. Trained by `training/deep.py`, no class weights (D5).

## Training & eval (shared `training/deep.py`)
- Loss: class-weighted cross-entropy; optimizer AdamW; cosine schedule; early stop on val macro-F1.
- Augmentation: random truncation/jitter of the sequence; padding mask.
- Same group-aware split + macro-F1 as Phase 3.

## Scorecard (to produce)
| Variant | macro-F1 | #params | infer ms/flow |
| --- | --- | --- | --- |
| seq_cnn_baseline (DF) | … | … | … |
| seq_cnn (dilated-res) | … | … | … |
| seq_cnn (+Mamba) | … | … | … |

## DoD
All variants registered; scorecard on ISCXVPN traffic_type + MobileApp activity; recommended ≥ DF.

## Risks
Long sequences → memory (cap N, bucket by length); Mamba CUDA kernel availability (S4 fallback);
small ISCX data → augmentation + early stop.
