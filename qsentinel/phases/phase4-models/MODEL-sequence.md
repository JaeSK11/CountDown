# Model: Sequence (packet size / direction sequence)

**input_type:** `packet_seq` · **feature:** `features/packet_seq.py` (Phase 0)
**Primary targets:** traffic_type (ISCXVPN/Tor), activity (MobileApp), app (CSTNET)
**Datasets:** ISCXVPN, ISCXTor, MobileAppActivity, CSTNET
**Ensemble role:** the universal deep voter — cheap, generalizes across targets.
**Input:** first `N` packets as **signed sizes** (`direction·min(size,1500)`) + IATs; `N` config
(32 for flow-type, up to 1000 for fine-grained/long).

## Baseline (paper) — DF 1D-CNN
- **Reference:** Sirinam et al., "Deep Fingerprinting," CCS 2018 (`reference/…` WF family).
- **Architecture:** VGG-like 1D-CNN — 4 conv blocks, filters [32, 64, 128, 256]; each block =
  2× (Conv1D k=8 → BatchNorm → activation) → MaxPool(pool 8, stride 4) → Dropout. Activations: ELU
  in block 1, ReLU after. Head: FC[512,512] with dropout 0.7/0.5 → softmax.
- **Adaptation:** DF uses a length-5000 **direction** sequence; for ETC use the **signed-size**
  sequence (length N from config).
- **Register:** `@register("seq_cnn_baseline")`.

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
