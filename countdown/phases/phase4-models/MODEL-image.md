# Model: Image (flow-as-image CNN)

**input_type:** `image` · **feature:** `features/flow_image.py`
**Primary targets:** in-app activity (MobileApp), traffic_type (ISCXVPN/Tor)
**Datasets:** MobileAppActivity-Sensors2022, ISCXVPN, ISCXTor
**Ensemble role:** a complementary view — encodes size×time distribution as a 2D image.
**Input:** per-flow image; two constructions (config):
- **Scatter** (Okonkwo): x = `frame.time_relative` (windowed 15/30/60 s), y = `frame.len` (cap 1500).
- **FlowPic** (recommended): 2D histogram of packet size × normalized arrival time → grayscale image.

## Baseline (paper) — Okonkwo scatter-image CNN  ✅ BUILT AND REPLICATED
- **Reference:** Okonkwo et al., "A CNN Based Encrypted Network Traffic Classifier," AISC 2022
  (`reference/papers/CNN-Based-Encrypted-Network-Traffic-Classifier_Okonkwo.pdf`).
- **Architecture:** eleven layers, Figure 2 — conv(32,s2)×2 → pool → conv(64,s2)×2 → drop(0.25) →
  pool → flatten(**576**) → dense(64) → drop(0.5) → softmax. 102,570 params. The 576-wide flatten
  matches the figure's own label, which confirms stride 2 sits on every conv *as well as* both
  pools.
- **Sample unit:** one image per **capture-window**, not per flow — the paper never splits by
  5-tuple (`data/windows.py`). Each window scales its x-axis to its own length (§4.2), so pooled
  15/30/60 s sizes keep their timing scale.
- **Register:** `@register("flow_image_cnn_baseline")` · `features: flow_image` · rank 4.
- **Full write-up:** [`REPLICATION-OKONKWO.md`](REPLICATION-OKONKWO.md).

### Replication result (7 tasks, both protocols)
Under the paper's own protocol we average **95.93% against their published 93.86%** — their
results reproduce. Under a `source_file`-grouped split the average is **82.93%** (macro-F1
0.942 → 0.805). See the replication doc for the per-task table.

**The important part is not the average.** The gap tracks **captures per class**, not task
difficulty: traffic-type tasks (25.2 and 6.8 captures/class) lose 1.9 and 1.1 points and are
genuine results; application tasks (2.7–5.0 captures/class) are where the protocol does the work.
And the degradation is *per class*, not broad — task 1 keeps 9 of 10 classes at 0.89–1.00 recall
and collapses only on `facebook_audio` (1.00 → **0.03**), which happens to be 51% of the test fold.

**Consequence for this plan:** the published 93/96/91 are not the bar for the recommended variant.
`flow_image_cnn_baseline` under the grouped protocol is.

## Recommended — decided 2026-09-19 (D1a): FlowPic 3-channel, the same small CNN
`@register("flow_image_cnn")` in `models/flow_image_cnn.py`. The change against the baseline is
the **representation** (log-density FlowPic histogram + direction + byte-volume channels), where
the measured gain is, not the backbone — ResNet-18 at 100× the parameters was never verified
across seeds. The representation is an extractor setting, selected per experiment with
`feature_params: {construction: flowpic, channels: 3}` (the harness field added for this) and
recorded on the class as `recommended_feature_params`. Training goes through `training/deep.py`
(D3): AdamW 1e-3, warmup + linear decay, batch 32, 60 epochs, best-validation-epoch restore, no
class weights (D5). The original plan text follows for the record.

### Original plan — FlowPic + modern backbone
- **Representation:** **FlowPic** 2D histogram (denser, normalized) instead of a sparse scatter.
- **Model:** a small **ResNet/EfficientNet** (via `timm`), or **ViT** when data is large (CSTNET-
  scale). Ref: Shapira & Shavitt, "FlowPic," 2019/2021.
- **Why better:** histogram is denser/less noisy than a scatter; residual/attention backbones beat a
  plain CNN; transfer-learning-friendly.
- **Register:** `@register("flow_image_cnn")` (recommended default).

## Training & eval (shared `training/deep.py`)
- Class-weighted CE; AdamW; cosine schedule; standard image aug (light — avoid distorting semantics);
  early stop on val macro-F1; group-aware split.
- Window-size as augmentation (per the Sensors/Okonkwo idea) to fight arrival-time disparity.

## Scorecard

### Baseline, measured (ISCX, grouped protocol — `runs/okonkwo/replication.json`)
| Task | captures/class | accuracy | macro-F1 |
| --- | --- | --- | --- |
| VPN traffic type | 6.8 | 96.9% | 0.917 |
| non-VPN traffic type | 25.2 | 96.4% | 0.934 |
| encryption type | 78.3 | 99.9% | 0.999 |
| Tor traffic type | 6.7 | 76.8% | 0.665 |
| Tor application | 5.0 | 77.9% | 0.434 |
| non-VPN application | 4.9 | 48.8% | 0.803 |
| VPN application | 2.7 | *(2 of 6 classes in fold — not measurable)* | — |

`flow_image_cnn_baseline` is 102,570 params.

### Variants, measured (ISCX, grouped protocol — `runs/okonkwo/variant_*.json`)
Same driver and settings as the replication (windows 15/30/60 s, 60 epochs, batch 10, no class
weights); only the construction, channel count or backbone changes. **Single seed (42)**, macro-F1:

| Task (classes) | scatter 1ch (baseline) | FlowPic 1ch | FlowPic 3ch | ResNet-18, FlowPic 3ch |
| --- | --- | --- | --- | --- |
| 1 non-VPN app (10) | 0.803 | 0.798 | 0.642 | 0.741 |
| 2 non-VPN traffic (4) | 0.934 | 0.939 | 0.943 | **0.955** |
| 3 VPN app (6) | *2 of 6 classes in the test fold — not measurable* | | | |
| 4 VPN traffic (4) | 0.917 | 0.860 | **0.968** | 0.949 |
| 5 Tor app (4) | 0.434 | 0.310 | 0.346 | 0.392 |
| 6 Tor traffic (7) | 0.665 | 0.669 | 0.710 | **0.747** |
| #params (4-class head) | 102,180 | 102,180 | 102,756 | 11,170,884 |

3 channels = presence + direction + byte volume. ResNet-18 (`flow_image_resnet18`, torchvision, stem
adapted, no pretraining) costs ~80× the wall-clock of the small CNN (2,519 s vs 31 s on task 4).

**Seed sweep** (`runs/okonkwo/seeds/`, 5 seeds per cell, same settings, mean ± sd; tasks 1/4/5 on
the RTX 3090 / torch 2.5.1, tasks 2/6 on the RTX 5090 / torch 2.14 on 2026-09-17):

| Task | scatter 1ch | FlowPic 3ch | Δ |
| --- | --- | --- | --- |
| 1 non-VPN app | 0.779 ± 0.022 | 0.794 ± 0.029 | +0.015 (noise) |
| 2 non-VPN traffic | 0.899 ± 0.057 | 0.894 ± 0.061 | −0.005 (noise; both swing 0.82–0.97 with the fold) |
| 4 VPN traffic | 0.864 ± 0.028 | **0.930 ± 0.027** | **+0.066** |
| 5 Tor app | **0.442 ± 0.010** | 0.397 ± 0.038 | **−0.045** |
| 6 Tor traffic | 0.666 ± 0.011 | **0.711 ± 0.019** | **+0.045** (every FlowPic seed above every scatter seed) |

How to read these:

- In `replicate_okonkwo.py` the seed sets **both** the init and the group split (which captures land
  in test), so the sd is split + init variance. At sd 0.01–0.04, single-seed gaps under ~0.05 in the
  first table are not evidence.
- Task 1's 0.642 for FlowPic 3ch (seed 42) is far below its 5-seed mean of 0.794: a hard fold, not
  a construction effect.
- Three differences hold across seeds. FlowPic 3ch is **+0.066 on VPN traffic type** (4 of its 5
  seeds above every scatter seed), **+0.045 on Tor traffic type** (all 5 above all 5), and
  **−0.045 on Tor application** (all 5 below all 5). Both traffic-type wins are on the tasks the
  member is actually for; the loss is on a 4-class app task with 5 captures per class, where the
  extra channels look like extra capture fingerprint rather than signal. Non-VPN traffic type is
  dominated by fold variance (sd 0.06 for both constructions) and separates nothing.
- ResNet-18 leads on the two tasks with the most captures per class (2 and 6) — single seed, 100×
  the parameters, unverified across seeds.

### Baseline vs recommended — the scorecard (2026-09-20)
`flow_image_cnn` = FlowPic 3ch + the same small CNN, trained by `training/deep.py`, no class
weights, final epoch reported. Five seeds, grouped protocol, macro-F1 mean ± sd
(`runs/okonkwo/seeds/{base,fp3,rec}_*.json`):

| Task | `flow_image_cnn_baseline` (scatter) | FlowPic 3ch, baseline loop | **`flow_image_cnn`** | Δ vs baseline |
| --- | --- | --- | --- | --- |
| 2 non-VPN traffic (4) | 0.899 ± 0.057 | 0.894 ± 0.061 | **0.950 ± 0.024** | **+0.051** |
| 4 VPN traffic (4) | 0.864 ± 0.027 | 0.930 ± 0.027 | **0.938 ± 0.015** | **+0.074** |
| 6 Tor traffic (7) | 0.666 ± 0.011 | 0.711 ± 0.019 | **0.729 ± 0.014** | **+0.063** |
| 1 non-VPN app (10) | **0.779 ± 0.022** | 0.794 ± 0.029 | 0.752 ± 0.044 | −0.027 |
| 5 Tor app (4) | **0.442 ± 0.010** | 0.397 ± 0.038 | 0.419 ± 0.048 | −0.023 |
| MobileApp activity (92), 1 seed | 0.115 *(0.165 with balanced weights)* | — | **0.264** | **+0.149** |

| | `flow_image_cnn_baseline` | `flow_image_cnn` |
| --- | --- | --- |
| #params (4-class head) | 102,180 | 102,756 |
| train, VPN traffic task | ~30 s (GPU-resident loop) | ~65 s (`deep.py` DataLoader) |

**The recommended member wins on every traffic-type task — the targets `MODEL-DECISIONS.md`
assigns to the image member — by 0.05–0.07, five seeds, with tighter spread**, and more than
doubles the baseline on MobileApp activity. It loses 0.02–0.03 on the two ISCX application
tasks, inside one sd; those are the 5-captures-per-class tasks where nothing generalises. It
still trails the tabular member on MobileApp (`flow_gbdt` 0.441), so E3 keeps the GBDT floor.

One trainer lesson is baked in: the first run of this table **restored the best-validation
epoch** and VPN traffic came out 0.798 ± 0.206 — seed 1 was rolled back to *epoch 1* because a
val fold of a few captures happened to peak there (`runs/okonkwo/seeds-restored/`). Recommended
members now report the final epoch unless early stopping is requested
(`DeepConfig.restore_best`). Same family as the MobileApp early-stop and the Tor val-fold
findings: on small-capture data, do not select on validation.

### Still to produce
- **MobileApp activity** (the primary target), first baseline runs, 2026-09-17 — superseded by
  the scorecard above for the D5 (no-class-weights) numbers
  (`configs/experiments/mobileapp_activity_image_baseline.yaml`, windows 5 + 10 s → 1,829 images,
  grouped by capture, 92 classes, `runs/mobileapp-image/`):

  | run | macro-F1 | accuracy |
  | --- | --- | --- |
  | scatter CNN, 60 epochs (paper schedule) | **0.165** | 0.266 |
  | same, early stop patience 10 on val accuracy (the config as first shipped) | 0.004 | 0.026 |
  | `flow_gbdt` on flow_stats, whole captures (Phase 3) | 0.441 | 0.489 |

  Early stopping stopped the first run at epoch 18 with the model still at chance — on 92 classes
  the val accuracy is flat for ~15 epochs — so the config now runs the fixed schedule. Even so
  the image baseline is far below the tabular member on its own primary target. Few-shot (4
  captures per class, 1 in test) is the regime, exactly as the risk below says. Not yet run
  here: the FlowPic constructions (the harness has no per-experiment feature params, so
  `flow_image` construction/channels are global in `configs/features.yaml`), more seeds, and a
  pretrained backbone.
- Inference latency per flow; seeds on MobileApp; a pretrained backbone for the few-shot regime.

**ViT row dropped.** `MODEL-DECISIONS.md` assigns the image member to in-app activity and
traffic_type only, so it never runs on CSTNET — the one corpus large enough to justify a ViT. The
row was untestable as specified. Re-add it only if the member is extended to app-ID.

## DoD
Both constructions + baseline/recommended registered; scorecard on MobileApp activity + ISCX
traffic_type; recommended ≥ scatter-CNN.

## Risks
- **Captures per class is the binding constraint, not the backbone.** Measured: no architecture
  change recovers `facebook_audio` from 0.03 recall — the model never saw an independent capture of
  it. MobileApp activity has **4 captures per class**, worse than the ISCX task that lost 50 points,
  so treat it as few-shot and expect pretraining + a metric head to matter more than depth.
- **Blank-window images** when no packets arrive (Sensors disparity issue) → filter/flag.
  Measured 0.0% on ISCX at 15/30/60 s, but MobileApp captures have a median duration of **11.1 s**,
  so windows ≥15 s there are mostly empty axis. Its config uses 5/10 s for this reason.
- **Augmentation must be semantics-preserving.** The paper's rotation + flipping reverses causality
  on a size-vs-time plot; not reproduced, and not to be adopted.
- **ViT needs large data** — use ResNet on small sets, and see the scorecard note above.
