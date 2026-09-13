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

## Recommended (our chat) — FlowPic + modern backbone
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

### Still to produce
| Variant | macro-F1 (MobileApp activity) | macro-F1 (ISCX traffic_type) | #params | infer ms/flow |
| --- | --- | --- | --- | --- |
| flow_image_cnn_baseline (scatter) | … | 0.917 / 0.934 ✅ | 102,570 | … |
| flow_image_cnn (FlowPic, 1ch) | … | … | … | … |
| flow_image_cnn (FlowPic, 3ch +direction) | … | … | … | … |
| flow_image_cnn (FlowPic ResNet-18) | … | … | … | … |

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
