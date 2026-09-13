# Image Member — Model-Selection Brief (open discussion)

> Purpose: context for choosing a classifier for the **flow-as-image** representation. Covers the
> associated papers and the models *they* used, plus the dimensions to weigh. **No selection is made
> here — the choice is intentionally left open for discussion.**

## 1. The representation
Each flow → a 2D image summarizing packet **size vs time**, then a vision model classifies it. Two
common constructions (a design axis of the discussion):
- **Scatter plot** — points at (arrival time, packet size), often over fixed windows.
- **Histogram (FlowPic)** — a 2D density of packet size × normalized arrival time.
Payload-independent; turns a flow into a picture so image backbones apply. Blank-window images (no
packets in a window) are a known artifact to handle.

## 2. Associated work and the models it used

| Source | What it is | Model(s) used |
| --- | --- | --- |
| **Shapira & Shavitt, 2019/2021 — FlowPic** | size×time histogram image | **CNN** on FlowPic images |
| **Okonkwo et al., AISC 2022 — CNN classifier** (in `reference/`) | scatter image, https/VPN/Tor | **CNN** on size-vs-time scatter images (15/30/60 s windows) |
| **Sensors 2022 — Deep Learning + Unknown Data Detection** (origin of MobileAppActivity, in `reference/`) | windowed images, open-set | **DNN/CNN** on windowed traffic images |
| **Wang et al., 2017 (USTC-TFC, related)** | bytes→image | **2D-CNN** on byte-image (a different input, same backbone idea) |

**Takeaway:** the representation is a flow image; models have been **plain CNNs**, with the main
variation in the **image construction** (scatter vs histogram) and **window size**.

## 3. Why those models were chosen (their reasoning)
- **CNN on flow-image (FlowPic, Okonkwo):** reuse mature vision architectures; size×time captures
  streaming/interactive differences visually.
- **Windowing (Okonkwo, Sensors):** multiple window sizes both augment data and address arrival-time
  disparity.
- **Open-set head (Sensors):** flag traffic the model wasn't trained on.

## 4. Dimensions to weigh when selecting (discussion checklist)
- **Image construction** — scatter vs histogram (density); resolution; window size(s).
- **Backbone depth** — plain CNN vs residual/attention; data available to support it.
- **Data volume** — deeper/attention backbones need more data (ISCX is modest, CSTNET larger).
- **Augmentation semantics** — image aug must not distort traffic meaning.
- **Blank/sparse windows** — filter or flag.
- **Compute/latency** for the real-time path.
- **Calibration** (feeds the downstream combiner).

## 5. Candidate model families for flow-images (neutral landscape)
Listed as the option space to discuss — **not ranked, no recommendation**:

| Family | Rough character | Notes |
| --- | --- | --- |
| Plain CNN | few conv/pool layers | FlowPic / Okonkwo / Sensors; light, easy |
| ResNet | residual, deeper | stronger, transfer-learnable |
| EfficientNet | compound-scaled CNN | accuracy/param efficiency |
| Vision Transformer (ViT) | patch self-attention | needs large data; strong at scale |
| (image-construction choices) | scatter vs histogram, window size | a first-class decision, orthogonal to backbone |

## 6. Open question for the discussion
Given a modest-to-large data range across targets, the scatter-vs-histogram construction choice, and
downstream combination — **which image construction + backbone depth fits the data budget without
overfitting?** This brief deliberately stopped short of answering. Section 7 is the evidence that
has since been gathered; the choice is still open, but it is now open over measured numbers.

## 7. Evidence since this brief was written

The Okonkwo baseline has been built and replicated across all seven of its tasks under both its own
protocol and a leak-free one — full write-up in [`REPLICATION-OKONKWO.md`](REPLICATION-OKONKWO.md),
raw results in `runs/okonkwo/replication.json`.

**Their numbers reproduce**: 95.93% average under their protocol against 93.86% published. Under a
`source_file`-grouped split the average is 82.93% (macro-F1 0.942 → 0.805).

Four things this changes about the checklist in §4:

1. **Backbone depth is not the binding constraint — captures per class is.** The gap between the
   two protocols tracks how many independent captures back each class (78.3/class → −1.1 points;
   4.9/class → +49.7 points), not task difficulty. Task 1 keeps 9 of its 10 classes at 0.89–1.00
   recall under grouping and collapses on exactly one, `facebook_audio`, from 1.00 to **0.03**.
   That is a class the model can only recognise from captures it has already seen. No backbone
   change fixes it.
2. **Image construction is confirmed as first-class, and for a concrete reason.** Okonkwo's
   pipeline extracts only `frame.len` and `frame.time_relative` — it has **no way to encode
   direction**, and for in-app activity "send image" vs "receive image" is largely a direction
   signal. The stride-2-on-every-conv architecture also aliases away isolated scatter marks in a
   way a dense histogram does not. Both point the same way: change the input before the backbone.
3. **Blank windows are a MobileApp problem, not an ISCX one.** Measured 0.0% blank on every ISCX
   task at 15/30/60 s. But MobileApp captures have a median duration of **11.1 s**, so windows
   ≥15 s there are mostly empty axis — the artifact is real exactly where the image member's
   primary target lives.
4. **Augmentation semantics: the paper's choice is unsafe.** Rotation and flipping on a
   size-vs-time plot reverses causality and maps packet size onto the time axis. Not reproduced.

**What the open question now reduces to.** Backbone depth is the cheap variable and the data
budget already answers it (ISCX supports a small CNN; MobileApp's 4 captures/class supports almost
nothing trained from scratch). The live question is whether a denser, direction-carrying
construction plus cross-corpus pretraining moves the *grouped* numbers — because that is the only
column that means anything. `flow_image_cnn_baseline` under the grouped protocol is the bar:

| Target | baseline accuracy | baseline macro-F1 |
| --- | --- | --- |
| VPN traffic type | 96.9% | 0.917 |
| non-VPN traffic type | 96.4% | 0.934 |
| Tor traffic type | 76.8% | 0.665 |
| non-VPN application | 48.8% | 0.803 |
