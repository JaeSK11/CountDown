# Bytes Member — Model-Selection Brief (open discussion)

> Purpose: context for choosing a classifier for the **raw-byte** representation. Covers the
> associated papers and the models *they* used, plus the dimensions to weigh. **No selection is made
> here — the choice is intentionally left open for discussion.**

## 1. The representation
The first `L` bytes of the first `k` packets of a flow, taken as raw byte values (or tokenized —
hex/bigram) → a byte matrix or token sequence. Uses header + (encrypted) payload bytes directly, so
the model learns byte-level structure rather than hand-crafted stats. **Endpoint bytes (IP/port/MAC)
must be masked** or the model can cheat on identifiers instead of content.

## 2. Associated work and the models it used

| Source | What it is | Model(s) used |
| --- | --- | --- |
| **Wang et al., 2017 (USTC-TFC)** — "End-to-end encrypted traffic classification with 1D-CNN" | Early raw-bytes deep model | **1D-CNN** on the first ~784 bytes (also byte→image 2D-CNN) |
| **Lotfollahi et al., 2020 — Deep Packet** | Per-packet bytes | **1D-CNN** and **stacked autoencoder (SAE)** |
| **He et al., 2020 — PERT** | Payload transformer | **Transformer** encoding of payload |
| **Lin et al., WWW 2022 — ET-BERT** (origin of CSTNET-TLS1.3, in `reference/`) | Pretrained byte model | **BERT-style transformer** on datagram tokens (pretrain → fine-tune) |
| **Zhao et al., AAAI 2023 — YaTC** | Recent, lighter | **Masked-autoencoder (MAE) transformer** on a multi-level byte matrix |

**Takeaway:** the representation is raw bytes; the modeling has moved **CNN/autoencoder → transformer
→ pretrained/self-supervised transformer**, trading more compute for higher fine-grained accuracy.

## 3. Why those models were chosen (their reasoning)
- **CNN/AE (Wang, Deep Packet):** learn byte patterns end-to-end without feature engineering; cheap.
- **Transformers (PERT, ET-BERT):** model long-range byte dependencies; **pretraining** lets a big
  model transfer to many classes with limited per-class labels — motivated by fine-grained app-ID.
- **MAE transformer (YaTC):** self-supervised reconstruction pretraining at lower cost than ET-BERT.

## 4. Dimensions to weigh when selecting (discussion checklist)
- **Number of classes** — raw bytes shine on many fine-grained classes (e.g. 120 apps).
- **Pretraining budget** — transformers benefit from (costly) self-supervised pretraining.
- **Compute/latency** — byte transformers are the heaviest members.
- **Input hygiene** — masking IP/port/MAC so the model learns content, not endpoints.
- **Tokenization** — raw byte ints vs hex/bigram tokens vs byte-image; consistency across inputs.
- **Data volume** — deep byte models are data-hungry; small datasets favor simpler nets.
- **Calibration** (feeds the downstream combiner).

## 5. Candidate model families for raw bytes (neutral landscape)
Listed as the option space to discuss — **not ranked, no recommendation**:

| Family | Rough character | Notes |
| --- | --- | --- |
| 1D-CNN on bytes | conv over byte stream | Wang / Deep Packet; cheap, strong baseline |
| 2D-CNN on byte-image | bytes reshaped to image | USTC-TFC MFR; reuses vision backbones |
| Stacked autoencoder | unsupervised features + head | Deep Packet; older |
| Pretrained transformer (BERT-style) | self-attention + pretrain | ET-BERT / PERT; high accuracy, heavy |
| MAE transformer | masked-reconstruction pretrain | YaTC; lighter pretraining |
| Byte-level RNN | recurrent over bytes | less common; slower |

## 6. Open question for the discussion
Given fine-grained targets (app-ID), a real pretraining/compute budget question, and downstream
combination — **how much model capacity and pretraining is justified here, and does a light CNN or a
pretrained transformer better fit the data/latency budget?** This brief deliberately stops short of
answering.
