# Sequence Member — Model-Selection Brief (open discussion)

> Purpose: context for choosing a classifier for the **packet-sequence** representation. Covers the
> associated papers and the models *they* used, plus the dimensions to weigh. **No selection is made
> here — the choice is intentionally left open for discussion.**

## 1. The representation
Each flow → an ordered sequence of the first `N` packets, encoded as **signed sizes**
(`direction · packet_size`) and/or inter-arrival times. Sometimes direction-only (±1). Payload-
independent; captures temporal/burst structure. `N` ranges from ~32 (coarse traffic types) to
hundreds/thousands (fine-grained or website fingerprinting).

## 2. Associated work and the models it used

| Source | What it is | Model(s) used |
| --- | --- | --- |
| **Rimmer et al., NDSS 2018 — AWF** ("Automated Website Fingerprinting through Deep Learning") | Early deep sequence models on direction traces | **SDAE, CNN, and LSTM** compared |
| **Sirinam et al., CCS 2018 — DF** (Deep Fingerprinting) | The canonical sequence baseline in our set | **1D-CNN**, VGG-style (stacked conv blocks) on direction sequences |
| **Bhat et al., PETS 2019 — Var-CNN** | Data-efficient variant | **ResNet-1D with dilated convolutions** + timing/cumulative stats |
| **Rahman et al., PETS 2020 — Tik-Tok** | Adds timing | DF-style **1D-CNN** on timing+direction |
| **Okonkwo et al., WCCI 2026 — Hybrid CNN-SSM** (in `reference/`) | Recent long-sequence approach | **CNN + state-space model (Mamba/SSM)** |

**Takeaway:** the representation is a packet sequence; the modeling has moved **RNN → plain 1D-CNN →
dilated/residual 1D-CNN → CNN+state-space**, chasing longer effective context with fewer parameters.

## 3. Why those models were chosen (their reasoning)
- **CNNs over RNNs (DF vs AWF):** faster, more stable to train, strong local/burst pattern capture.
- **Dilations/residuals (Var-CNN):** larger receptive field + data efficiency on limited traces.
- **State-space (Hybrid CNN-SSM):** long-range dependencies in long/interleaved flows that a fixed
  CNN receptive field misses.

## 4. Dimensions to weigh when selecting (discussion checklist)
- **Sequence length `N`** the target needs (short flow-type vs long activity/WF).
- **Local vs long-range structure** — bursts (CNN) vs long dependencies (RNN/SSM/Transformer).
- **Training-data size** — deep sequence models are data-hungry; ISCX is modest.
- **Compute/latency** — real-time path vs offline.
- **Parameter efficiency** and overfitting risk on small data.
- **Calibration** of output probabilities (feeds the downstream combiner).
- **Variable-length handling** (padding/masking/bucketing).

## 5. Candidate model families for packet sequences (neutral landscape)
Listed as the option space to discuss — **not ranked, no recommendation**:

| Family | Rough character | Notes |
| --- | --- | --- |
| RNN (LSTM / GRU / BiLSTM) | recurrent, sequential | AWF-era; long-range but slower, harder to train |
| 1D-CNN (VGG-style) | local conv | DF baseline; fast, strong on bursts, fixed receptive field |
| Dilated / residual 1D-CNN (TCN, Var-CNN) | wide receptive field | data-efficient, larger context |
| Transformer encoder | self-attention | flexible long-range; data-hungry, quadratic in `N` |
| State-space models (S4 / Mamba) | linear long-range | CNN+SSM hybrids; strong on long/interleaved flows |
| Classical on sequence-derived stats | HMM / features + tree | older; interpretable, weaker |

## 6. Open question for the discussion
Given the target's sequence length, modest labeled data, and that this member's probabilities feed a
downstream combiner — **which family (and `N`) balances long-range structure against data size and
latency?** This brief deliberately stops short of answering.
