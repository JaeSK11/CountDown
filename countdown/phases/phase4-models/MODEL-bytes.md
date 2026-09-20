# Model: Bytes (raw packet-byte representation)

**input_type:** `bytes` · **feature:** `features/payload_bytes.py`
**Primary target:** app-ID (CSTNET 120) — where fine-grained byte patterns matter most
**Datasets:** CSTNET-TLS1.3 (raw pcaps **or** the ET-BERT `flow_500`/`packet_5000` preprocessed sets)
**Ensemble role:** the high-capacity member for large-class fine-grained ID.
**Input:** first `L` bytes of the first `k` packets per flow → token/byte matrix (hex/bigram tokens
per ET-BERT, or raw byte ints).

## Baseline (paper) — ET-BERT — **built**
- **Reference:** Lin et al., "ET-BERT," WWW 2022 (`reference/papers/ET-BERT_WWW2022_arXiv-2202.06335.pdf`;
  CSTNET dataset origin). Source transcribed from `reference/ET-BERT/uer/`.
- **Architecture:** BERT-style transformer over "datagram" tokens (packet bytes → bigram/hex
  tokens, grouped into BURSTs). Pretrain (masked-token + same-origin) → fine-tune per dataset.
  12 layers × 12 heads, H=768, vocab 60,005. **132.2M params** fine-tuned.
- **Register:** `@register("byte_net_baseline")` in `models/byte_net.py`. `expects_ndim = 3`
  (`(N, 2, L)` stacking token ids and segment ids).
- **Assets:** `scripts/fetch_etbert_assets.sh` → `reference/pretrained/et-bert/`
  (716 MB checkpoint + 60,005-token vocab) and `reference/ET-BERT/` (their UER-py source).
  The released checkpoint loads into our transcription with **no key remapping** — 197 tensors,
  nothing unexpected, nothing missing but the fresh classifier head.

### Three transcription details that silently break the checkpoint
None of these is visible in the paper; each was found by reading their code. They will recur for
any UER-derived model, so they are recorded here rather than only in the module docstring.

1. **UER's LayerNorm is not torch's.** It normalises by the *sample* std and divides by
   `std + eps`; `nn.LayerNorm` uses biased variance and `sqrt(var + eps)`. Small per layer,
   compounding over 12. Parameter names are `gamma`/`beta`, which is also what makes the
   checkpoint load by name.
2. **No trailing `[SEP]`.** UER's single-sentence classification path builds `[CLS] + tokens` and
   stops. Appending `[SEP]` shifts every position embedding by one relative to pre-training.
3. **The attention mask comes from the segment ids** (`seg > 0`), not a separate mask tensor, so
   padding must carry `seg = 0`.

### Named deviations from the authors' run
- **Mixed precision.** They run fp32 on V100S; we default to AMP because fp32 puts the 10-epoch
  packet run out of wall-clock reach. `--no-amp` restores fp32. (Both replication runs were made on
  the RTX 3090 this box had at the time; it now has an RTX 5090 — see `../PHASE-4-HANDOFF.md` §0.)
- **Model selection.** They train a fixed 10 epochs and report the final model; we restore the best
  validation macro-F1 epoch. Standard, and can only help the baseline.
- **Optimiser.** Faithful by default: UER's `AdamW(correct_bias=False)` is reproduced by folding the
  bias-correction factor out of the step size. `--correct-bias` switches to torch-standard AdamW.

## ⚠ The released flow-level corpus leaks endpoints

**Measured, reproducible via `data/etbert_corpus.audit_endpoint_leakage()`, which runs at the head
of every `scripts/repro_etbert.py` invocation.**

ET-BERT §4.1.2 states that the Ethernet header, the IP header and the TCP ports were removed. In
the released **flow-level** artifacts (`flow_500/*.npy` and `cstnet-tls1.3/*.tsv`) they are still
there: both MACs, both IP addresses and both ports decode cleanly from the first 38 bytes.

| Corpus | parses as Ethernet+IPv4 | server-IP-only lookup table |
| --- | --- | --- |
| `packet_5000` (packet-level) | **0.0%** | n/a — prefix lookups reach 0.045–0.067 |
| `flow_500` + `*.tsv` (flow-level) | **95.9%** | **0.8029 accuracy over all test**, 0.9043 on the 88.8% covered, 4,729 distinct IPs |

So on a 120-class problem, a majority-vote table keyed on **nothing but the destination IP** scores
0.80 against ET-BERT(flow)'s reported 0.9510 accuracy. Flow-level numbers on this corpus measure
endpoint memorisation more than byte modelling.

**Packet-level is clean** and is the corpus to prefer: headers stripped (TCP header begins at byte 2,
first TLS record at byte 18), no MAC or IP bytes, 0.57% exact-duplicate overlap between train and
test. TCP seq/ack are retained at bytes 2–9, which is the SoK's "contextual overfitting" vector but
not an app-identifying one.

This confirms and quantifies the risk that was listed at the bottom of this file as a hypothetical.
It also matches the 2025 SoK (`arXiv:2503.20093`), which reports ET-BERT dropping to 0.59 accuracy
once shortcuts are removed and to chance on encrypted payload alone.

## Replication status

Command: `python scripts/repro_etbert.py --level {packet,flow}`. Records land in
`runs/etbert-repro/<level>.json` with the full epoch history and the leakage audit inlined.

Their configuration, held fixed: batch 32, 10 epochs, warmup 0.1, dropout 0.5, seq_length 128,
lr 2e-5 (packet) / 6e-5 (flow), their corpus, their 8:1:1 split.

| Run | our macro-F1 | paper | delta | notes |
| --- | --- | --- | --- | --- |
| ET-BERT(flow), seq 128 | **0.8585** (acc 0.8736) | 0.9426 (acc 0.9510) | **−0.0841** | 20.8 min, best epoch 9 |
| ET-BERT(packet), seq 128 | **0.9076** (acc 0.9066) | 0.9741 (acc 0.9737) | **−0.0665** | 5.0 h, best epoch 10 (the last) |
| ET-BERT(packet), no pretrain | **0.6877** (acc 0.6842) | — | **−0.2199 vs pretrained** | 1.4 h on the RTX 5090, best epoch 10 (still rising) |

**The flow gap is not undertraining.** Train loss reached 0.044 and val macro-F1 plateaued from
epoch 7 (0.847 → 0.848 → 0.852 → 0.851). Leading hypothesis: **`seq_length 128` truncates the
320-token flow datagram to 127 tokens**, discarding ~60% of each sample. The flow-level sequence
length is not stated in the paper and their README documents only the packet command
(`--seq_length 128`). Testing it costs one `--seq-length 512` run (~2 h; attention is quadratic).

**The packet gap is smaller, and the curve had not flattened.** Validation macro-F1 rose every
epoch — 0.450 → 0.725 → 0.797 → 0.841 → 0.866 → 0.883 → 0.895 → 0.902 → 0.906 → 0.908 — so the
best model is the final one, unlike the flow run. But the last three epochs add 0.002–0.004 each;
training longer would not close 0.066. The leakage audit on `packet_5000` is 0.0 (nothing parses
as Ethernet/IPv4), so this is the one ET-BERT number that measures byte modelling rather than the
address book: **0.9076 macro-F1 over 58,171 test packets, 0.67 ms/sample** on the paper's own
8:1:1 split (`runs/etbert-repro/packet.json`, `packet_metrics.json` for the per-class table).

**Pre-training is worth +0.22 macro-F1 on the clean corpus** (`runs/etbert-repro-nopretrain/
packet_scratch.json`, 2026-09-17, same 10-epoch protocol, random init instead of the checkpoint):
0.688 against 0.908. The from-scratch curve (0.12 → 0.33 → 0.44 → 0.53 → 0.59 → 0.62 → 0.65 →
0.67 → 0.68 → 0.69) is still climbing ~0.01 per epoch at the end, so part of the gap is the
paper's fixed 10-epoch budget rather than capacity — but the pretrained model reached 0.69 by
epoch 2. The paper's own ablation (−37.57 %) was measured at ≤ 100 samples/class; at CSTNET's
~500/class the effect is smaller but still the largest single factor measured on this member,
which is the fact the bytes selection (`BYTES-SELECTION-BRIEF.md`, D1) has to weigh: a candidate
without pretraining starts 0.22 behind on this corpus, and one with lighter pretraining
(YaTC / NetMamba) is betting that most of the 0.22 comes cheaply.

Read the flow gap against the leakage table above: our 0.8736 accuracy is 7 points above the address
book, the paper's 0.9510 is 15 points above it. Both sit in a band where endpoint memorisation
dominates, so the flow reproduction is a pipeline check, not evidence about byte modelling.

**Label ids are not mappable to domains.** Their `dataset_generation.py` assigns the 120 ids in
`os.walk` order over the capture directories — their filesystem's ordering, not ours. Macro-F1 is
invariant to the permutation, so ids are used as class names and the ambiguity is confined to
`data/etbert_corpus.py`.

## Recommended — decided 2026-09-19 (D1b): the ET-BERT checkpoint + field-randomising fine-tune
`@register("byte_net")` in `models/byte_net.py`. The no-pretrain ablation (0.688 vs 0.908) is the
case for keeping the public checkpoint; what changes is the fine-tune. `FieldRandomiser`
implements TrafficFormer's random-initialisation field augmentation for the one such field this
corpus still carries: **TCP seq/ack, datagram bytes 2..9**, re-drawn on every access. It works
in byte space — decode the prefix (including WordPiece-split words), rewrite, re-tokenise —
because ~12 % of bi-grams are outside the vocabulary, so a token-space version reached only a
third of packets. `scripts/score_byte_net.py` scores baseline and recommended **twice**: on the
released test split and on the same packets with seq/ack randomised; the drop between the two
columns is how much of a member's score was that shortcut. The earlier discussion follows.

### Earlier discussion — candidates considered
`MODEL-bytes` originally paired the ET-BERT baseline with **YaTC** (Zhao et al., AAAI 2023) as the
recommended upgrade. That is a reasonable 2023 answer, but the efficiency axis has moved:
**NetMamba** (ICCCN 2024) matches YaTC with ~4× less GPU memory via a unidirectional SSM, and
**MambaNetBurst** (arXiv 2026, preprint) reports 0.9824 F1 on 215-class CrossPlatform-Android with
**2.5–2.7M params and no pretraining at all**. **TrafficFormer** (IEEE S&P 2025) is the direct
ET-BERT successor if we stay in the BERT lineage (+10% F1, same cost profile); its field-randomising
fine-tune augmentation is backbone-agnostic and worth borrowing regardless.

Selection deliberately left open — see `BYTES-SELECTION-BRIEF.md`. What the packet-level replication
and the `--no-pretrained` ablation measure is precisely the question that decides it.

- **Fallback baseline option:** a plain **byte-CNN** (Wang et al. USTC-TFC, first 784 bytes → 28×28
  → LeNet-style 2D-CNN) as an even-cheaper reference if transformer pretraining is too costly.

## Training & eval (shared `training/deep.py`)
- Two-stage: (optional) self-supervised pretrain → supervised fine-tune with class-weighted CE.
- AdamW + warmup; early stop on val macro-F1; atomic best-epoch checkpointing to disk.
- **Splitting:** the replication uses ET-BERT's own 8:1:1 split so only the model varies. For our
  own runs, note that grouping CSTNET by `source_file` is **vacuous** — one pcap is one flow, 46,372
  groups for 46,372 samples — so `group_split` requires `allow_vacuous=True` there. See
  `eval/splits.py`.
- Privacy hygiene: strip/mask IPs/ports from byte input so the model learns content, not endpoints.
  Now demonstrably load-bearing, not a nicety.

## Scorecard (to produce)
| Variant | macro-F1 (CSTNET app) | #params | pretrain cost | infer ms/sample |
| --- | --- | --- | --- | --- |
| byte_net_baseline (ET-BERT), flow | 0.8585 | 132.2M | reused public checkpoint | 0.77 |
| byte_net_baseline (ET-BERT), packet | **0.9076** | 132.2M | reused public checkpoint | 0.67 |
| byte_net_baseline, no pretrain, packet | **0.6877** | 132.2M | none | 0.28 (fp16 eval, 5090) |
| byte_net (recommended) | … | … | … | … |

## DoD
Both registered; scorecard on CSTNET app-ID; recommended matches/lowers cost vs ET-BERT.

## Risks
- **Endpoint leakage — CONFIRMED on the released flow corpus** (see above). Any byte member must be
  audited with `audit_endpoint_leakage()` before its number is believed.
- **Pretraining compute.** ET-BERT's own pretraining is 500k steps over 30 GB on multi-GPU; we
  reuse the public checkpoint rather than reproduce it. The `--no-pretrained` run is what tells us
  whether that cost buys anything on our data — and note their −37.57% "pre-training is everything"
  ablation was measured at ≤100 samples/class, a few-shot regime CSTNET is not in.
- **Tokenisation consistency** between raw-pcap and preprocessed inputs.
- **`features/payload_bytes.py` is not yet BURST-capable.** `Flow` retains only the reassembled
  handshake window per direction, not per-packet payloads, so the extractor yields opening bytes.
  ET-BERT's BURST construction needs per-packet payload retention in `flows/extract.py` — a
  separate change (task 4.2 proper). Until then the replication reads the released corpus directly,
  so no tokenisation difference can be mistaken for a modelling difference.
