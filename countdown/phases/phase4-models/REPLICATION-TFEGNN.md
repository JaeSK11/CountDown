# Replicating TFE-GNN (Zhang et al., WWW 2023) as the graph paper baseline

> Cache: `python scripts/build_tfegnn_cache.py --split tor --workers 8`
> (add `--keep-addressing --out cache/tfegnn-refbytes` for the authors' byte layout — diagnostic only)
> Run: `python scripts/replicate_tfegnn.py --split tor [--cache cache/tfegnn-refbytes] [--split-mode random] --out runs/<dir>`
> Raw: `runs/tfegnn/tor.json` · `runs/tfegnn-refbytes/tor.json` · `runs/tfegnn-random-tfegnn/tor.json` ·
> `runs/tfegnn-random-tfegnn-refbytes/tor.json`
> Paper: arXiv:2307.16713 · authors' code: github.com/ViktorAxelsen/TFE-GNN

**Status:** ISCX-Tor and ISCX-VPN, one seed per cell. The non-VPN / non-Tor caches are built but
untrained (≈ 30 h and 9 h of GPU at the authors' epoch counts); CSTNET (the plan's primary target for this member) has no byte cache yet. Since
2026-09-17 the network is a registered `BaseModel` (`graph_gnn_baseline`) and has a grouped
5-fold number on Tor (section below). See `MODEL-graph.md` for what is left.

## What was built

| Component | Where | What it is |
| --- | --- | --- |
| Byte-retaining pcap pass | `features/byte_prep.py` | The paper's §4.1.2 preprocessing, which Phase 0 deliberately does not keep: bidirectional 5-tuple flows, Ethernet header + IPs + ports removed, payload-less packets dropped, flows > 10,000 packets dropped, ISCX-Tor flows cut into 60 s blocks, each sample padded to 50 packets × (40 header + 150 payload) bytes. Two deliberate deviations, recorded in `DEVIATIONS` (below). |
| PMI byte graphs | `features/traffic_graph.py` | One graph per packet part (header, payload): byte values as nodes, edges where the windowed PMI clears the threshold, self-loops, padding token 256 participating like any byte. Five details the paper does not state were taken from the authors' `construct_graph`. |
| `TFEGNNNet` | `models/tfe_gnn.py` | Two towers (header / payload) of 4× GraphSAGE + PReLU + BatchNorm with jumping-knowledge concat and mean pool; cross-gated fusion; 2-layer BiLSTM over the ≤ 50 packet vectors; 2-layer PReLU classifier. PyG `SAGEConv` in place of DGL. **44.3M params** with the 8-class head. Where paper and code disagree (embedding 50 vs 64, epochs 120 vs per-dataset, batch 512 vs 102 × 5 accumulation) the code is followed, because the code produced the published numbers. |
| Cache builder | `scripts/build_tfegnn_cache.py` | One `.npz` shard per capture under `cache/tfegnn/<split>/`, so a crash costs one file. `--keep-addressing` reproduces the reference byte layout. |
| Replication driver | `scripts/replicate_tfegnn.py` | Builds graphs on access (~0.09 ms each, worker pool), trains with the authors' hyperparameters, reports AC / PR / RC / macro-F1 next to their Table 2. `--split-mode sequential|random`. |
| Tests | `tests/test_tfe_gnn.py` | 12 tests. The graph builder is checked against a verbatim port of the authors' `construct_graph`, because a wrong edge rule trains fine and reports the wrong number. |

**Caches built** (shards = captures): `vpn` 31 / 18,233 samples / 6 classes · `nonvpn` 109 /
290,198 / 5 · `tor` 51 / 3,003 / 8 · `nontor` 44 / 88,600 / 8. Only `tor` has been trained.

## Setup, held to the authors' code

ISCX-Tor: 51 captures → 3,003 samples over the eight `traffic_type` classes, heavily skewed —
p2p 1,260 · browsing 454 · voip 411 · audio_streaming 376 · video_streaming 255 · chat 97 ·
file_transfer 94 · email 56.

Training: batch 102 × 5-step accumulation (~512 effective), **100 epochs** (their per-dataset
override for Tor; the paper says 120 and their base config 20), Adam, lr 1e-2 → 1e-4 with 10%
warmup, dropout 0.2, no validation set, final model reported. Seed 32, one run per cell. Roughly
25 min per run on the RTX 3090 (from the run-file timestamps).

Split: **`sequential`** reproduces their `construct_dataset_from_bytes_ISCX` — within each class,
the *first* 10% of samples (in capture order) are the test set. Because samples accumulate capture
by capture, that is largely capture-disjoint, i.e. close to our grouped protocol. **`random`** is a
stratified 9:1 shuffle, offered only to measure the leak. n_test = 305 (sequential) / 301 (random).

## The published number does not replicate — and what it takes to reach it

Published, ISCX-Tor (Table 2): AC 0.9886 · PR 0.9792 · RC 0.9939 · **F1 0.9855**.

Macro-F1 (accuracy) for the four combinations of *split* × *byte layout*:

| Split | addressing stripped (ours, per §4.1.2) | addressing kept (`--keep-addressing`, authors' code) |
| --- | --- | --- |
| `sequential` (authors' split) | **0.387** (0.623) | 0.564 (0.702) |
| `random` (stratified shuffle) | 0.800 (0.920) | 0.938 (0.960) |

Full metrics: sequential/stripped PR 0.390 RC 0.391 · sequential/kept PR 0.589 RC 0.702 ·
random/stripped PR 0.835 RC 0.776 · random/kept PR 0.963 RC 0.919.

Three readings:

1. **The faithful reconstruction scores 0.387.** Their own split, the bytes the paper says it uses,
   their hyperparameters, their epoch count. Accuracy is 0.623 because p2p is 42% of the samples.
2. **Each leak is worth a lot, and they compound.** Leaving IPs and ports in the bytes adds +0.18
   under the authors' split; a random split adds +0.41 over stripped bytes; together they reach
   0.938 — within 0.05 of the published 0.9855. Neither alone gets near it.
3. **Which of these the authors' pipeline actually had is not something this table settles.** Their
   `remove()` demonstrably leaves addresses in (see deviations), so the "kept" column is the layout
   their released code produces. Their split code is sequential. What we can say is that in our
   reconstruction the published value is approached only when both leaks are present.

The mechanism is the one `REPLICATION-DRAPERGIL.md` and `REPLICATION-OKONKWO.md` found: ISCX has
too few independent captures per class, and anything that lets the model recognise a capture —
its addresses, or its own near-duplicate 60 s blocks on the other side of a random split — is
scored as classification.

**Caveats on the 0.387 itself.** One seed; 305 test samples; email, chat and file_transfer have
≤ 10 test samples each, so macro-F1 swings on a handful of predictions. Treat it as the baseline's
order of magnitude on Tor, not a number to compare at the third decimal. Seeds are in step B.5 of
`../PHASE-4-HANDOFF.md`.

## The registered member, through the Phase-3 harness (2026-09-17)

`models/graph_gnn.py` now wraps the network as `graph_gnn_baseline` (same hyperparameters, seed
42, RTX 5090 / torch 2.14), and `scripts/train_graph_gnn.py` scores it with `eval/metrics.py`
over the 8-way `traffic_type` space. Runs: `runs/graph-gnn-noval/` (paper protocol: no
validation fold, all remaining captures train) and `runs/graph-gnn/` (a 10 % validation carve-out
— see point 1 for why that column is not the one to quote). Macro-F1 over the classes present in
the test fold (over all 8, absent classes as zero, in brackets where different):

| Protocol | test | captures | macro-F1, no val fold | macro-F1, with val carve-out | absent from test |
| --- | --- | --- | --- | --- | --- |
| `sequential` (authors' split) | 305 | 13 | **0.526** (acc 0.679) | — | — |
| grouped fold 0 of 5 | 1,393 | 11 | 0.297 [0.223] | 0.176 [0.132] | email, file_transfer |
| grouped fold 1 | 435 | 13 | 0.392 | 0.476 | — |
| grouped fold 2 | 415 | 11 | 0.437 | 0.463 | — |
| grouped fold 3 | 385 | 8 | 0.420 | 0.365 | — |
| grouped fold 4 | 375 | 8 | 0.665 [0.582] | 0.522 [0.457] | file_transfer |
| **grouped, 5-fold mean** | | | **0.442 ± 0.136** [0.411] | 0.400 ± 0.138 [0.378] | |

Three things to take from this, in order of importance:

1. **ISCX-Tor is dominated by one capture.** A single p2p capture holds 1,094 of the 3,003
   samples (36 %). Whichever side of a grouped split it lands on decides the fold: fold 0 has it
   in test (hence 1,393 test samples, accuracy 0.12), and in folds 1–4 the 10 % validation
   carve-out picked it up — a "val fold" that is 1,094 p2p samples plus ~20 others, which is why
   those runs' validation curves sit near zero while their test folds score 0.36–0.52, and why
   those four models trained without the largest capture. The paper protocol has no validation
   fold, so the "no val fold" column is the grouped number: **0.44 ± 0.14**. Adding the ~1,000
   samples back moved single folds by −0.08 to +0.14 in both directions, which is a measure of
   how much one seed on this dataset is worth. Any early-stopping design for this member (D3)
   has to reckon with this: on Tor there is no honest validation fold to stop on.
2. **The authors'-split number moved from 0.387 to 0.526** between the replication script (seed
   32, RTX 3090) and the member (seed 42, RTX 5090), with identical hyperparameters, epochs and
   split. That spread — on 305 test samples with three classes at ≤ 10 samples — is the variance
   the caveat above warned about; neither value is the TFE-GNN number on Tor to three decimals,
   and both are far from 0.9855. `email` scores 0.00 in every grouped fold (56 samples, and its
   captures never straddle train and test usefully).
3. **The grouped mean (~0.44) and the authors' split (~0.39–0.53) agree in magnitude**, which is
   consistent with the authors' sequential split being nearly capture-disjoint already. The leak
   that mattered in the 2×2 table above was the bytes; the split leak only bites under a random shuffle.

Per-class F1 across the no-val folds 1–4: audio_streaming 0.70–0.91, voip 0.71–0.92, p2p
0.45–0.91, video_streaming 0.45–0.60, browsing 0.33–0.71, chat 0.00–0.32, file_transfer
0.00–0.18, email 0.00 except one fold at 0.32. About 13.5 min per 100-epoch fold on the 5090
(16.5 with the per-epoch validation pass).

### ISCX-VPN through the member (2026-09-17)

Same script on the `vpn` cache (31 captures, 18,233 samples, the paper's six classes with audio
and video merged into `streaming`; authors' 20 epochs; ~3.3 min per run). Published F1 0.9536.
Runs: `runs/graph-gnn-vpn/`.

| Protocol | test | captures | macro-F1 (present classes) | [all 6] | accuracy | absent from test |
| --- | --- | --- | --- | --- | --- | --- |
| `sequential` (authors' split) | 1,827 | 9 | **0.622** | 0.622 | 0.791 | — |
| grouped fold 0 | 5,130 | 10 | 0.491 | 0.327 | 0.969 | email, p2p |
| grouped fold 1 | 3,849 | 4 | 0.893 | 0.595 | 0.969 | email, p2p |
| grouped fold 2 | 2,765 | 5 | 0.831 | 0.692 | 0.850 | p2p |
| grouped fold 3 | 3,349 | 7 | 0.541 | 0.541 | 0.732 | — |
| grouped fold 4 | 3,140 | 5 | 0.897 | 0.598 | 0.928 | email, p2p |
| **grouped, 5-fold mean** | | | 0.731 ± 0.198 | **0.551** | | |

Two readings. First, the authors' split with stripped addressing lands at 0.62 against 0.95,
the same shape as Tor (0.39–0.53 against 0.99); the addressing-kept diagnostic was not built for
VPN, and the Tor 2×2 already shows what it would say. Second, **ISCX-VPN alone cannot score
`p2p` under any grouped protocol**: the class is one capture, so it is either wholly in train
(absent from test) or wholly in test with nothing to learn from (F1 0.00 in fold 3). `email` is
two captures and has the same problem in three folds. This is the reason `PHASE-3.md` pools
ISCX-VPN with ISCX-Tor for the traffic-type head, and it means the VPN column of Table 2 is not
a target the honest protocol can even express. Per-class where it can be scored: voip
0.76–0.99, chat 0.82–0.96, streaming 0.48–0.92, file_transfer 0.16–0.80.

## Fidelity gaps and deviations, stated

1. **Addressing is really removed** (`byte_prep.DEVIATIONS[0]`). The authors' `remove()` slices
   `p[:12]` and `p[20:][4:]` from an array that still carries the 14-byte Ethernet header, so at that
   offset it removes MAC and IP-header bytes and leaves both addresses and both ports. §4.1.2 says
   they are removed; we do what the paper says. The `refbytes` column measures the difference.
2. **Header and payload streams stay aligned** (`DEVIATIONS[1]`). Their filter drops payload-less
   packets from the payload stream only, so packet *i*'s header can be paired with packet *j*'s
   payload. We drop from both.
3. **Backbone library.** DGL `SAGEConv` → PyG `SAGEConv`; same aggregation. The LSTM readout follows
   their code (`cat(h_n[-1], h_n[-2])`), not the paper's ambiguous prose; `readout="mean"` exists
   to score the alternative.
4. **Epochs.** 100, from their per-dataset override, not the paper's 120 — 20 (their base config)
   leaves the loss still descending.
5. **No validation set and no early stopping**, as in their code. The final epoch is reported.
6. **Sample counts.** The paper does not enumerate its Tor sample count, so ours (3,003 blocks
   from 51 captures) cannot be checked against theirs.

## What this establishes for model selection

- **0.387 macro-F1 under a near-grouped split is the TFE-GNN bar on ISCX-Tor**, and 0.564 is what
  the authors' actual bytes buy on that split. The published 0.9855 is not a target for
  `graph_gnn`, any more than Okonkwo's 93% or Draper-Gil's 0.898 were for the other members.
- **`audit_endpoint_leakage()`-style checks are mandatory for any byte-derived input.** This is the
  second member (after ET-BERT's flow corpus) where addressing bytes alone moved the number by more
  than the difference between competing architectures.
- **The graph member's primary target is still unmeasured.** CSTNET is 1 pcap → 1 flow, so per-app
  session graphs are impossible there (`MODEL-graph.md`), and `byte_prep.py` only knows the ISCX
  loaders. Building a CSTNET byte cache is the next concrete step, followed by registering the
  network as `graph_gnn_baseline` so it runs through `scripts/train.py` and the Phase-3 harness.
- **The comparison that matters for `graph_gnn` (GraphSAGE / GIN)** is against the stripped,
  sequential column, on the same cached samples.
