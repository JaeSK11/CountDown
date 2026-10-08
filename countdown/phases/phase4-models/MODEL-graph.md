# Model: Graph (traffic-graph GNN)

**input_type:** `graph` · **feature:** `features/traffic_graph.py` (byte graphs) over
`features/byte_prep.py` (byte-retaining pcap pass) — the plan's `features/graph.py` name was not used
**Primary target:** app-ID (CSTNET 120) / relational fingerprinting
**Datasets:** CSTNET-TLS1.3 ✅ local (46,372 flows, 120 classes); MAppGraph 🔒 not obtained
**Ensemble role:** the relational member — captures structure a sequence/byte model misses.
**Input:** **per-flow** graph — nodes = burst-level/temporal bins or byte segments;
edges = temporal adjacency + data-driven dependencies (mutual information / PMI).

> ⚠️ **Per-app-session graphs are not constructible on CSTNET.** The release is 1 pcap → 1 flow
> (46,372 = 46,372, verified), so it carries no session grouping. MAppGraph-style *inter-flow*
> graphs need the MAppGraph dataset, which is still 🔒 unobtained. Scope this member to per-flow
> graphs (Okonkwo / TFE-GNN style) until that changes.

## Baseline (paper) — TFE-GNN
- **Reference:** Zhang et al., "TFE-GNN," WWW 2023 (`reference/…`; uses ISCX + WWT).
- **Architecture:** byte-level traffic graph with **dual embedding** (header + payload), GraphSAGE-
  style message passing + cross-gated feature fusion, then **temporal fusion** across packets. GCN
  backbone.
- **Register:** `@register("graph_gnn_baseline")` in `models/graph_gnn.py` — ✅ since 2026-09-17.
  `input_type = "byte_matrix"` (rank 3, `(N, 50, 40 + 150)` int16 rows from the byte_prep cache);
  no extractor of that name exists yet, so it is driven by `scripts/train_graph_gnn.py`, not
  `scripts/train.py`, until byte retention lands in the flow layer (task 4.2).

### Status — built, registered, scored on all four ISCX splits and on CSTNET
Full write-up: [`REPLICATION-TFEGNN.md`](REPLICATION-TFEGNN.md).

| Component | Where | State |
| --- | --- | --- |
| byte-retaining pcap pass | `features/byte_prep.py` | ✅ paper §4.1.2 preprocessing; ISCX loaders only |
| PMI byte graphs | `features/traffic_graph.py` | ✅ verified against the authors' `construct_graph` (`tests/test_tfe_gnn.py`, 12 tests) |
| `TFEGNNNet` | `models/tfe_gnn.py` | ✅ PyG `SAGEConv` towers + cross-gated fusion + BiLSTM; 44.3M params with the 8-class head |
| `graph_gnn_baseline` | `models/graph_gnn.py`, `tests/test_graph_gnn.py` (7 tests) | ✅ `BaseModel` wrapper: authors' schedule by default, optional `class_weight="balanced"` and early stop on val macro-F1, `LabelSpace`-ordered `predict_proba`, save/load |
| caches | `cache/tfegnn/{vpn,nonvpn,tor,nontor,cstnet}` | ✅ 31 / 109 / 51 / 44 / 120 shards (one per capture; one per app for CSTNET) = 18,233 / 290,198 / 3,003 / 88,600 / 46,284 samples |
| training | `scripts/replicate_tfegnn.py` (authors' split), `scripts/train_graph_gnn.py` (member through the Phase-3 harness, `--fold k` of a grouped 5-fold) | ✅ all five caches trained, single seed per cell (`nonvpn` 3.2 h, `nontor` 2.2 h, authors' split only) |

ISCX-Tor, macro-F1, published **0.9855** (Table 2):

| Split | addressing stripped (ours) | addressing kept (authors' byte layout) |
| --- | --- | --- |
| `sequential` (authors' split, ~capture-disjoint) | **0.387** (acc 0.623) | 0.564 (acc 0.702) |
| `random` (stratified shuffle — leaks) | 0.800 (acc 0.920) | 0.938 (acc 0.960) |

The published number is approached only with both leaks present — the same pattern as the
flow-stats and image replications. Through the registered member (seed 42, 2026-09-17) the
authors' split gives **0.526** and a grouped 5-fold split **0.442 ± 0.136** under the paper's
no-validation protocol (`runs/graph-gnn-noval/`; 0.400 ± 0.138 with a 10 % val carve-out,
`runs/graph-gnn/`) — see `REPLICATION-TFEGNN.md` for why the fold spread is that wide: one p2p
capture is 36 % of ISCX-Tor. Every cell is a single seed over
300–450 test samples with three classes at ≤ 10 samples; read the Tor numbers as ~0.4–0.5.
ISCX-VPN (six paper classes): authors' split **0.622** vs published 0.954; grouped 5-fold 0.55
over all six classes — `p2p` is one capture and `email` two, so no grouped protocol on VPN
alone can score them (`REPLICATION-TFEGNN.md`). ISCX-non-Tor (authors' split) **0.644** vs
published 0.851; ISCX-non-VPN **0.350** vs 0.924, barely above voip-majority accuracy.

**Left for this member:** seeds; a payload-masking ablation on CSTNET (see the scorecard caveat).

## Recommended — decided 2026-09-19 (D1c): GIN in the same towers
`@register("graph_gnn")` in `models/graph_gnn.py`; `TFEGNNNet(conv="gin")` swaps each GraphSAGE
layer for a `GINConv` (sum aggregation + 2-layer MLP, `train_eps`). Everything else — byte-graph
construction, dual towers, JKN concat, cross-gated fusion, BiLSTM — is unchanged, so a scorecard
difference is attributable to the message-passing rule alone. Trained by `training/deep.py` (D3):
AdamW 5e-3, one 256-sample batch in place of the authors' 5 × 102 accumulation, no class weights
(D5), no validation fold where none is honest (ISCX-Tor). `scripts/train_graph_gnn.py --model
graph_gnn`. A CSTNET byte cache now exists (`cache/tfegnn/cstnet`, 46,284 samples, 120 apps, grouped
by `capture_day`), so both variants are scored on the member's real target. Original plan text:

### Original plan — GraphSAGE / GIN
- **Model:** a cleaner **GraphSAGE** (inductive, scales) or **GIN** (max expressive power, WL-test
  discriminative) backbone on the same graph construction.
- **Why better:** simpler, inductive (generalizes to unseen graphs), strong on graph classification;
  fewer moving parts than TFE-GNN's dual-embed+fusion while competitive.
- **Register:** `@register("graph_gnn")` (recommended default).
- **Lib:** `torch-geometric` (preferred) or `dgl`.

## Training & eval (shared `training/deep.py`)
- Graph-classification head (global mean/attention pool → MLP → softmax); class-weighted CE; AdamW.
- **Use ET-BERT's provided `{train,valid,test}_dataset.tsv` split** for comparability with
  published CSTNET numbers. Group-aware splitting is degenerate here (one flow per pcap), so
  there is no leakage to guard against — but also no freedom to re-split without breaking
  comparability.
- macro-F1 headline; **also report macro-F1 excluding the sub-50-sample tail** — `chia.net` has
  16 flows total (≈13/1/2) and alone contributes 0.83% of macro-F1 with huge variance.
- Node/edge features from `features/graph.py`; cache built graphs (graph construction is the
  bottleneck).

## Scorecard (2026-09-20)
`scripts/train_graph_gnn.py`, grouped 5-fold (CSTNET by `capture_day`, ISCX by capture), no
validation fold, no class weights (D5), seed 42, 100 epochs. Mean ± sd over the five folds.
Runs: `runs/graph-cstnet/{graph_gnn_baseline,graph_gnn}/`, `runs/graph-gnn-noval/`,
`runs/graph-gin-tor/`.

| Variant | CSTNET app (120) | ISCX-Tor traffic (8) | ISCX-Tor, authors' split | #params (120-class head) | train / fold |
| --- | --- | --- | --- | --- | --- |
| `graph_gnn_baseline` (TFE-GNN, SAGE) | 0.821 ± 0.021 (acc 0.851) | 0.442 ± 0.136 | 0.526 | 44.40M | 17.4 min |
| **`graph_gnn`** (GIN) | **0.865 ± 0.008** (acc 0.891) | **0.547 ± 0.167** | **0.584** | 44.42M | 18.4 min |
| *floor:* `flow_gbdt` on flow_stats | 0.811 ± 0.002 (fold 0, 3 seeds) | — | — | — | — |

Per fold, CSTNET: SAGE 0.851 / 0.794 / 0.814 / 0.829 / 0.814, GIN 0.866 / 0.863 / 0.859 / 0.861 /
0.878 — **GIN wins on every fold**, by +0.044 on average, with a third of the spread. On Tor it wins
4 of 5 folds (+0.105 mean) but the Tor folds stay as noisy as described above.

**GIN is the first Phase-4 member to beat the CSTNET tabular floor**: 0.866 vs 0.811 on fold 0,
the fold the GBDT number is from (test sets of 9,338 graph samples vs 9,384 flows — the byte cache
holds 88 fewer samples than there are flows, so the sets are near-identical, not identical). TFE-GNN itself
clears it on fold 0 (0.851) but not on the 5-fold mean.

**Caveat — payload bytes and the label.** CSTNET's label is the server domain, and the payload
tower sees the first 150 bytes of each packet, ClientHello included. A string search found the
app's domain stem in **3.3 %** of samples (196 of 5,966 checked, 50 per app), so SNI is not what
carries the score, but it is not zero either. A rerun with the ClientHello payload masked is the
clean test; until then read the CSTNET numbers as an upper bound by a small margin.

Not met from the DoD: "lower complexity". Swapping the conv rule leaves the towers, fusion and
BiLSTM in place, so GIN is the same 44M parameters. The win
is accuracy and fold stability, not size. Inference latency was not measured.

## DoD
✅ Both registered; scorecard on CSTNET app-ID; graph cache working; recommended beats TFE-GNN
(+0.044 CSTNET, every fold). ✗ Not at lower complexity — same 44M parameters (see scorecard).

## Risks
Graph construction cost at 46k CSTNET flows → cache aggressively; MI/PMI edge estimation params;
`torch-geometric`/`dgl` install weight → lazy-import + capability flag so it can't break the registry.
