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

### Status — built, registered, replicated on ISCX-Tor; not on CSTNET
Full write-up: [`REPLICATION-TFEGNN.md`](REPLICATION-TFEGNN.md).

| Component | Where | State |
| --- | --- | --- |
| byte-retaining pcap pass | `features/byte_prep.py` | ✅ paper §4.1.2 preprocessing; ISCX loaders only |
| PMI byte graphs | `features/traffic_graph.py` | ✅ verified against the authors' `construct_graph` (`tests/test_tfe_gnn.py`, 12 tests) |
| `TFEGNNNet` | `models/tfe_gnn.py` | ✅ PyG `SAGEConv` towers + cross-gated fusion + BiLSTM; 44.3M params with the 8-class head |
| `graph_gnn_baseline` | `models/graph_gnn.py`, `tests/test_graph_gnn.py` (7 tests) | ✅ `BaseModel` wrapper: authors' schedule by default, optional `class_weight="balanced"` and early stop on val macro-F1, `LabelSpace`-ordered `predict_proba`, save/load |
| caches | `cache/tfegnn/{vpn,nonvpn,tor,nontor}` | ✅ 31 / 109 / 51 / 44 shards (one per capture) = 18,233 / 290,198 / 3,003 / 88,600 samples |
| training | `scripts/replicate_tfegnn.py` (authors' split), `scripts/train_graph_gnn.py` (member through the Phase-3 harness, `--fold k` of a grouped 5-fold) | ✅ `tor` and `vpn` trained, single seed per cell; `nonvpn` / `nontor` not (≈ 30 h / 9 h) |

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
alone can score them (`REPLICATION-TFEGNN.md`).

**Left for this member:** seeds. (`nonvpn` / `nontor`, the CSTNET cache and GIN were done
2026-09-19/20 — see the scorecard.)

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

## Scorecard (to produce)
| Variant | macro-F1 (CSTNET app) | #params | graph-build ms | infer ms/flow |
| --- | --- | --- | --- | --- |
| graph_gnn_baseline (TFE-GNN) | … *(ISCX-Tor grouped 5-fold: 0.44 ± 0.14, see status)* | 44.3M (8-class head) | ~0.09 per graph (build on access) | … |
| graph_gnn (SAGE/GIN) | … | … | … | … |

## DoD
Both registered; scorecard on CSTNET app-ID; graph cache working; recommended competitive with
TFE-GNN at lower complexity.

## Risks
Graph construction cost at 46k CSTNET flows → cache aggressively; MI/PMI edge estimation params;
`torch-geometric`/`dgl` install weight → lazy-import + capability flag so it can't break the registry.
