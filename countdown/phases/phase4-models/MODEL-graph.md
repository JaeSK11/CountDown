# Model: Graph (traffic-graph GNN)

**input_type:** `graph` · **feature:** `features/graph.py`
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
- **Register:** `@register("graph_gnn_baseline")`.

## Recommended (our chat) — GraphSAGE / GIN
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
| graph_gnn_baseline (TFE-GNN) | … | … | … | … |
| graph_gnn (SAGE/GIN) | … | … | … | … |

## DoD
Both registered; scorecard on CSTNET app-ID; graph cache working; recommended competitive with
TFE-GNN at lower complexity.

## Risks
Graph construction cost at 46k CSTNET flows → cache aggressively; MI/PMI edge estimation params;
`torch-geometric`/`dgl` install weight → lazy-import + capability flag so it can't break the registry.
