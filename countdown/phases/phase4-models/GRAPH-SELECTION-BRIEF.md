# Graph Member — Model-Selection Brief (open discussion)

> Purpose: context for choosing a classifier for the **traffic-graph** representation. Covers the
> associated papers and the models *they* used, plus the dimensions to weigh. **No selection is made
> here — the choice is intentionally left open for discussion.**

## 1. The representation
Each flow (or app session) → a **graph**. Nodes are burst-level/temporal bins or byte segments;
edges encode **temporal adjacency** and **data-driven dependencies** (mutual information / PMI /
k-NN). A GNN does message passing over the graph, then a graph-level readout → class. Captures
*relational structure* a sequence or byte model treats as flat. Note: two design axes exist — the
**graph construction** (nodes/edges) and the **GNN backbone**; both are up for discussion.

## 2. Associated work and the models it used

| Source | What it is | Model(s) used |
| --- | --- | --- |
| **Pham et al., ACSAC 2021 — MAppGraph** | Mobile-app traffic graphs | **Deep Graph Convolution Network (GCN)** |
| **Zhang et al., WWW 2023 — TFE-GNN** (uses ISCX + WWT) | Byte-level dual-embed graph | **GraphSAGE-style GNN** + cross-gated fusion + temporal fusion |
| **Okonkwo et al., Information Sciences 2026 — Structure-Aware GNN** (in `reference/`) | WF via graphs, MI edges | **GNN variants** (ablated) with explainability |
| **SciRep 2025 — Lightweight graph encoder** (in `reference/`) | Efficiency-focused | **Lightweight GNN** |

**Takeaway:** the representation is a per-flow graph; the backbone has ranged over **GCN and
GraphSAGE-style GNNs**, with recent work varying **edge construction** (temporal, mutual-information)
and pushing efficiency/explainability.

## 3. Why those models were chosen (their reasoning)
- **GCN (MAppGraph):** capture app-level structure across correlated flows that a per-flow model
  misses.
- **GraphSAGE + fusion (TFE-GNN):** inductive message passing over byte-level graphs; fuse
  header/payload and across packets for fine-grained classification.
- **MI/temporal edges (Structure-Aware):** encode dependencies beyond adjacency; support
  explainability.

## 4. Dimensions to weigh when selecting (discussion checklist)
- **Graph construction** — node definition (bursts/bins/bytes), edge type (temporal / MI / k-NN):
  often matters more than the backbone.
- **Inductive vs transductive** — need to generalize to unseen flows (inductive).
- **Expressiveness vs cost** — more expressive backbones cost more.
- **Graph-build cost** at scale (46k CSTNET flows) — construction is often the bottleneck.
- **Data volume** — GNNs need enough labeled graphs.
- **Library/deployment weight** (PyG/DGL) and latency.
- **Calibration** (feeds the downstream combiner).

## 5. Candidate model families for traffic graphs (neutral landscape)
Listed as the option space to discuss — **not ranked, no recommendation**:

| Family | Rough character | Notes |
| --- | --- | --- |
| GCN | spectral-style convolution | MAppGraph; simple, transductive-leaning |
| GraphSAGE | inductive neighbor sampling | TFE-GNN; scales, generalizes to new graphs |
| GAT | attention over neighbors | learns edge importance; heavier |
| GIN | max-expressive (WL-test) | strong for graph classification |
| Graph transformer | global attention on nodes | flexible; data/compute-hungry |
| (edge-construction choices) | temporal / MI-PMI / k-NN | a first-class decision, orthogonal to backbone |

## 6. Open question for the discussion
Given that graph *construction* may dominate the backbone choice, a 46k-flow build-cost concern, and
downstream combination — **which node/edge design + GNN backbone balances expressiveness, inductive
generalization, and build cost?** This brief deliberately stops short of answering.
