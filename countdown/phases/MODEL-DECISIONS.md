# Model Decisions — Originals vs Our Choices

Cross-cutting summary of the per-model plans (`phase4-models/`) and the ensemble design. For each
representation we record the **paper's original model**, **our decision**, whether it's an
**upgrade / keep-same / simplify**, and why. Then the deliberate **"same model reused everywhere"**
choices that hold the system together.

## Per-member: original → decision

| Stage / member | Original model (paper) | Our decision | Move | Why |
| --- | --- | --- | --- | --- |
| **Crypto / PQC verdict** (Stage 2–3) | Deterministic handshake inspection (PostQuantumTLS / Mankowski) | Deterministic **KEM-codepoint lookup** (no ML) | **Keep same** | Quantum-resistance is *readable* from the TLS/QUIC handshake; ML would add error, not value |
| **Flow-stats** (tabular) | **C4.5 + k-NN** (Draper-Gil, ICISSP'16 — the founding paper). RF is the *RADCOM repo*, not a paper | **LightGBM** (+ CatBoost in the bake-off) | **Upgrade** | GBDT is still the tabular frontier at this data scale. Baseline ✅ **replicated**: the classifier barely mattered (C4.5 ≈ k-NN, ~0.07 apart); the **protocol** did |
| **Sequence** (pkt size/dir) | **DF 1D-CNN** (Sirinam, CCS'18) | dilated-residual CNN → **CNN + Mamba (SSM)** | **Upgrade** | Larger receptive field, fewer params, far better on long / interleaved flows (the group's own Hybrid CNN-SSM validates this) |
| **Bytes** (raw payload) | **ET-BERT** transformer (Lin, WWW'22) | **YaTC** (MAE traffic transformer) | **Upgrade / lighter** | Comparable app-ID accuracy with much less pretraining cost + simpler input pipeline |
| **Graph** | **TFE-GNN** (GCN + dual-embed + temporal fusion, WWW'23) | **GraphSAGE / GIN** | **Simplify** | Inductive, fewer moving parts, competitive; GIN is maximally expressive (WL-test) |
| **Image** (flow→image) | **Okonkwo scatter-CNN** (AISC'22) | **FlowPic** histogram + **direction channel**, backbone second | **Upgrade** | Baseline now measured, not assumed ([REPLICATION-OKONKWO](phase4-models/REPLICATION-OKONKWO.md)): the scatter-CNN reproduces the paper (95.9% vs 93.9% published) but the binding constraint is **captures per class**, not backbone depth. Direction is the signal the paper's two-field extraction cannot encode at all. |

*(In every case we still build the original as `<name>_baseline` so the upgrade is measured, not
assumed — see each model plan's scorecard.)*

**Two baselines are now built and replicated**, and both reached the same conclusion from
different directions:

| Replication | Their protocol vs published | Leak-free protocol | What actually drives the gap |
| --- | --- | --- | --- |
| [Draper-Gil, ICISSP'16](phase4-models/REPLICATION-DRAPERGIL.md) (flow-stats) | 0.94 vs 0.898 ✅ | 0.84 | inflation grows with **class count** |
| [Okonkwo, AISC'22](phase4-models/REPLICATION-OKONKWO.md) (image) | 95.9% vs 93.9% ✅ | 82.9% | inflation grows as **captures per class** shrinks |

Same mechanism seen twice: too few independent captures per class and a random split scores
capture recall. It is the quantified form of `PHASE-3.md` §12, and it means **published numbers on
ISCX are not targets for any member** — the grouped column is.

### What building the first baseline actually changed

Flow-stats is the first member whose original was built end-to-end
(`phase4-models/REPLICATION-DRAPERGIL.md`). Three lessons that apply to **every** remaining
baseline, not just this one:

1. **Reproduce the paper's features, not just its model.** Draper-Gil used 23 time-only columns;
   our `flow_stats` has 56 including sizes. Running their *classifier* on our *features* would not
   have been a replication. Each `<name>_baseline` needs its own extractor where the paper's
   representation differs from ours.
2. **Reproduce the paper's protocol too, then re-run under ours.** Random 10-fold CV over flows
   inflated results by **+0.10 (binary) to +0.35 (7-class)** versus a `source_file`-grouped split.
   Published ETC numbers in the 0.80s–0.90s are largely this. Our lower numbers are not
   under-performance.
3. **Check the class set is even present.** ISCXVPN2016 as released has no `browsing` captures and
   no non-VPN `p2p`, so the paper's Scenario A2/B numbers are **not reproducible on the released
   data** at all — the two absent classes are precisely the two easy ones.

The corollary for the remaining baselines (`seq_cnn_baseline`, `byte_*`, `graph_gnn_baseline`,
`image_cnn_baseline`): budget for a paper-faithful feature extractor, and expect the honest number
to land below the published one.

## Where we deliberately use the SAME model for every task

The papers each used **one bespoke model for one dataset/task**. We instead standardized a **common
spine applied to every target**, then add specialists only where they earn their place:

1. **LightGBM = the universal baseline** — run on **every** target (traffic_type, tunnel_type, app,
   activity) as the always-included, fast, interpretable member. One model, all groups.
2. **LightGBM again = the stacking meta-learner** (Phase 5 combiner) — the same family that blends
   all members' calibrated probabilities. Same tool, different job.
3. **1D-CNN (sequence) = the universal deep voter** — cheap, generalizes across all targets; included
   for every group.
4. **Specialists added only where the target justifies it** — byte-transformer + GNN for
   fine-grained **app-ID** (120 classes); image/SSM for **in-app activity** (92 classes). Not every
   model runs on every group.
5. **One `BaseModel` interface + one training harness + one eval harness** for *all* models — so
   "same scaffolding for each" is enforced structurally (Phase 3), which is what lets the ensemble
   mix heterogeneous types.

## The meta-decision (one line)
Replace "a different bespoke model per paper/task" with **a common spine (LightGBM + 1D-CNN) on every
task + selective specialists (transformer/GNN/image) where complexity demands, all combined by one
calibrated LightGBM stacking meta-learner.**

## Model → target guidance (which wins where)
| Target | Best member(s) | Notes |
| --- | --- | --- |
| traffic_type (7–8) | **LightGBM** (+ 1D-CNN) | GBDT on flow-stats is SOTA-competitive; deep adds little |
| tunnel_type (VPN/Tor/plain) | **LightGBM** | coarse; don't over-engineer |
| app-ID (120, CSTNET) | **byte-transformer / GNN** | many fine-grained classes → deep wins; GBDT is the baseline to beat |
| in-app activity (92) | **sequence CNN/SSM / FlowPic** | timing/sequence structure matters — but only **4 captures per class**, so this is a few-shot target; see the risks in [MODEL-image](phase4-models/MODEL-image.md) |
