# Phase 5 — Router & Calibrated Stacking Combiner (Pipeline Stage 4, part 3)

**Roadmap ref:** `../PLAN.md` → Phase 5. **Depends on:** Phase 4 (trained `BaseModel` members +
experts), Phase 3 (eval harness, `LabelSpace`), Phase 1 (`l7_hint` for routing).
**Goal:** combine the independently-trained models into an ensemble that **beats any single member**,
via (1) per-model **probability calibration**, (2) a **router** (mixture-of-experts by protocol),
and (3) a **stacking combiner** (LightGBM meta-learner over calibrated `predict_proba`).

**One-line exit criterion:** on held-out (group-aware) test data, the ensemble's macro-F1 **≥ the
best single member**, and an **ablation table** (single vs soft-vote vs stacking, ± calibration) is
produced.

---

## 1. Why calibrate before combining (the key correctness point)
Tree (`flow_gbdt`) and neural (`seq_cnn`, `byte_net`, …) models emit `predict_proba` on **different
scales** — NN softmax is often over-confident, trees under-confident. Averaging or stacking raw
probabilities lets a mis-calibrated member dominate. So: **calibrate each member first**, then
combine. This is the single biggest ensemble win most ETC papers skip.

**One interaction to pin down: class weights x calibration.** `flow_gbdt` defaults to
`class_weight="balanced"` (Phase-3 §5.2), so it estimates a *reweighted* posterior, not the true
prior. Post-hoc calibration fitted on a held-out split that carries the **true** class distribution
largely corrects this — weights and calibration are compatible — but only under two conditions:
(1) the calibration split must not itself be rebalanced, and (2) **every member must follow the same
convention**, or the combiner is fed members sitting on different implicit priors. Thin classes are
the failure mode: isotonic on `p2p` (10 capture groups) or CSTNET's `chia.net` (16 samples) will
overfit its own calibration set — prefer temperature/Platt below roughly 200 samples/class.
`test_calibration.py` should assert the convention holds, not only that ECE drops.

---

## 2. Scope
**In scope**
- `ensemble/calibration.py` — per-model temperature/Platt/isotonic calibration + ECE metric.
- `ensemble/router.py` — rule-based MoE: route by Stage-1 `l7_hint`/`tunnel_type` to expert(s).
- `ensemble/stacking.py` — **out-of-fold** base-prediction generation (leakage-safe).
- `ensemble/combiner.py` — LightGBM meta-learner over stacked, calibrated probas (+ soft-vote for
  ablation).
- `ensemble/ensemble.py` — the `Ensemble` object (router + calibrated members + combiner) with a
  per-flow **explainability breakdown**.
- `eval/ablation.py` — the comparison harness.

**Out of scope (later)**
- End-to-end pcap→verdict CLI, live capture, paper tables → Phase 6.
- New model *types* → Phase 4 (frozen here; Phase 5 only combines).

---

## 3. Dependencies
`scikit-learn` (calibration, isotonic), `lightgbm` (meta-learner) — both already present. No new
heavy deps.

---

## 4. Directory additions
```
countdown/ensemble/
├── calibration.py   # CalibratedModel wrapper (still a BaseModel) + ECE / reliability
├── router.py        # protocol/tunnel → expert(s)  (rule-based, from deterministic Stage-1)
├── stacking.py      # K-fold OOF base predictions (no leakage)
├── combiner.py      # LightGBM meta-learner; soft-vote fallback
└── ensemble.py      # Ensemble: build/fit/predict + per-member breakdown
eval/ablation.py     # single vs vote vs stack, ± calibration → table
tests/
├── test_calibration.py   # ECE drops after calibration
├── test_stacking_oof.py  # asserts no train row used to predict itself
└── test_ensemble.py      # ensemble >= best member on a toy set
```

---

## 5. Components

### 5.1 Calibration (`calibration.py`)
- `CalibratedModel(base, method)` wraps any `BaseModel`, **still implements `BaseModel`** (so it's a
  drop-in). Methods: temperature scaling (deep), Platt/sigmoid or isotonic (trees).
- Fit on a **dedicated calibration split** (not train, not test). Report **ECE** + reliability
  diagram before/after.
- **DoD:** ECE decreases post-calibration on val; wrapper round-trips save/load.

### 5.2 Router (`router.py`)
- Rule-based mixture-of-experts using the **deterministic Stage-1 signal**: `l7_hint`/`tunnel_type`
  → expert set. e.g. `tls→{TLSExpert, AppExpert}`, `vpn→{VPNExpert}`, `tor→{TorExpert}`,
  `wifi/opaque→{ActivityExpert, VPNExpert}`; default/fallback expert always available.
- Returns the expert(s) (hence the member models) relevant to a flow. Interpretable, no training.
- **DoD:** routes each dataset's flows to the expected expert; fallback covers unknowns.

### 5.3 Out-of-fold stacking (`stacking.py`) — leakage-safe
- **The trap:** training the meta-learner on base predictions made over the *same* rows the bases
  trained on leaks. Fix: **K-fold OOF** — for each fold, train bases on K−1 folds, predict the held
  fold; assemble OOF probas → train meta; then refit bases on full train for inference.
- Group-aware folds (by `source_file`) carried from Phase 3.
- **DoD:** `test_stacking_oof.py` asserts no row's meta-feature came from a base trained on that row.

### 5.4 Combiner (`combiner.py`)
- Input: concatenated **calibrated** `predict_proba` from routed members (fixed slot layout aligned
  to `LabelSpace`; missing member → zero-block + mask).
- Meta-learner: **LightGBM** → final class + proba. Alternative: weighted **soft-vote** (weights from
  val macro-F1) for the ablation.
- **DoD:** produces final proba in `LabelSpace`; both stacking and soft-vote selectable.

### 5.5 Ensemble object (`ensemble.py`)
- `Ensemble.build(members, router, combiner)`, `.fit(train)` (calibrate + OOF-stack), `.predict(x)`
  → `{label, proba, per_member: {name: top_class, conf}}` for **explainability**.
- Itself usable by Phase 6's pipeline as the Stage-4 classifier.

---

## 6. Work breakdown & Definition of Done
| # | Task | DoD |
| --- | --- | --- |
| 5.1 | `CalibratedModel` + ECE | ECE ↓ after calibration; still a BaseModel |
| 5.2 | `router.py` rule-based MoE | correct expert per `l7_hint`; fallback works |
| 5.3 | `stacking.py` OOF (group-aware) | leakage test passes |
| 5.4 | `combiner.py` LightGBM meta (+ soft-vote) | final proba in LabelSpace |
| 5.5 | `ensemble.py` build/fit/predict + breakdown | end-to-end on one target |
| 5.6 | `eval/ablation.py` | table: single vs vote vs stack, ±calib |

---

## 7. Acceptance test (end of Phase 5)
```
from countdown.ensemble import Ensemble
from countdown.eval import evaluate, ablation

ens = Ensemble.build(members=["flow_gbdt","seq_cnn","byte_net"],
                     router="protocol", combiner="stacking", calibrate=True)
ens.fit(train)                      # calibrate on val, OOF-stack meta
rep = evaluate(ens, test)
assert rep["macro_f1"] >= best_single_member_macro_f1     # ensemble wins

tbl = ablation(members, test)       # single / soft-vote / stacking × ±calibration
# stacking+calibration is top row; committed under runs/ablation/
```
**Then Phase 6 wraps this ensemble as the Stage-4 classifier in the end-to-end pipeline.**

---

## 8. Risks & open questions
- **Stacking leakage** (biggest) → OOF protocol + group-aware folds; unit-tested.
- **Calibration overfit** on small datasets → use isotonic only with enough val data, else
  temperature/Platt; separate calibration split.
- **Router mis-routing** on ambiguous/opaque flows → always-on fallback expert; log routing.
- **Latency** (calibrate + stack + multiple members) → allow a "fast" combiner (GBDT+CNN, soft-vote)
  for the real-time path in Phase 6.
- **Marginal gains:** if the ensemble barely beats the best single member, report honestly and keep
  the simpler model — the ablation table makes this explicit.

## 9. Not doing yet (explicit)
End-to-end pcap→verdict CLI, live capture, cross-dataset experiments & paper tables → Phase 6.
New model types → Phase 4 (frozen here).

---

## 10. Status and results (2026-10-09)

**Built:** every module in §4, plus `ensemble/views.py`, `scripts/run_ensemble.py` and a Docker
queue (`docker/README.md`). Tests: `test_calibration.py`, `test_stacking_oof.py`, `test_router.py`,
`test_ensemble.py` — including the §5.3 leakage test (no row, and no capture, is ever predicted by
a model that saw it). Suite **549/549**, in the container.

**Exit criterion: met on E2 and E3, not on E1.** The verdict uses only configurations chosen
without test labels (§10.2). Macro-F1 on the outer grouped test split, seed 42, one run each;
runs in `results/phase5/ensemble/`.

| Expert | test rows (groups) | best single | **CV-selected** ensemble | declared primary | verdict |
| --- | --- | --- | --- | --- | --- |
| E1 traffic type, `iscx_pooled` (8) | 17,918 (75 captures) | `flow_gbdt` **0.6654** | 3 members, vote, temperature: 0.6648 | 0.5856 | ✗ tie (−0.0006) |
| E2 app-ID, CSTNET (120) | 9,338 (51 days) | `graph_gnn` 0.8680 | GBDT + GIN, vote, temperature: **0.8900** | 0.8726 | ✓ **+0.022** |
| E3 activity, MobileApp (92) | 92 (92 captures) | `flow_gbdt` 0.4409 | 3 members, vote, temperature: **0.5134** | 0.3978 | ✓ +0.073, weak (below) |

### 10.1 What the protocol is

Per expert, one flow table and one outer split (`group_split`, the Phase-3 fold 0). On the train
rows, each member gets **5 grouped out-of-fold (OOF) fits** (3 on MobileApp: its thinnest class
has 3 training captures) and then a refit on all of train that scores test. Calibrators and
combiners are fit on the OOF predictions only. Each member's OOF and test predictions are cached
per fold under `runs/ensemble/<E>/preds-s42/`.

**Deviations from the plan, and why:**
- **The sample unit is the flow** — what Phase 6 classifies. Members with another unit are mapped
  to it: the image member uses windows cut on each flow's own timeline (E1: one 60 s window,
  64 px, 15 epochs at batch 256 — ~136k images; the 224-px, 15/30/60 s scorecard construction
  would be >100 GB per flow and three copies of one image for the median ISCX flow), and
  averages its window probabilities per flow. Its E1 number (0.483) is therefore *not* the
  `MODEL-image.md` number, which was capture-window.
- **E2 is GBDT + GIN, without `byte_net`.** ET-BERT's corpus carries no flow identity to join
  on; `byte_net` joins once byte extraction from our own pcaps exists (task 4.2). E2's table is
  the 46,284 CSTNET flows that have a graph sample (of 46,372).
- **E3 has 3 OOF folds**, not 5 (above).

### 10.2 Choosing the configuration without looking at test

The ablation scores ~30 variants (single / vote / `stack_logreg` / `stack_lgbm`, ± temperature,
every member subset). Taking the best by test would be selection on test, so the verdict uses:
- the **declared primary** — all members, `stack_logreg`, temperature — fixed before any run; and
- the **CV-selected** variant — highest macro-F1 when each combiner is fit on K−1 OOF folds and
  scored on the held one, entirely inside train.

**The declared primary was the wrong choice, and the CV selection caught it:** it picked the
soft vote on all three experts, and the vote beats both stackers everywhere:

| full member set, temperature | E1 | E2 | E3 |
| --- | --- | --- | --- |
| soft vote (OOF-F1 weights) | **0.6648** | **0.8900** | **0.5134** |
| `stack_logreg` (primary) | 0.5856 | 0.8726 | 0.3978 |
| `stack_lgbm` | 0.6100 | 0.5261 | 0.4199 |

The logistic stacker minimises log-loss, which trades thin classes for frequent ones and so
costs macro-F1. The 120-class LightGBM meta-learner collapses on E2 (NLL 11) with default
parameters — 240 meta-features, classes with a handful of OOF rows. **Recommendation for
Phase 6: the calibrated soft vote** is the default combiner; stacking stays in the ablation.

### 10.3 Per expert

- **E2 is the real result.** GBDT (0.832) + GIN (0.868) → 0.890 macro-F1, accuracy 0.913. The
  two see different things — flow statistics vs payload byte graphs — which is what
  `ENSEMBLE-MAP.md` predicted would pay off on a fine, data-rich target. 9,338 test flows.
- **E1 gains nothing.** Every subset ties or trails the GBDT (best pair 0.6627, all three
  0.6648). On a coarse 8-class target the sequence and per-flow image members add no
  information the 56 flow statistics lack — consistent with the Phase-4 finding that neither
  beats the tabular floor (`MODEL-sequence.md`). **Ship E1 as the GBDT alone**; the deep
  members cost latency for nothing. **Done (2026-10-09):** `Expert.deploy` lists the members a
  deployed ensemble uses — E1 deploys `flow_gbdt` alone, E2 `flow_gbdt` + `graph_gnn`, E3 all
  three; the other members stay in `members` so this ablation reruns. `Ensemble.build` uses the
  deploy list; a one-member ensemble is that member, calibrated.
- **E3's gain is weak evidence.** The test set is one capture per class (92 rows), so a class's F1
  is all or nothing and +0.073 is ~7 captures. The direction agrees with the CV score (0.4675 vs
  0.4305), but it needs seeds or a leave-one-capture-out protocol before it is a result.
  **Resolved (2026-11-09, §10.7):** across 10 seeds the vote wins every time, +0.080 mean.

### 10.4 Calibration

Temperature scaling fixes the members it should: test ECE `graph_gnn` 0.069 → **0.012**,
`flow_gbdt` 0.047 → 0.023 (E1) and 0.030 → 0.012 (E2). NLL drops or holds for every member
except the E1/E3 image member, where ECE and NLL both rise — likely because its OOF fits saw
less data than the refit, so a temperature learnt on OOF mis-states the refit (not tested).
Fitted temperatures: E1 GBDT 1.68, seq 5.77, image 1.43; E2 GBDT 1.14, GIN 2.39; E3 GBDT 0.98,
seq 1.61, image 1.04 — **on E1 and E2 every member is over-confident, the GBDT included**,
contrary to §1's "trees under-confident"; on E3 the GBDT and image member are near 1.

Two consequences: calibration barely moves macro-F1 (temperature cannot change a member's
arg-max), and **a vote of calibrated members is itself under-confident** (E2 vote ECE 0.024 raw
vs 0.061 calibrated). Phase 6 should fit one more temperature on the *combined* OOF output.

### 10.5 Router

`ensemble/router.py` is rule-based (§5.2), tested in `test_router.py`. On real Stage-1 hints:

| dataset | flows | → expected expert | other |
| --- | --- | --- | --- |
| CSTNET (re-parsed 2026-10-09) | 46,372 | **E2 99.998 %** (`tls`) | 1 flow hinted `tor` → E1 |
| `iscx_pooled` | 70,523 | E1 85.5 % | **14.5 % (`tls`) → E2** |
| MobileApp | 368 | E3 100 % (capture context; no hint) | — |

CSTNET's flow cache predated Phase 1 and carried no `l7_hint`; it was re-parsed with the
Phase-1 reassembler (18 s) after backing up the old cache (`cache/cstnet/phase0-backup/`).
`scripts/reparse_check.py` confirmed **all 46,372 flows have identical packets and identical
`flow_stats`** — only the hint and handshake fields were added — so every Phase-4/5 CSTNET score
stands (`results/phase5-data/reparse/`).

**The open design question is ISCX's 10,189 `tls` flows** (14.5 % of `iscx_pooled`). By the
rules they are "direct TLS" and go to E2, which labels a 2016 VPN-dataset flow with one of 120
CSTNET websites — the cross-distribution case `ENSEMBLE-MAP.md` calls out of scope. Measured
with `scripts/eval_routing.py` (`results/phase5-data/routing-eval/`). E2 is represented by its
calibrated GBDT member, because GIN needs per-flow byte graphs ISCX does not have; read every
E2 number below as a lower bound for the full ensemble.

| measurement | value |
| --- | --- |
| E2 confidence, median: CSTNET test / ISCX `tls` | **0.998 / 0.239** |
| AUROC of E2 confidence, CSTNET vs ISCX `tls` | **0.972** |
| E2's top labels for ISCX `tls` flows | twimg.com 1,485 · steampowered.com 901 · deepl.com 878 · … |
| E1 (GBDT) on its `tls` test flows / on the rest | macro-F1 **0.548** / 0.666 · accuracy 0.609 / 0.933 |

Confidence-fallback threshold, chosen to keep a share of CSTNET flows in E2:

| keep CSTNET in E2 | threshold | CSTNET accuracy on the kept flows | ISCX `tls` caught → E1 |
| --- | --- | --- | --- |
| 99 % | 0.18 | 0.936 | 33 % |
| 95 % | 0.42 | 0.961 | 83 % |
| **90 %** | **0.72** | **0.984** | **96 %** |
| 80 % | 0.98 | 0.998 | 99.9 % |

The options:

| option | for | against |
| --- | --- | --- |
| **A. Route by hint (today)** | Deterministic, free, explainable | Every non-CSTNET-like TLS flow gets a confident-looking website label from a closed set of 120; no way to say "not one of these" |
| **B. Confidence fallback** (E2, then E1 below a threshold) | Data says it works: AUROC 0.97; at a 90 %-keep threshold 96 % of the out-of-distribution flows fall back, and E2's accuracy on what it keeps rises 0.927 → 0.984. Needs no deployment metadata | Runs E2 on every TLS flow, E1 again on the fallbacks; threshold is calibration-dependent and must be re-set if E2 changes; confidence-based out-of-distribution detection can fail on shifts that look in-distribution; 10 % of real CSTNET flows lose their app label (they get a traffic type instead); measured with E2's GBDT only |
| **C. Capture-level context** (operator says what the tap sees, as E3 already does) | Deterministic, zero inference cost, no threshold | A real tap carries mixed traffic (`DEPLOYMENT-AND-PRIMARY-DATASET.md`: passive capture on a real network), so one label per capture is wrong for most of it; needs metadata every deployment must supply |
| **D. Route on the Phase-2 verdict** | Uses the authoritative parser instead of the advisory hint | The verdict is about the key exchange, not the traffic population, so it does not separate these flows. The one field that would — TLS version (CSTNET is all 1.3, ISCX 2016 mostly older) — is a dataset artefact: modern traffic is mostly 1.3, so on a real network it routes everything to E2 anyway |

**Recommendation: B, with C as an override.** The fallback is the only option that handles mixed
traffic without metadata, and the numbers support it. Threshold: start at the 90 %-keep point
(0.72 on the calibrated GBDT), re-fit on the *ensemble's* calibrated output once Phase 6 has it,
and record `route_reason` (`hint` / `fallback` / `context`) in every verdict so the trade-off
stays auditable. Two caveats travel with it: E1 itself is weak on these flows (0.548 macro-F1 vs
0.666 elsewhere), so falling back moves them to a better-suited model, not a good one; and the
measurement must be repeated with GIN in the loop. Not implemented yet — this is a Phase-6
pipeline decision.

### 10.6 Left for Phase 5 / carried to Phase 6

- Seeds (and, for E3, leave-one-capture-out) before quoting E2/E3 gains as stable. **Done for all
  three (§10.7)**; E2 across outer folds queued.
- A temperature on the combined output (§10.4).
- `byte_net` in E2 once flow-aligned byte input exists.
- The ISCX-TLS routing decision (§10.5): recommended confidence fallback, to implement in the
  Phase-6 pipeline.
- Latency of the ensembles (E2 = GBDT + a 44M-parameter GNN with on-the-fly graph building).

### 10.7 Every expert across seeds (2026-11-09 – 11)

`scripts/ensemble_seeds.py` → `results/ensemble_seeds/<E>.{json,md}`, from
`run_ensemble.py --expert <E> --seed N` (same protocol as seed 42, on the GPU queue).

**What a seed changes.** Always the members' initialisation. The grouped split moves only as
far as sklearn lets it: `StratifiedGroupKFold(shuffle=True)` shuffles the groups, then
stable-sorts them by the spread of their class counts, so the seed only reorders groups that
tie. Single-class captures all tie: **E3**'s split moves fully, **E1**'s partly (seeds 1–3 share
63–65 of 75 test captures with seed 42). CSTNET days hold many classes and never tie: **E2's
split is identical on every seed** (same fold digests). For a different E2 test set,
`run_ensemble.py --outer-fold k` takes the k-th of the five outer folds as test (default 0, the
split every phase used); folds are disjoint test sets.

#### E3 (10 seeds, ~6 min each)

Each class keeps one test capture, 92 test flows.

| test macro-F1 (10 seeds) | mean | sd | range |
| --- | --- | --- | --- |
| `flow_gbdt` | 0.453 | 0.035 | 0.397–0.519 |
| best single, picked **on test** | 0.455 | 0.036 | 0.397–0.519 |
| **deployed** (vote, temperature) | **0.535** | 0.036 | 0.482–0.605 |
| CV-selected | 0.539 | 0.038 | 0.482–0.605 |

- **Deployed − best single: +0.080, 95 % t-interval [+0.052, +0.108]; 10 wins, 0 losses
  (sign test p = 0.002).** Picking the single member on test favours the single side, so this
  is a lower bound on the gap. The GBDT is the best single member on 8 seeds.
- **Seed 42 was not lucky:** its 0.513 is below the 10-seed mean.
- **CV still picks the vote every time;** on 4 seeds it drops the temperature (vote/none), within
  ±0.02 of the deployed variant. The deployed configuration stays as it is.
- **Caveat:** with 4 captures per class the 10 test sets overlap, so the seeds are not
  independent samples and the interval is somewhat narrow. Leave-one-capture-out is still the
  stricter protocol; the 10/0 sign is what makes the result robust.

**Decision: E3 keeps the three-member vote.**

#### E1 (10 seeds, ~9–30 min each)

| test macro-F1 (10 seeds) | mean | sd | range |
| --- | --- | --- | --- |
| `flow_gbdt` (deployed, alone) | 0.672 | 0.006 | 0.664–0.682 |
| CV-selected | 0.674 | 0.006 | 0.665–0.683 |

The GBDT is the best single member on every seed. **CV-selected − GBDT: +0.002, 95 % interval
[−0.002, +0.005]; 5 wins, 2 ties, 3 losses (p = 0.73)** — no gain. **Decision: E1 stays the
GBDT alone.** The spread is small partly because the test sets overlap (above).

#### E2 (5 seeds, ~2 h each; one test split)

| test macro-F1 (5 seeds) | mean | sd | range |
| --- | --- | --- | --- |
| `flow_gbdt` | 0.832 | 0.002 | 0.829–0.834 |
| best single (`graph_gnn` every seed) | 0.868 | 0.002 | 0.864–0.870 |
| **deployed** (GBDT + GIN vote, temperature) | **0.892** | 0.001 | 0.890–0.894 |

**Deployed − best single: +0.024, [+0.020, +0.028]; 5 wins, 0 losses** (p = 0.062, the floor
for five). Because the split never moved, this shows the gain survives initialisation noise,
not that it survives other test days. **Outer folds 1–4** (seed 42, disjoint test days;
`results/ensemble_seeds/E2-folds.md`) were queued on 2026-11-11 to test that; until they land,
E2 keeps the vote on this evidence.
