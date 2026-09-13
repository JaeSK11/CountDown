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
qsentinel/ensemble/
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
from qsentinel.ensemble import Ensemble
from qsentinel.eval import evaluate, ablation

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
