# Model: Flow-Stats (tabular)

**input_type:** `flow_stats` · **feature:** `features/flow_stats.py` (Phase 0, 56 cols)
**Primary targets:** traffic_type, tunnel_type (every target as a cheap baseline)
**Datasets:** `iscx_pooled`, CSTNET (+ any, as a universal baseline)
**Ensemble role:** the fast, interpretable, always-included member; also the Phase-5 **stacking
meta-learner**.

> **Status:** the recommended variant (`flow_gbdt`, LightGBM) shipped in **Phase 3** with measured
> results. The **paper baseline is now built and measured too** — see
> [`REPLICATION-DRAPERGIL.md`](REPLICATION-DRAPERGIL.md). This plan records what that produced.

## Two different "baselines" — don't conflate them

The selection brief lists three sources, and only one of them is a peer-reviewed paper. They are
separate baselines with separate purposes:

| | Draper-Gil et al., ICISSP 2016 | RADCOM `encrypted-traffic-classifier` |
| --- | --- | --- |
| What | The founding paper; **origin of ISCXVPN2016** | A competition repo, not peer-reviewed |
| Model | **C4.5 + k-NN** | **Random Forest** |
| Features | 23 **time-only** (no size stats) | flow features + TLS first-packet sizes + silence windows |
| Status | ✅ **built & replicated** (`flow_c45_paper`, `flow_knn_paper`) | ⬜ not built |
| Purpose | Reproduce the literature number and expose its protocol | A second tabular point of comparison |

Earlier revisions of this file named Random Forest as "the paper baseline". That was wrong — RF
comes from the repo, not the paper — and it mattered, because the actual paper baseline uses a
**different feature set** (23 time-only columns), which is most of what separates it from us.

## Baseline (paper) — C4.5 + k-NN ✅ built
- **Reference:** `reference/papers/Characterization-Encrypted-VPN-Traffic-Time-Related-Features_DraperGil-ICISSP2016.pdf`
- **Features:** `features/timeonly.py` (`@register_extractor("timeonly")`) — the paper's Table 2,
  23 columns, **deliberately no packet-size statistics**.
- **Segmentation:** `flows/segment.py` — the paper's "flow timeout" is a **duration cap**, not our
  64 s *idle* timeout. Without segmenting, the ftm sweep means nothing.
- **Models:** `@register("flow_c45_paper")` (entropy tree, J48 `minNumObj=2`),
  `@register("flow_knn_paper")` (MinMax + `k=1`, matching Weka IBk's in-distance normalisation).
- **Run:** `python scripts/replicate_drapergil.py --ftm 15 30 60 120`

### What it measured
All four published VPN-detection values reproduce within **0.02–0.05** under the paper's own
protocol:

| ftm | Model | Published | Ours (paper protocol) | Ours (grouped) |
| --- | --- | --- | --- | --- |
| 15 s | C4.5 | 0.898 | **0.9396** | 0.8441 |
| 15 s | k-NN | 0.847 | **0.8649** | 0.7401 |
| 120 s | C4.5 | 0.874 | **0.9217** | 0.8164 |
| 120 s | k-NN | 0.826 | **0.8577** | 0.7020 |

Two findings that change how this member is evaluated:

1. **Random 10-fold CV inflates by +0.10 (binary) to +0.35 (7-class).** Holding features, model and
   ftm fixed and changing *only* the fold assignment. The inflation grows with class count. This is
   why our own `iscx_pooled` 0.6317 is not comparable to published numbers in the 0.80s.
2. **The classifier barely matters; the features and the protocol do.** C4.5 and k-NN differ by
   ~0.07, reproducing the paper's own "C4.5 and KNN had similar results".

## Recommended — Gradient Boosting (LightGBM) ✅ shipped
- **Model:** LightGBM — `flow_gbdt` (Phase 3), `@register("flow_gbdt")`.
- **Measured:** macro-F1 **0.6317** (`iscx_pooled`/traffic_type, 8 classes), **0.8110**
  (CSTNET/app, 120 classes) — both under a group-disjoint split. See `PHASE-3.md` §12.
- **Why:** tree ensembles remain the tabular frontier at this data scale (Grinsztajn et al.
  NeurIPS 2022; Shwartz-Ziv & Armon 2022). Deep tabular models did **not** displace them.

### Bake-off: add CatBoost, XGBoost optional
Accuracy across LightGBM / XGBoost / CatBoost clusters within ~1 point when all are tuned, so
accuracy is *not* the reason to choose. Pick on mechanism:

| Candidate | Why it might win here | Verdict |
| --- | --- | --- |
| **LightGBM** | already shipped; fast | **primary** |
| **CatBoost** | *ordered boosting* fights the target leakage standard boosting introduces → most robust on small/imbalanced data (`p2p`, `chia.net`=16). *Oblivious trees* = strongest regulariser and by far the fastest inference | **add to scorecard** |
| **XGBoost** | level-wise growth is more conservative than LightGBM's leaf-wise on small data | optional — mechanically closest to LightGBM, so it adds the least new information |

CatBoost's categorical handling is irrelevant here (all 56 columns numeric); its *other* two
mechanisms are the reason it belongs in the comparison.

⚠️ **Do not promote two GBDTs to separate ensemble members.** Three boosted-tree variants on the
same `flow_stats` vector produce highly correlated errors and add almost nothing to a combiner that
already holds bytes / sequence / image / graph members. One winner becomes the member; the others
exist only to justify that pick.

## ⚠️ Imbalance: `class_weight="balanced"` is costing accuracy

`flow_gbdt` ships `class_weight="balanced"`. Measured on ISCXVPN (ftm 15 s, grouped folds,
`n_estimators=200`), turning it **off** improves every cell — and the damage grows with class count:

| Scenario | Features | balanced | unweighted | Δ |
| --- | --- | --- | --- | --- |
| A1 (2 classes) | flow_stats | 0.9735 | 0.9753 | +0.002 |
| A1 (2 classes) | timeonly | 0.8225 | 0.8684 | **+0.046** |
| A2 non-VPN (6) | timeonly | 0.3169 | 0.4578 | **+0.141** |
| A2 non-VPN (6) | flow_stats | 0.3431 | 0.5019 | **+0.159** |

*(avg precision; macro-F1 moves the same way — 0.3840 → 0.4599 on the last row.)*

On the 6-class task this is the **single largest lever measured** — bigger than the feature set and
bigger than the model. It also cuts against the reason the setting was adopted: `PHASE-3.md` §5.2
pairs balanced weights with a macro-F1 headline, but macro-F1 is one of the metrics it hurts.

**Not yet a config change.** This was measured on ISCXVPN alone at ftm 15 s with 200 rounds and no
early stopping; the Phase-3 headline (`iscx_pooled`, 600 rounds, early stopping) is a different
configuration. Re-run there before flipping the default — but treat `balanced` as **suspect, not
settled**, and do not add it to new members by default.

Whatever is chosen, the Phase-5 convention still applies (`PHASE-5.md` §1): calibration fitted on a
split carrying the **true** class distribution largely corrects a reweighted posterior, and **every
member must follow the same convention** or the combiner sees members on different implicit priors.

## Training & eval
- Shared Phase-3 harness; **group-aware split** (`source_file` for ISCX, `capture_day` for CSTNET);
  macro-F1 headline.
- Same features/split/target across variants → the delta is attributable to the model.
- Interpretability is preserved, not lost: TreeSHAP gives exact per-prediction attributions on a
  GBDT in near-linear time — more useful than a C4.5 tree that is hundreds of nodes deep in
  practice. `shap` is **not yet installed**.

## Scorecard — DoD 2 resolved ✅

`scripts/ablate_features_vs_model.py` varies representation and classifier independently, with the
fold assignment, segmentation and scenario held fixed. Grouped folds, ftm 15 s, avg precision:

**A1 — binary VPN detection**

| Features | C4.5 | k-NN | LightGBM (bal) | LightGBM (unw) |
| --- | --- | --- | --- | --- |
| `timeonly` (23) | 0.8441 | 0.7401 | 0.8225 | 0.8684 |
| `flow_stats` (56) | 0.9575 | 0.8236 | 0.9735 | **0.9753** |

**A2 non-VPN — 6-class characterisation**

| Features | C4.5 | k-NN | LightGBM (bal) | LightGBM (unw) |
| --- | --- | --- | --- | --- |
| `timeonly` (23) | 0.3593 | 0.3805 | 0.3169 | 0.4578 |
| `flow_stats` (56) | 0.3804 | 0.4252 | 0.3431 | **0.5019** |

### The answer flips with the task

| Effect | A1 (2 classes) | A2 (6 classes) |
| --- | --- | --- |
| **Features** (`timeonly` → `flow_stats`, model fixed) | **+0.107** | +0.044 |
| **Model** (C4.5 → LightGBM, features fixed) | +0.018 | **+0.122** |

So neither "our model is better" nor "our features are better" is universally true:

- **Binary tunnel detection is a feature problem.** `timeonly` carries no packet-size statistic at
  all, and the VPN signature is largely a size signature — a 23-column model cannot see it. Swapping
  the classifier buys +0.02; adding the size columns buys +0.11.
- **Multiclass characterisation is a model problem.** Both representations hold roughly the same
  information; carving six overlapping behavioural classes out of it needs boosting. Features buy
  +0.04; the model buys +0.12.

One more result worth keeping: on `timeonly`, **LightGBM with the shipped balanced weights is worse
than a plain unpruned decision tree** (0.8225 vs 0.8441 on A1; 0.3169 vs 0.3593 on A2). Only after
unweighting does it lead. A default can cost more than an architecture.

## DoD
1. ✅ Paper baseline built, replicated, documented (`REPLICATION-DRAPERGIL.md`).
2. ✅ **Feature effect separated from model effect** (`scripts/ablate_features_vs_model.py`,
   `runs/ablate_features_vs_model.json`). Answer above; it is task-dependent.
3. ⬜ Re-test `class_weight` on the Phase-3 config (`iscx_pooled`, 600 rounds, early stopping)
   before changing the default. **Highest-value open run.**
4. ⬜ CatBoost added to the scorecard.

## Risks
Mature methods; low modelling risk. The real risks are evaluation-shaped and now quantified:
protocol leakage (+0.10…+0.35), and the feature/model confound in DoD 2.
