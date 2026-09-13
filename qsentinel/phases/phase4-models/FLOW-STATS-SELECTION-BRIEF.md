# Flow-Stats Member — Model-Selection Brief (open discussion)

> Purpose: context for choosing a classifier for the **flow-statistics** representation. Covers the
> associated papers and the models *they* used, plus the dimensions to weigh. **No selection is made
> here — the choice is intentionally left open for discussion.**

## 1. The representation
Per-flow **tabular** features computed from the packet stream — no payload needed, so it works on
encrypted traffic:
- ISCXFlowMeter / CICFlowMeter style: packet-size stats (min/max/mean/std, per direction), inter-
  arrival-time (IAT) stats, flow duration, byte/packet counts, up/down ratios (~several dozen
  features).
- Behavioral additions seen in practice: **TLS first-packet sizes** (a handshake "fingerprint"),
  **silence windows**, **inter-arrival timing**.
Each flow → one fixed-length numeric vector → a standard tabular classifier.

## 2. Associated work and the models it used

| Source | What it is | Model(s) used | Stated rationale |
| --- | --- | --- | --- |
| **Draper-Gil et al., ICISSP 2016** — "Characterization of Encrypted and VPN Traffic Using Time-Related Features" (origin of ISCXVPN2016) | The foundational flow-stats paper for this dataset | **C4.5 decision tree** and **k-Nearest Neighbors (k-NN)** on time-related flow features | Time-based features are payload-independent (work on encrypted/VPN); simple interpretable classifiers were sufficient; evaluated across flow-timeout settings |
| **Lashkari et al., ICISSP 2017** — "Characterization of Tor Traffic Using Time Based Features" (origin of ISCXTor2016) | Companion Tor/non-Tor study | **Tree-based / classical classifiers** on the same time-based features | Same payload-independent, time-feature approach extended to Tor |
| **RADCOM `encrypted-traffic-classifier`** (the cloned reference repo) | A competition project (not a peer-reviewed paper) | **Random Forest** on flow features + TLS first-packet fingerprint + silence/IAT features | "Robustness under limited labeled data and resistance to overfitting in encrypted-traffic scenarios" |

**Takeaway:** across this lineage the representation is constant (statistical flow features); the
classifier has ranged over **decision trees (C4.5), k-NN, and Random Forest** — all classical,
interpretable, payload-independent methods.

## 3. Why those models were chosen (their reasoning, to inform yours)
- **Payload independence** — features are purely statistical, so any generic tabular classifier fits.
- **Interpretability** — trees/rules explain a decision, which is valued in a security setting.
- **Limited, imbalanced labeled data** — favored robust, low-variance methods over data-hungry ones.
- **Low latency** — high-volume traffic rewards cheap inference.

## 4. Dimensions to weigh when selecting (discussion checklist)
- **Accuracy vs complexity** on tabular features.
- **Number of classes** — a handful of traffic types vs 120 apps behaves very differently.
- **Class imbalance** — some categories have <1000 samples (BitTorrent/Google/Twitter noted in the
  Okonkwo paper).
- **Interpretability / explainability** needs for the security use-case.
- **Training-data size** available per target.
- **Inference latency** — does it need to run in the real-time path?
- **Probability calibration quality** — matters because this member feeds a downstream
  ensemble/combiner (calibrated probabilities combine better).
- **Tuning burden / reproducibility**.

## 5. Candidate model families for tabular flow-stats (neutral landscape)
Listed as the option space to discuss — **not ranked, no recommendation**:

| Family | Rough character | Notes for discussion |
| --- | --- | --- |
| Decision tree (C4.5 / CART) | interpretable, low variance capacity | used by Draper-Gil; weak alone, strong as an ensemble base |
| k-Nearest Neighbors | no training, instance-based | used by Draper-Gil; slow at inference, scaling-sensitive |
| Random Forest | bagged trees, robust | used by RADCOM; strong default, easy to tune |
| Gradient-boosted trees (XGBoost / LightGBM / CatBoost) | boosted trees | typically strong on tabular; more hyperparameters |
| SVM (RBF/linear) | margin-based | scaling-sensitive; can be slow at large N |
| Logistic regression | linear baseline | fast, calibrated, a useful floor |
| Shallow MLP | small neural net | flexible; needs more data/tuning, less interpretable |
| Tabular deep (TabNet, FT-Transformer) | attention on tabular | newer, data-hungry; often matched by tree ensembles |

## 6. Open question for the discussion
Given payload-independent features, modest+imbalanced labeled data, a security-explainability angle,
and that this member's probabilities feed a downstream combiner — **which of the above (or
combination) best fits, and under which target (traffic-type vs app-ID)?** This brief deliberately
stops short of answering.

---

# 7. Resolved (2026-08-24) — the discussion this brief opened

The brief above is kept as written: it is the record of where the discussion *started*, and its
neutrality was deliberate. This section records where it ended, and what evidence closed it.

## The answer

**LightGBM on the 56-column `flow_stats` vector**, with **CatBoost added to the bake-off** and
XGBoost optional. Rationale, scorecard and the imbalance/calibration convention live in
[`MODEL-flow-stats.md`](MODEL-flow-stats.md).

Tree ensembles remain the tabular frontier at this data scale — deep tabular models (TabNet,
FT-Transformer, §5's last row) did not displace them, and this dataset is far from the regime where
they compete. Of the three GBDT implementations, accuracy clusters within ~1 point; CatBoost earns
its place in the comparison on *mechanism* (ordered boosting for small/imbalanced data, oblivious
trees for inference speed), not on expected accuracy.

## What actually decided it — and it was not §4's checklist

The paper baseline was built and run
([`REPLICATION-DRAPERGIL.md`](REPLICATION-DRAPERGIL.md)), which reordered the dimensions in §4:

- **"Accuracy vs complexity" barely discriminates.** C4.5 and k-NN land ~0.07 apart, and the three
  GBDTs within ~1 point of each other. The *classifier* is not where the variance is.
- **Evaluation protocol dominates everything on the list.** Random 10-fold CV over flows inflates by
  **+0.10 (binary) to +0.35 (7-class)** versus a `source_file`-grouped split. That is larger than
  every model-choice effect combined, and it is not in §4 at all.
- **Feature set is the second-largest effect.** 23 time-only columns vs our 56 — still unseparated
  from the model effect (see `MODEL-flow-stats.md` DoD 2).
- **"Inference latency" does not discriminate.** The paper's own figures are ~99.9% flow-build
  window (15.011 s of which 0.01 s is k-NN). The flow timeout sets the real-time floor, not the
  model.
- **"Probability calibration quality" survives as a real dimension**, but the resolution is a
  cross-member *convention* (`PHASE-5.md` §1), not a model choice — every candidate here needs
  post-hoc calibration equally.

## Two corrections to this brief's own text

- **§1 overstates the shared representation.** Draper-Gil's features are **time-only** — no
  packet-size statistics whatsoever. The size stats in §1 come from later CICFlowMeter versions, not
  from the 2016 paper. The two are separate extractors here: `timeonly` (23) vs `flow_stats` (56).
- **§2's "Takeaway" is right that the representation is constant across the lineage — but it is
  not.** The representation *grew*; that growth is part of what the newer numbers measure.

## What the brief got right

§4's **class imbalance** bullet understated the problem rather than overstating it. On ISCXVPN
`p2p` is a **single capture file**, so under a group-aware split it is not imbalanced — it is
unlearnable, and the dataset must be pooled with ISCXTor for the target to be viable at all
(`PHASE-3.md` §6).
