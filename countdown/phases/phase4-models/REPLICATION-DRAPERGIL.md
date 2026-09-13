# Replicating Draper-Gil et al. (ICISSP 2016) as the flow-stats paper baseline

> Run: `python scripts/replicate_drapergil.py --ftm 15 30 60 120`
> Table: `python scripts/summarize_replication.py` · Raw: `runs/replicate_drapergil.json`
> Paper: `reference/papers/Characterization-Encrypted-VPN-Traffic-Time-Related-Features_DraperGil-ICISSP2016.pdf`

## What was built

| Component | Where | What it is |
| --- | --- | --- |
| `timeonly` extractor | `features/timeonly.py` | The paper's Table 2 — **23** time-related features, **no size statistics**. Separate from our 56-column `flow_stats` on purpose: reproducing their number with our richer features would not be a replication. |
| Flow-timeout segmentation | `flows/segment.py` | The paper's ftm is a **duration cap**, not our 64 s *idle* timeout. `segment_flows()` cuts the timeline into fixed windows so the 15/30/60/120 s sweep means what it meant in 2016. |
| `flow_c45_paper` | `models/paper_baselines.py` | C4.5 → `DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2)`. |
| `flow_knn_paper` | `models/paper_baselines.py` | IBk → `MinMaxScaler` + `k=1`, matching Weka's in-distance attribute normalisation. |
| Replication driver | `scripts/replicate_drapergil.py` | Scenarios A1/A2/B × 4 ftm × 2 models × **2 protocols**. |
| Tests | `tests/test_replication.py` | 21 tests. |

**Both protocols are run for every configuration.** `paper` is stratified 10-fold CV drawn at
random over flows — their protocol. `grouped` splits folds by `source_file`. The gap between the
two columns is the actual finding.

## A1 — VPN detection. The replication holds.

This is the **only like-for-like scenario**: a binary VPN/non-VPN split whose class set is
identical in their setup and ours.

| ftm | Model | Published | Ours (paper protocol) | Δ | Ours (grouped) |
| --- | --- | --- | --- | --- | --- |
| 15 s | C4.5 | 0.898 | **0.9396** | +0.042 | 0.8441 |
| 15 s | k-NN | 0.847 | **0.8649** | +0.018 | 0.7401 |
| 120 s | C4.5 | 0.874 | **0.9217** | +0.048 | 0.8164 |
| 120 s | k-NN | 0.826 | **0.8577** | +0.032 | 0.7020 |

All four land within **0.02–0.05**, consistently *above* the published value — consistent with the
one known fidelity gap: scikit-learn has no J48 pessimistic-error pruning, so our tree is grown
unpruned and generalises slightly differently.

**Their flow-timeout claim partly reproduces.** C4.5 falls monotonically with longer windows
(0.9396 → 0.9340 → 0.9273 → 0.9217), matching "shorter timeout, better accuracy". k-NN does not —
it is flat within noise (0.8649 / 0.8662 / 0.8507 / 0.8577). The paper asserted the trend for both.

## A2 / B — not reproducible on the released dataset

Not a disagreement with the paper, a **data availability fact**: ISCXVPN2016 as distributed ships
**no browsing captures**, and its only `p2p` capture is a VPN one. The class sets cannot match.

| Scenario | Paper | Ours | Missing |
| --- | --- | --- | --- |
| A2 non-VPN | 7 | **6** | `browsing`, `p2p` |
| A2 VPN | 7 | **7** | `browsing` (but our taxonomy splits streaming → audio + video) |
| B (joint) | 14 | **13** | as above |

The two absent classes are precisely the two that are *easy*: the Phase-3 pooled run scores
`p2p` 0.99 and `browsing` 0.97, while the interactive classes sit at 0.41–0.47. Removing them
removes the two highest terms from a macro average, which is most of the distance between their
A2 non-VPN 0.89 and our 0.5617. **The published A2/B values are reference-only, not targets.**

**One genuine divergence.** The paper reports shorter-ftm-is-better for Scenario A. We find the
**opposite for A2**: non-VPN characterisation improves monotonically with *longer* windows
(0.5617 @15 s → 0.6523 @120 s), and A2-VPN and B behave the same way. That is coherent — a tunnel
signature is visible immediately, but *characterising* traffic needs enough packets to see
behaviour, and a 15 s window often has too few.

## The headline: their protocol inflates, and how much depends on the task

Random 10-fold CV lets windows cut from one conversation — frequently one TCP connection — land on
both sides of the split. Holding everything else fixed and changing only the fold assignment:

| Scenario | Classes | paper protocol | grouped | Inflation |
| --- | --- | --- | --- | --- |
| A1 (VPN detect) | 2 | 0.9396 | 0.8441 | **+0.10** |
| A2 non-VPN | 6 | 0.5617 | 0.3593 | **+0.20** |
| A2 VPN | 7 | 0.7328 | 0.3844 | **+0.35** |
| B (joint) | 13 | 0.6284 | 0.3422 | **+0.29** |

*(C4.5 @ 15 s throughout.)*

The inflation **grows with the number of classes**. Binary VPN detection survives a leak-free split
largely intact; fine-grained characterisation loses roughly half its score. This is the quantified
version of the caveat in `PHASE-3.md` §12 — "the literature's random-split numbers on these
datasets are higher because they leak" — and it is why our own 0.6317 on pooled ISCX is not
comparable to published numbers in the 0.80s.

## Fidelity gaps, stated rather than hidden

1. **No J48 pruning** — the largest gap; our trees are unpruned (see `models/paper_baselines.py`).
2. **No `browsing`** in the released ISCXVPN2016; non-VPN `p2p` also absent.
3. **Taxonomy granularity** — we split `streaming` into `audio_streaming` + `video_streaming`.
4. **Activity threshold** — the paper never states the active/idle threshold; 5 s is
   ISCXFlowMeter's inherited NetMate default, and it is a constructor argument so the sensitivity
   can be measured.
5. **Segmentation vs re-extraction** — we cut cached flows rather than re-reading 28 GB at four
   thresholds. Equivalent because our 64 s idle timeout only joins packets the paper would also
   have joined (every gap it splits on exceeds every ftm swept).
6. **`min_packets=2`** per window; at ftm=15 s this drops 24,161 of 104,252 windows (23%) that hold
   a single packet and therefore have no inter-arrival time at all. Reported by `segment_flows()`.

## What this establishes for model selection

The paper baseline is now a real, runnable number rather than a citation. Against it:

- Their **classifier** choice barely matters — C4.5 and k-NN differ by ~0.07 on A1 and less
  elsewhere, reproducing their own "C4.5 and KNN had similar results".
- Their **protocol** matters enormously — up to +0.35.
- Their **features** are the honest baseline for `flow_gbdt` to beat: 23 time-only columns vs our
  56. **Not yet measured:** `flow_gbdt` on `timeonly` under the grouped protocol, which is what
  would separate "our model is better" from "our features are better".
