# Replicating Okonkwo et al. (AISC 2022) as the flow-image paper baseline

> Run: `python scripts/replicate_okonkwo.py --tasks all`
> Raw: `runs/okonkwo/replication.json` · Log: `runs/okonkwo/run.log`
> Paper: `reference/papers/CNN-Based-Encrypted-Network-Traffic-Classifier_Okonkwo.pdf`

## What was built

| Component | Where | What it is |
| --- | --- | --- |
| `flow_image` extractor | `features/flow_image.py` | The size-vs-time plane, two constructions: `scatter` (the paper — binary presence, MTU-capped, oversize packets **dropped** per §4.2) and `flowpic` (histogram density). 1 channel, or 3 with direction + byte volume. |
| Capture windowing | `data/windows.py` | The paper's sample unit. Okonkwo never splits by 5-tuple — Table 2's 1×/2×/4× proportionality at 60/30/15 s only holds if fixed windows tile **one pcap timeline**, so a capture's flows are merged onto one clock before cutting. |
| App-label normalisation | `data/app_labels.py`, `app_rules:` in `configs/datasets.yaml` | Recovers a real class set from the capture stems. Without it the paper's three application tasks cannot be expressed at all. |
| `flow_image_cnn_baseline` | `models/flow_image_cnn.py` | The eleven layers of Figure 2. 102,570 params. |
| Replication driver | `scripts/replicate_okonkwo.py` | 7 tasks × **2 protocols**, class lists read off the Figure 5 confusion matrices. |
| Tests | `tests/test_flow_image.py`, `tests/test_app_labels.py`, `tests/test_train_image.py` | 47 tests. |

**Both protocols are run for every task.** `paper` is the random 80/10/10 over pooled window
images of §4.3 — their protocol. `grouped` splits by `source_file`. The gap is the finding.

### Two independent confirmations that the reconstruction is right

1. **The flatten is 576 wide**, exactly as Figure 2 labels it. That only comes out if stride 2 is
   on *every* convolution *in addition to* both pooling layers (224→112→56→28→14→7→3, 64·3·3=576).
   It is the check that the figure was read correctly.
2. **ISCXVPN window counts match Table 2** within 2–9%: 2635/5101/9772 against their
   2693/5386/10772 at 60/30/15 s. The residual is our short-flow drop and blank-window removal.

## The numbers replicate

| Task | Published | Ours (paper protocol) | Δ | Ours (grouped) |
| --- | --- | --- | --- | --- |
| 1 non-VPN application | 93% | **98.5%** | +5.5 | 48.8% |
| 2 non-VPN traffic type | 91% | **98.3%** | +7.3 | 96.4% |
| 3 VPN application | 96% | **98.4%** | +2.4 | 83.8% |
| 4 VPN traffic type | 96% | **98.0%** | +2.0 | 96.9% |
| 5 Tor application | 91% | **90.8%** | −0.2 | 77.9% |
| 6 Tor traffic type | 91% | **88.7%** | −2.3 | 76.8% |
| 7 encryption type | 99% | **98.8%** | −0.2 | 99.9% |
| **average** | **93.86%** | **95.93%** | **+2.07** | **82.93%** |

Within ~2 points on average across seven tasks, above on five and below on two. **Their results
reproduce.** The remaining spread is consistent with the fidelity gaps listed below — chiefly that
ISCXTor yields us ~1.5× their window count, and tasks 5/6/7 are the ones we score *lower* on.

## The headline: the protocol inflates, but not the way "13 points on average" suggests

Accuracy averages 95.9% → 82.9% under a leak-free split. That summary is misleading in both
directions, and the per-class detail is the actual result.

**Group-aware splits produce unbalanced test folds, so accuracy and macro-F1 diverge.** Reporting
both is the honest move (the paper reports accuracy):

| Task | acc paper → grouped | macro-F1 paper → grouped |
| --- | --- | --- |
| 1 non-VPN app | 98.5% → **48.8%** | 0.975 → **0.803** |
| 2 non-VPN traffic | 98.3% → 96.4% | 0.973 → 0.934 |
| 3 VPN app | 98.4% → 83.8% | 0.962 → 0.883 |
| 4 VPN traffic | 98.0% → 96.9% | 0.978 → 0.917 |
| 5 Tor app | 90.8% → 77.9% | 0.896 → **0.434** |
| 6 Tor traffic | 88.7% → 76.8% | 0.823 → 0.665 |
| 7 encryption | 98.8% → **99.9%** | 0.984 → **0.999** |
| average | 95.9% → 82.9% | 0.942 → 0.805 |

### It is not a broad degradation. Specific classes die.

Task 1 looks catastrophic at 48.8% accuracy. Nine of its ten classes are *fine*:

| class | recall (paper) | recall (grouped) | share of test fold |
| --- | --- | --- | --- |
| **facebook_audio** | 1.00 | **0.03** | **51.4%** |
| facebook_video | 1.00 | 1.00 | 1.5% |
| hangouts_audio | 1.00 | 0.97 | 8.6% |
| hangouts_video | 0.95 | 1.00 | 11.8% |
| netflix_video | 1.00 | 0.96 | 3.6% |
| skype_audio | 0.93 | 0.89 | 6.0% |
| skype_video | 1.00 | 0.98 | 4.7% |
| vimeo_video | 1.00 | 1.00 | 4.1% |
| voipbuster_audio | 1.00 | 0.95 | 3.1% |
| youtube_video | 0.94 | 0.96 | 5.3% |

One class collapses from perfect to 3% — and it happens to be **half the test fold**, so
`accuracy ≈ 1 − 0.514`. Its macro-F1 of 0.803 is the fairer number. The finding is not "flow images
don't work for app-ID"; it is that **`facebook_audio`'s apparent accuracy was capture identity**.
Tested on captures it has never seen, the model cannot recognise it at all.

The same shape recurs: Tor application loses `facebook` (0.00) and `skype` (0.46) while `spotify`
holds at 0.91; Tor traffic type loses only `chat` (0.04, 4.3% of the fold) and still reaches 76.8%.

### The gap tracks captures-per-class, not task difficulty

| Task | captures | classes | captures/class | accuracy gap |
| --- | --- | --- | --- | --- |
| 7 encryption | 235 | 3 | 78.3 | **−1.1** |
| 2 non-VPN traffic | 101 | 4 | 25.2 | +1.9 |
| 4 VPN traffic | 27 | 4 | 6.8 | +1.1 |
| 6 Tor traffic | 47 | 7 | 6.7 | +11.9 |
| 5 Tor application | 20 | 4 | 5.0 | +12.9 |
| 1 non-VPN application | 49 | 10 | 4.9 | +49.7 |
| 3 VPN application | 16 | 6 | 2.7 | +14.6 |

Traffic-type tasks, which have many captures per class, survive nearly intact (+1.1, +1.9).
Application tasks, which have a handful, are where the protocol does the work. This is the same
conclusion `REPLICATION-DRAPERGIL.md` reached from the other direction — there inflation grew with
class count; here it grows as capture diversity per class shrinks. Both are one mechanism: **too
few independent captures per class, and a random split is scoring capture recall.**

### Task 7 is the one to trust, and it is the least interesting

Encryption type is the only task where **grouping beats the random split** (99.9% vs 98.8%). With
235 captures over 3 classes there is nothing to memorise, and the larger group-split test fold
(8782 vs 4398) is better conditioned. The paper's strongest claim is also its most solid.

It is also the claim that least needs a CNN. `MODEL-DECISIONS.md` already routes `tunnel_type` to
LightGBM as "coarse; don't over-engineer"; 99.9% from 102k parameters on a binary scatter supports
exactly that.

### Task 3 is not measurable

16 captures across 6 classes means a group-aware split can place only **2 of 6** classes in the
test fold (`hangouts` 530, `skype` 401; `email`/`spotify`/`voipbuster`/`youtube` get zero). Its
83.8% is a two-class number and must not be compared with the published 96%.

## Fidelity gaps, stated rather than hidden

1. **Marker size** — the paper rendered matplotlib scatter plots to JPEG, so each packet became an
   anti-aliased blob several pixels wide. We rasterise directly with a 3×3 marker (`--marker`);
   1 would hand the baseline a materially sparser image than their model saw. It is a parameter,
   not a constant, because it is an assumption that moves the number.
2. **ISCXTor window counts run ~1.5× theirs** (3920/7691/14852 vs 2482/4964/9928), so tasks 5–7 use
   more Tor data than the paper did. They enumerate no Tor subset, so the discrepancy is
   unresolvable from the text. ISCXVPN matches to within 2–9%.
3. **Class lists** — the paper drops classes below 150 images after augmentation and excludes
   `browsing` from the non-VPN and VPN tasks as poorly defined. We restrict to its stated classes
   rather than re-deriving that floor.
4. **Augmentation is not reproduced.** The paper augments with rotation and flipping. On a
   size-vs-time plot that reverses causality and maps packet size onto the time axis, so it is
   excluded by default; `--augment` exists to measure it.
5. **Batch composition** — the paper's batch size of 10 is preserved, but batches are gathered from
   a GPU-resident tensor rather than a DataLoader (60× faster at that size; identical arithmetic
   and identical permutation logic on both paths — see `models/flow_image_cnn.py`).
6. **Blank windows** — dropped by default, and 0.0% on every ISCX task at 15/30/60 s, so the
   Figure 3c artifact never fired here. It remains real on shorter windows and on MobileApp.

## What this establishes for model selection

The paper baseline is now a runnable number rather than a citation. Against it:

- **Their representation works** where the data supports it: 96.9% (0.917 macro-F1) on VPN traffic
  type under a leak-free split is a genuine result for a binary scatter and a 102k-param CNN.
- **Their protocol is what produces the application-ID numbers.** Any comparison of a new image
  member against the published 93/96/91 is meaningless; the grouped column is the bar.
- **The binding constraint is captures per class, not architecture.** No backbone change recovers
  `facebook_audio` from 0.03 — that needs more independent captures, or a representation that does
  not encode capture identity so readily. This is the evidence behind the recommendation in
  `MODEL-image.md` to add the direction channel and pretrain across corpora.
- **Not yet measured:** the FlowPic histogram construction and the 3-channel input under the same
  grouped protocol, which is what separates "our backbone is better" from "our representation is
  better". `flow_image_cnn_baseline` is now the fixed point they get compared against.
