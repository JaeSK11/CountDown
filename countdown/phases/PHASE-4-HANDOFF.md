# Phase 4 — Handoff: what is done, what is left

**Snapshot:** 2026-09-13, commit `586d6bb`. Verified against the code, `runs/` and a full test run —
not only against the other docs, several of which lag the results on disk (§3).
**Updated 2026-09-14:** §0 (fix verified; a second, hardware-level GPU blocker found), §4 A, §6.
**Updated 2026-09-16:** §0 (torch fix applied to `.venv`; GPU stable so far, cause not found), §4 A.1, §6.

**Read first, in order:** `../PLAN.md` → `PHASE-4.md` (task list + exit criterion) →
`MODEL-DECISIONS.md` (lessons that bind every remaining baseline) → the `phase4-models/MODEL-*.md`
for whichever member you pick up. `ENSEMBLE-MAP.md` and `PHASE-5.md` §1 constrain design choices.

---

## 0. Blocker — fix before any GPU training

**2026-09-16: the PyTorch blocker is fixed; the hardware one is unexplained but has not recurred.**
With the user's OK, `.venv` now has `torch 2.14.0+cu130` / `torchvision 0.29.0+cu130` (pre-upgrade
`pip freeze` in `runs/venv-freeze-before-torch2.14-20260916.txt`). `pip check` is clean,
`torch_geometric` 2.8.0.post1 imports, and the suite is **509/509 in `.venv`**, alone on the GPU
(`runs/pytest-torch2.14cu130-venv-20260916.log`). Okonkwo seed 1 ran all three tasks for the first
time on the 5090 (`runs/okonkwo/seeds/base_s1_torch2.14cu130_20260916.json`, 76 s). Grouped
macro-F1 vs `base_s1.json` (RTX 3090, torch 2.5.1): task 1 0.755 vs 0.754, task 4 0.902 vs 0.871,
task 5 0.442 vs 0.442. Task 4 on the 5090 has now given 0.879, 0.903 and 0.902 over three runs,
so its spread against the 3090 value is run-to-run non-determinism, not the failing card. Note:
the `.venv/bin/*` launchers still point at the pre-rename `qsentinel/.venv`, so `.venv/bin/pip`
and `.venv/bin/pytest` fail with "required file not found" — use `.venv/bin/python -m pip` /
`-m pytest`. The history below is kept as found.

**The installed PyTorch cannot run on this machine's GPU.** The box now has an **RTX 5090**
(compute capability `sm_120`); the venv has `torch 2.5.1+cu121`, which supports up to `sm_90`.
`torch.cuda.is_available()` still returns `True`, so nothing fails until the first kernel:
`RuntimeError: CUDA error: no kernel image is available for execution on the device`.

- This is the cause of the only 2 test failures: `pytest` → **507 passed, 2 failed**, both in
  `tests/test_train_image.py` (`test_resident_and_streaming_paths_agree`,
  `test_oversize_tensor_falls_back_to_streaming`).
- Fix: a CUDA ≥ 12.8 build of PyTorch (≥ 2.7) plus matching `torchvision`; re-check
  `torch_geometric 2.8.0` still imports. **Confirm with the user before changing the venv.**
  **Verified 2026-09-14 in an isolated venv (not applied to `.venv`):** `torch==2.14.0+cu130`,
  `torchvision==0.29.0+cu130` from `https://download.pytorch.org/whl/cu130`, `torch-geometric`
  2.8.0.post1 unchanged. The 610.43.02 driver (CUDA 13.3) accepts it; matmul, conv fwd/bwd, AMP and a
  PyG `SAGEConv` all run on the 5090; the `test_train_image.py` GPU tests pass; Okonkwo seed 1,
  tasks 1 and 4, reproduce `runs/okonkwo/seeds/base_s1.json` on identical splits — two attempts,
  both cut off at task 5 by the bus fault below (`runs/okonkwo/seeds/base_s1_torch2.14cu130.json`):
  task 1 +0.007 / +0.006, task 4 +0.008 / +0.032 macro-F1. The 0.032 spread between two runs of the
  same seed on the same split is GPU non-determinism and is the same size as the across-seed sd
  (§3.2) — one more reason not to read single-seed gaps under 0.05. **The full suite is 509/509 on
  this build** when it runs alone on the GPU (`runs/pytest-torch2.14cu130-509-passed.log`). Why this build: the cu128 index stops at torch 2.11, cu126 wheels carry
  no `sm_120`, and no local CUDA toolkit (`nvcc`) exists to match, so the current cu130 line is the
  one with a future. One new, harmless `FutureWarning` (`torch.jit.script` deprecated) on PyG import.
  To apply:
  `.venv/bin/pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130`
- **Second blocker, hardware-level: the GPU drops off the PCIe bus.** `journalctl -k` shows
  `Xid 79, GPU has fallen off the bus` **three times**: 2026-09-11 01:00:05 (previous boot,
  nothing running), 2026-09-14 00:03:33 (two small-CNN processes, ~90 s in) and, after a reboot,
  **2026-09-14 00:45:45 with a single process, ~65 s into `replicate_okonkwo.py` seed 1** — a
  0.9 GB resident tensor and a 102k-param CNN at batch 10, i.e. nowhere near the card's limits.
  The 45 s test suite passed just before it. The pattern is sustained load for about a minute,
  then loss of the bus. `Xid 154` sets the recovery action to "OS Reboot"; afterwards `nvidia-smi` finds no
  device, the PCI config space reads all-ones, and both venvs report `cuda.is_available() == False`.
  Two events in four days, one at idle, point at power / PCIe / driver rather than at a workload
  (driver 610.43.02 open kernel module; nothing was hot — 33 °C, P8 before the runs). For the user,
  not the agent: cold power-cycle rather than warm reboot; check the 12V-2x6 cable seating; try
  PCIe Gen 5 → Gen 4 in BIOS and/or a power limit (`nvidia-smi -pl`) as a diagnostic; consider a
  newer driver. Until it is understood, run GPU jobs **one at a time** and check
  `journalctl -k | grep Xid` after any CUDA "unspecified launch failure".
  **2026-09-16:** no drop since the 09-14 power-off (box was off ~2.5 days). Under a logger
  (`~/gpu-watch/watch.sh`: nvidia-smi every 200 ms, root-port link state every 100 ms, kernel PCIe
  tracepoints) the card survived a 6.5-min synthetic load (avg 269 W, peak 297 W, 63 °C), the suite
  and the full seed-1 run, with the link at Gen 5 x16 throughout. Still unresolved: the user reports
  that at each drop both the GPU fans and the case fans jump to full speed, which points at a brief
  12 V dip more than at the PCIe link. Facts gathered: Dell OEM 5090 (subsystem 1028:5305), 1200 W
  PSU on a single native 12V-2x6 cable, Gigabyte X870E AORUS MASTER X3D ICE on BIOS F7 (F9 is
  current), power-limit range 400–575 W (so `-pl 300` is rejected). The BIOS update and a cable
  reseat are the user's to do. Keep running GPU jobs one at a time, under the logger when possible.
- Every number in `runs/` was produced on the earlier GPU and torch 2.5.1. After the upgrade,
  re-run one existing result (e.g. an Okonkwo seed) to check it still reproduces before comparing
  new runs against old ones.
- Docs naming the RTX 3090 / A4000 were updated 2026-09-14 (`PHASE-4.md` §8, `MODEL-bytes.md`).

Environment otherwise: `.venv/bin/python`; installed `torch_geometric`, `torchvision`; **not**
installed: `mamba-ssm`, `timm`, `catboost`, `xgboost`, `dgl`, `shap`. None of the Phase-4 deep deps
are declared in `pyproject.toml` yet.

---

## 1. Ground rules (from the existing docs — do not relitigate)

1. **Two variants per member:** `<name>_baseline` (the paper, faithfully) and `<name>` (our
   recommended upgrade). Both subclass `BaseModel`, register via `@register`, and return
   `predict_proba` in `LabelSpace` column order.
2. **The headline is macro-F1 under a group-aware split** (`source_file`, or `capture_day` on
   CSTNET). Published numbers are not targets. Paper-protocol runs are allowed only as a
   replication diagnostic, reported next to the grouped number.
3. **A baseline reproduces the paper's features and protocol, not only its model** — budget a
   paper-faithful extractor (`MODEL-DECISIONS.md` "What building the first baseline changed").
4. **Byte and graph inputs strip IPs/ports**, and must pass `audit_endpoint_leakage()` before a
   number is believed. Measured twice already (§3.1, §3.3): addressing bytes inflate results hugely.
5. **Same class-weight / calibration convention for every member** (`PHASE-5.md` §1).
6. **No image augmentation that rotates or flips** a size-vs-time plot.
7. **Heavy deps are lazy-imported** so a missing one cannot break `ModelRegistry`.
8. `runs/` and `cache/` are **gitignored** — results exist only on this machine. Docs cite them by
   path.

---

## 2. Status by task (`PHASE-4.md` §6)

| # | Task | State | Evidence / gap |
| --- | --- | --- | --- |
| 4.1 | `training/deep.py` shared trainer | 🟡 built, under-used | 524 lines: early stop on val macro-F1, checkpoint + resume, UER-style AdamW. **Only `byte_net` uses it.** `flow_image_cnn` has its own loop; TFE-GNN trains inside `scripts/replicate_tfegnn.py`. **No test covers `deep.py`** (DoD "trains a toy net" unverified). |
| 4.2 | `features/payload_bytes.py` | 🟡 partial | ET-BERT bi-gram tokenizer is exact. Extractor yields handshake-window bytes only — **not BURST-capable**; needs per-packet payload retention. `features/byte_prep.py` already does a separate byte-retaining pcap pass (for TFE-GNN) and may be reusable. The ET-BERT replication reads the released corpus instead (`data/etbert_corpus.py`). |
| 4.3 | `features/flow_image.py` | ✅ | `scatter` + `flowpic`, 1 or 3 channels (3 = + direction + byte volume). `data/windows.py` gives capture- or per-flow windows. |
| 4.4 | graph feature | 🟡 built, not wired | Lives in `features/traffic_graph.py` (plan said `graph.py`). Verified against the authors' `construct_graph`. **Not a registered extractor** — only the replication script calls it. |
| 4.5 | **Sequence** (DF + dilated-res → CNN+Mamba) | ⬜ **not started** | No model file. `packet_seq` defaults to 32 packets × 2 channels (`configs/features.yaml`); DF expects long direction sequences. Selection brief still open. |
| 4.6 | **Bytes** (ET-BERT + recommended) | 🟡 baseline done | `byte_net_baseline` registered, loads the public checkpoint with no key remapping. Flow + packet replications finished (§3.1); no-pretrain ablation done 2026-09-17 (0.688 vs 0.908, `MODEL-bytes.md`). **Left:** flow `--seq-length 512` test (optional), choice + build of the recommended model (D1), scorecard on our own pipeline. |
| 4.7 | **Graph** (TFE-GNN + SAGE/GIN) | 🟡 baseline registered | `models/tfe_gnn.py` (PyG `SAGEConv`) + `tests/test_tfe_gnn.py`; **registered as `graph_gnn_baseline` 2026-09-17** (`models/graph_gnn.py`, `tests/test_graph_gnn.py`, `scripts/train_graph_gnn.py` — `input_type=byte_matrix`, no extractor yet, driven from the cache). Trained on ISCX-Tor only (§3.3; grouped 5-fold 0.44 ± 0.14 under the paper's no-val protocol, `REPLICATION-TFEGNN.md`). Caches for `vpn/nonvpn/tor/nontor` are built (31/109/51/44 `.npz` shards, one per capture, plus a `_summary.json` each) but only Tor was trained. Nothing on CSTNET, the plan's primary target. Write-up: `phase4-models/REPLICATION-TFEGNN.md` (2026-09-14). GraphSAGE/GIN not built. |
| 4.8 | **Image** (Okonkwo + FlowPic) | 🟡 baseline done, variants run | `flow_image_cnn_baseline` replicated on 7 tasks (`REPLICATION-OKONKWO.md`). `flow_image_resnet18` registered. FlowPic 1ch / 3ch / ResNet-18 variants + a 5-seed sweep are done and written up in `MODEL-image.md` (2026-09-14; §3.2). **Recommended `flow_image_cnn` is not registered** (D1). MobileApp activity baseline run 2026-09-17: **0.165 macro-F1** with the paper schedule (the config's early stop had to be removed — it fired at chance); tasks 2/6 seed sweep done. No latency column yet. |
| 4.9 | **Flow-stats** comparison | 🟡 mostly done | Draper-Gil C4.5 + k-NN replicated; features-vs-model ablation done. `class_weight` re-test done 2026-09-17 (unweighted +0.014, → D5). **Left:** CatBoost (not installed — ask). RADCOM Random Forest: not built, optional. |
| 4.10 | **`experts/`** bindings | ⬜ **not started** | No directory. See decision D2. |

Registered today: `byte_net_baseline, flow_c45_paper, flow_gbdt, flow_image_cnn_baseline,
flow_image_resnet18, flow_knn_paper, graph_gnn_baseline` (the last since 2026-09-17). The acceptance test in `PHASE-4.md` §7 also expects
`seq_cnn_baseline, seq_cnn, byte_net, graph_gnn_baseline, graph_gnn, flow_image_cnn`.

---

## 3. Results on disk that the docs did not reflect until 2026-09-14

*Written into `MODEL-bytes.md`, `MODEL-image.md`, `MODEL-graph.md`, a new `REPLICATION-TFEGNN.md`
and the README (step A.2). Kept here as the cross-check against `runs/`.*

### 3.1 ET-BERT, packet level — finished (`MODEL-bytes.md` still says "_running_")
`runs/etbert-repro/packet.json`: **macro-F1 0.9076, accuracy 0.9066** vs paper 0.9741 / 0.9737
(**Δ −0.0665**). 132.2M params. Endpoint-leakage audit on this corpus: 0.0 (clean, as expected).
Flow level, already documented: 0.8585 vs 0.9426.

### 3.2 Image variants (`runs/okonkwo/`) — `MODEL-image.md` scorecard still blank
Grouped protocol, macro-F1, **single seed (42)**, windows 15/30/60 s, 60 epochs:

| Task | scatter (baseline) | FlowPic 1ch | FlowPic 3ch | ResNet-18, FlowPic 3ch |
| --- | --- | --- | --- | --- |
| 1 non-VPN app (10 cls) | 0.803 | 0.798 | 0.642 | 0.741 |
| 2 non-VPN traffic (4) | 0.934 | 0.939 | 0.943 | 0.955 |
| 3 VPN app | *2 classes in test fold — not measurable* | | | |
| 4 VPN traffic (4) | 0.917 | 0.860 | 0.968 | 0.949 |
| 5 Tor app (4) | 0.434 | 0.310 | 0.346 | 0.392 |
| 6 Tor traffic (7) | 0.665 | 0.669 | 0.710 | 0.747 |

5-seed sweep (`runs/okonkwo/seeds/`, same model, tasks 1/4/5), mean ± sd:

| Task | scatter | FlowPic 3ch |
| --- | --- | --- |
| 1 non-VPN app | 0.779 ± 0.022 | 0.794 ± 0.029 |
| 4 VPN traffic | 0.864 ± 0.028 | **0.930 ± 0.027** |
| 5 Tor app | 0.442 ± 0.010 | 0.397 ± 0.038 |

Configs are identical to the single-seed runs except the seed — and in `replicate_okonkwo.py` the
seed sets **both** the init and the group split (which captures land in test), so the sd above is
split + init variance. With sd 0.01–0.04, single-seed gaps under ~0.05 in the first table are not
evidence. Task 1's 0.642 (seed 42) is far below FlowPic 3ch's 5-seed mean, so read it as a hard
fold, not a construction effect. Two differences hold across seeds: FlowPic 3ch on VPN traffic
(+0.066; 4 of 5 FlowPic seeds above every scatter seed) and FlowPic 3ch on Tor app (**−0.045**;
every FlowPic seed below every scatter seed).

### 3.3 TFE-GNN on ISCX-Tor — undocumented anywhere
Published F1 0.9855. All single seed, 100 epochs:

| Split | addressing stripped (ours) | addressing kept (`--keep-addressing`, authors' byte layout) |
| --- | --- | --- |
| `sequential` (authors' split, ~capture-disjoint) | **0.387** (acc 0.623) | 0.564 (acc 0.702) |
| `random` (stratified shuffle, leaks) | 0.800 (acc 0.920) | 0.938 (acc 0.960) |

Run dirs: `runs/tfegnn`, `runs/tfegnn-refbytes`, `runs/tfegnn-random-tfegnn`,
`runs/tfegnn-random-tfegnn-refbytes`. The published number is only approached with **both** leaks
(random split + addresses) — same pattern as the other two replications.

---

## 4. Remaining work, suggested order

**A. Unblock (no modelling)**
1. Fix PyTorch for the RTX 5090 (§0), with the user's OK; confirm the suite is 509/509.
   **2026-09-14: build verified in isolation; blocked on (a) the user's OK for `.venv` and (b) a
   power cycle of the box (§0, second blocker).** The full suite on the new build reached 506
   passed before the GPU fell off the bus; the 3 failures are all the resulting sticky CUDA error in
   `test_train_image.py` (`runs/pytest-torch2.14cu130-aborted-by-xid79.log`). Re-run after the
   power cycle, alone on the GPU.
   ✅ **Done 2026-09-16:** applied to `.venv` with the user's OK; suite 509/509; Okonkwo seed 1
   reproduces on all three tasks (§0).
2. ✅ **Done 2026-09-14.** Docs brought up to date with §3: `MODEL-bytes.md` packet row;
   `MODEL-image.md` scorecard + seed sweep; `MODEL-graph.md` TFE-GNN status; `REPLICATION-TFEGNN.md`;
   README status / install note / layout; `PHASE-4.md` §8 GPU line.

**B. Finish the half-built members**
3. **Flow-stats:** ✅ `class_weight` re-test done 2026-09-17 (→ D5). CatBoost still to add
   (needs `pip install catboost` — ask first).
4. **Image:** ✅ seeds on tasks 2/6 and the MobileApp baseline run 2026-09-17 (`MODEL-image.md`).
   **Left:** pick the recommended construction (D1), register `flow_image_cnn` (needs per-experiment
   `flow_image` params — the harness has none today), FlowPic on MobileApp, latency column.
5. **Graph:** ✅ registered as `graph_gnn_baseline` and scored on Tor (grouped 5-fold) 2026-09-17.
   VPN split done later that day (authors' split 0.62 vs 0.95; grouped cannot score p2p/email).
   **Left:** non-VPN / non-Tor caches (≈ 30 h / 9 h GPU — ask); a CSTNET graph cache
   (`byte_prep.py` is ISCX-only today); seeds; then GraphSAGE/GIN as `graph_gnn`.
6. **Bytes:** ✅ `--no-pretrained` ablation done 2026-09-17. **Left:** optionally the flow
   `--seq-length 512` run; then the recommended model (after D1).

**C. Build what is missing**
7. **Sequence member** end to end: DF `seq_cnn_baseline` with a paper-faithful input, then
   dilated-residual `seq_cnn`, then the Mamba/S4 variant. Scorecard on `iscx_pooled` traffic_type +
   MobileApp activity.
8. Decide D3, then either move image/graph onto `training/deep.py` or record why not; add a
   `deep.py` unit test.
9. BURST-capable byte extraction (4.2) if the recommended bytes model trains on our own pcaps.
10. **`experts/`** once D2 is settled.

---

## 5. Decisions that belong to the user — ask, don't pick

- **D1 — recommended models still open.** `SEQUENCE-`, `BYTES-` and `GRAPH-SELECTION-BRIEF.md` are
  open discussions. Bytes candidates: YaTC, NetMamba, MambaNetBurst, TrafficFormer. Image has
  evidence (§3.2) but no recorded decision. Flow-stats is resolved (LightGBM).
- **D2 — five experts or three ensembles?** `PHASE-4.md` specifies five experts
  (`tls/vpn/tor/app/activity`); the later `ENSEMBLE-MAP.md` collapses Stage 4 into three
  (E1 traffic-type, E2 app-ID, E3 in-app activity) and says tunnel type needs no model.
- **D3 — one trainer or several.** The plan says every deep member shares `training/deep.py`; the
  image and graph members don't.
- **D5 — class-weight convention (new, 2026-09-17).** On the Phase-3 headline config, unweighted
  `flow_gbdt` beats `class_weight="balanced"` by +0.014 macro-F1 on all three seeds
  (`MODEL-flow-stats.md`). `PHASE-5.md` §1 says every member must share one convention, so the
  choice (keep balanced everywhere, drop it everywhere, or leave it to calibration) is one decision,
  not a per-member default flip. Deep members currently default to `class_weight=None`.
- **D4 — where scorecards live.** The Phase-4 exit criterion says scorecards are "committed under
  `runs/`", but `runs/` is gitignored.

---

## 6. Progress log

- **2026-09-17** — Step B, everything that needs no §5 decision (all uncommitted; suite **516/516**,
  no Xid all day under the logger, GPU jobs one at a time):
  - Image: 5-seed sweep on tasks 2/6 — FlowPic 3ch +0.045 on Tor traffic (all seeds), noise on
    non-VPN traffic; MobileApp activity baseline **0.165** macro-F1 (config's early stop removed —
    it fired at chance). `MODEL-image.md`.
  - Flow-stats: `class_weight` re-test on the Phase-3 config, unweighted **+0.014** on 3 seeds →
    new **D5** (cross-member convention). `MODEL-flow-stats.md`.
  - Graph: **`graph_gnn_baseline` registered** (`models/graph_gnn.py`, 7 tests,
    `scripts/train_graph_gnn.py`); ISCX-Tor grouped 5-fold **0.44 ± 0.14**, authors' split 0.53
    (vs 0.39 from the replication script, other seed) — one p2p capture is 36 % of Tor, so fold 0
    is degenerate and a val carve-out is unusable there. `REPLICATION-TFEGNN.md`.
  - Bytes: ET-BERT packet **no-pretrain 0.688 vs 0.908** pretrained, 1.4 h on the 5090.
    `MODEL-bytes.md`.
  - Graph, later: ISCX-VPN through the member — authors' split **0.62** vs 0.95 published;
    grouped 5-fold 0.55 over the six paper classes, with `p2p` (one capture) and `email` (two)
    unscoreable under any grouped protocol on VPN alone. `scripts/train_graph_gnn.py` now falls
    back to the cache's class set when it is not the taxonomy.
  - Harness gaps found: `ExperimentConfig` has no per-experiment feature params (blocks a FlowPic
    config through `scripts/train.py`); `group_split` always returns fold 0 of the
    `StratifiedGroupKFold` regardless of seed (fine on 235 captures, degenerate on 51).

- **2026-09-16** — Step A.1 done: `.venv` upgraded to `torch 2.14.0+cu130` (user approved),
  suite **509/509**, Okonkwo seed 1 tasks 1/4/5 reproduce `base_s1.json` (0.755 / 0.902 / 0.442
  vs 0.754 / 0.871 / 0.442). GPU held under a PCIe/power logger through a 6.5-min load test, the
  suite and the seed run; no Xid since 09-14. Hardware cause still open (§0). Nothing in §5 decided.
- **2026-09-14, later** — After a reboot: suite **509/509** on the cu130 build, alone on the GPU.
  The seed-1 re-run then lost the GPU again (third Xid 79, single process, ~65 s of load). GPU work
  is blocked on the hardware, not on torch. `.venv` still untouched.
- **2026-09-14** — Step A.1 verified, not applied: `torch 2.14.0+cu130` + `torchvision 0.29.0+cu130`
  in a throwaway venv run kernels on the 5090 and reproduce Okonkwo seed 1 (tasks 1, 4) within
  0.01 macro-F1. During the check the GPU fell off the PCIe bus (Xid 79) for the second time in
  four days, the first time at idle — recorded in §0 as a separate hardware blocker; the box needs a
  power cycle before any GPU work. `.venv` untouched. Step A.2 done (docs listed in §4). Nothing in
  §5 decided.
