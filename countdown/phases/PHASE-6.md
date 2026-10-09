# Phase 6 — End-to-End Pipeline, Eval Framework & Live Capture

**Roadmap ref:** `../PLAN.md` → Phase 6. **Depends on:** all prior phases.
**Goal:** assemble Stages 1→4 into one runnable pipeline that realizes the thesis —
**pull packets → detect crypto → is it quantum-resistant? → if not, classify the encrypted traffic
type** — plus a reproducible **research eval framework** and **live capture**.

**One-line exit criterion:** `countdown analyze <pcap>` emits per-flow verdicts
`{five_tuple, l7, tls_version, cipher, kem_group, pqc_verdict, traffic_class?, confidence}`, and
`countdown experiments run --all` regenerates every results table (PQC-triage %, per-target
macro-F1, ensemble ablation, cross-dataset) deterministically.

---

## 1. Scope
**In scope**
- `pipeline/pipeline.py` — wire Source → Reassembler (P1) → crypto/PQC verdict (P2) → **branch**:
  PQC/hybrid → mark quantum-safe, stop; classical/not-observable → **ensemble classify** (P5).
- `cli.py` — `countdown {analyze, live, train, eval, experiments}` console entry.
- `sources/live.py` — implement the P1 `LiveSource` stub (real-time capture).
- `eval/experiments.py` + `experiments/` configs — cross-dataset runs, results tables, **PQC-triage
  statistics** (the headline research output), reproducible + seeded.
- `eval/report_paper.py` — emit LaTeX/markdown tables + figures.
- Packaging: quickstart README, model zoo (trained artifacts), example pcaps.

**Out of scope (future work)**
- Active defenses / mitigation, cert-chain PQC-signature analysis at scale, distributed/streaming
  deployment, ECH inner recovery — listed under §7.

---

## 2. Dependencies
Runtime: prior phases only. Live: `scapy`/`pyshark` (or AF_PACKET) — needs `CAP_NET_RAW`/root.
Reporting: optional `matplotlib`/`jinja2` for tables/figures.

---

## 3. Directory additions
```
countdown/
├── pipeline/
│   └── pipeline.py         # Stages 1→4 orchestration + branch logic
├── sources/live.py         # real-time capture (implements P1 interface)
├── eval/
│   ├── experiments.py      # cross-dataset, PQC-triage stats, ablation runner
│   └── report_paper.py     # LaTeX/markdown tables + figures
├── cli.py                  # countdown {analyze,live,train,eval,experiments}
experiments/                # experiment configs (seeded, reproducible)
results/                    # generated tables/figures (gitignored)
examples/                   # sample pcaps + expected output
```

---

## 4. Components

### 4.1 End-to-end pipeline (`pipeline/pipeline.py`)
Per flow, in order:
1. **Stage 1** reassemble → flow + handshake bytes + `l7_hint`.
2. **Stage 2–3** parse handshake → `{tls_version, cipher, kem_group}` → **PQC verdict**.
3. **Branch (the thesis):**
   - `pqc` / `hybrid` → emit verdict `quantum_safe`, **skip classification** (optional: still tag
     traffic_class if configured).
   - `classical` / `not_observable` → **Stage 4 ensemble** → `traffic_class`/`app` + confidence.
4. Emit a **verdict record** (JSON/CSV row).
- Modes: `full` (all ensemble members) and `fast` (GBDT+CNN, soft-vote) for real-time.
- **DoD:** a pcap yields verdict records; branch behaviour matches config.

### 4.2 CLI (`cli.py`)
- `countdown analyze <pcap> [--out csv|json] [--mode full|fast]`
- `countdown live <iface> [--bpf ...] [--mode fast]`
- `countdown train|eval` — unify P3–P5 training/eval behind one entry.
- `countdown experiments run [--all|--name X]` — regenerate results.
- **DoD:** `countdown analyze` on an example pcap prints/writes verdicts.

### 4.3 Live capture (`sources/live.py`)
- Implements the P1 `Source` interface for a live NIC (scapy/pyshark/AF_PACKET); streaming reassembly
  with real-time (wall-clock) flow expiry + bounded memory + backpressure.
- Requires `CAP_NET_RAW`; document setcap/sudo. `fast` mode default for real-time latency.
- **DoD:** classifies live TLS flows on a loopback/replayed capture; graceful on partial flows.

### 4.4 Research eval framework (`eval/experiments.py`)
- **PQC-triage statistics** (headline): over a dataset, the % `pqc`/`hybrid`/`classical`/
  `not_observable`; and among **classical (vulnerable)** flows, the traffic-type/app breakdown — the
  core result ("what vulnerable traffic looks like").
- **Per-target results:** macro-F1 tables for traffic_type / app / activity / tunnel_type.
- **Ensemble ablation** (from P5) + **cross-dataset generalization** (train on X → test on Y).
- Seeded, config-driven; one command regenerates all tables → `results/`.
- **DoD:** `experiments run --all` reproduces identical tables across runs.

---

## 5. Work breakdown & Definition of Done
| # | Task | DoD |
| --- | --- | --- |
| 6.1 | `pipeline/pipeline.py` Stages 1→4 + branch | pcap → verdict records; branch correct |
| 6.2 | `cli.py analyze` | one-command pcap→verdicts (json/csv) |
| 6.3 | `sources/live.py` | real-time verdicts on replayed/loopback traffic |
| 6.4 | `cli.py live` (+ fast mode) | live capture classifies TLS flows |
| 6.5 | `eval/experiments.py` PQC-triage + per-target | tables generated to `results/` |
| 6.6 | cross-dataset + ablation experiments | generalization table produced |
| 6.7 | `eval/report_paper.py` | LaTeX/markdown tables + figures |
| 6.8 | packaging: README quickstart, model zoo, examples | fresh clone → `analyze` works |

---

## 6. Acceptance test (end of Phase 6)
```
# 1) end-to-end analyze
$ countdown analyze examples/cstnet_sample.pcap --out verdicts.csv
#   rows: five_tuple,l7,tls_version,cipher,kem_group,pqc_verdict,traffic_class,confidence

# 2) the thesis in action
#   - a hybrid/PQC flow  -> pqc_verdict=hybrid,   traffic_class=(skipped/optional)
#   - a classical flow   -> pqc_verdict=classical, traffic_class=streaming (conf 0.8x)
#   - a Tor flow         -> pqc_verdict=not_observable, traffic_class=<ensemble>

# 3) reproducible research tables
$ countdown experiments run --all      # -> results/{pqc_triage,per_target,ablation,cross_dataset}.*

# 4) live
$ sudo countdown live eth0 --bpf "tcp port 443" --mode fast   # streams verdicts
```
**Outcome:** a working prototype **and** a reproducible eval framework — the full research artifact.

---

## 7. Future work (beyond Phase 6)
- PQC **signature**-chain analysis where observable (TLS 1.2 / CertificateRequest).
- Mitigation guidance per vulnerable flow (harvest-now-decrypt-later risk scoring).
- ECH-aware handling as adoption grows; QUIC v2+ coverage.
- Streaming/distributed deployment; on-NIC/eBPF prefiltering for line-rate.
- Concept-drift monitoring + periodic retraining of Stage-4 models.

## 8. Risks
- **Live perf/root:** capture + reassembly + inference at rate → `fast` mode, bounded buffers,
  optional BPF prefilter; document capabilities.
- **Cross-dataset domain shift:** expect macro-F1 drops train≠test; report honestly (it's a result).
- **Reproducibility drift:** pin seeds/versions; freeze configs with each results table.
- **Privacy:** verdict records may contain IPs/SNI → add a `--redact` option for shareable output.

---

## 9. Status (2026-10-09): 6.1 and 6.2 built

```bash
python scripts/export_models.py                       # Phase-5 runs -> models_store/E{1,2,3}.joblib
countdown analyze capture.pcap --out verdicts.csv
countdown analyze *.pcap --mode fast --redact             # GBDTs only, no IPs/SNI
countdown analyze phone.pcap --context mobile              # capture-level override
```
(`countdown` is the `[project.scripts]` entry point, registered in `.venv` with
`pip install -e . --no-deps`; `python -m countdown` is equivalent. In Docker:
`docker compose -f docker/compose.yaml run --rm job countdown ...`.)

**Pipeline** (`countdown/pipeline/pipeline.py`): Stage 1 reassembly → `crypto.enrich_flow` →
negotiated `hybrid`/`pqc` skipped as quantum-safe (`--classify-quantum-safe` tags them anyway) →
router → expert. Routing implements `PHASE-5.md` §10.5's recommendation: `--context` wins; else
the Stage-1 hint; a flow the hint sent to E2 whose calibrated E2 confidence is under E2's exported
threshold is re-classified by E1, with E2's guess kept in `fallback_from`. Every record carries
`route_reason` (`hint` / `fallback` / `context` / `skipped:quantum_safe`).

**Record:** `flow_id, five_tuple, proto, start, packets, l7, tls_version, cipher, kem_group, sni,
pqc_verdict, pqc_basis, pqc_confidence, quantum_safe, at_risk, expert, route_reason, target,
traffic_class, confidence, fallback_from` (+ `members` with `--members`). Output CSV / JSON /
JSON lines; a summary goes to stderr. `--redact` drops IPs, SNI and file paths (§8 privacy risk).

**Deployed experts** (`pipeline/artifacts.py`, `models_store/`, gitignored): exported from the
Phase-5 runs without retraining — members per `Expert.deploy` (E1 GBDT; E2 GBDT + GIN; E3 all
three), temperature calibrators, the calibrated soft vote, and each member's exact input recipe.
Export reproduces the Phase-5 test scores exactly (E1 0.6654, E2 0.890, E3 0.5134). They are the
**refit-on-train** models — trained on the outer train split, so those test scores stay valid
for them; a refit on all labelled data is still to do.

Fallback thresholds (90 % of each expert's own out-of-fold flows keep their route):

| expert | full | fast (GBDT only) |
| --- | --- | --- |
| E2 | 0.446 | 0.390 |

**Modes.** `full`: all deployed members; E2's GIN gets byte matrices from a second
`byte_prep` pass over the pcap, joined to reassembled flows on the canonical 5-tuple (a
5-tuple split by the idle timeout gives its bytes to its first flow; the GIN abstains on the
rest). `fast`: each expert's GBDT only — no GPU, no graph building — with the GBDT-only
threshold. On a 3-pcap sample, `fast` left 2 ISCX chat flows in E2 (labelled as websites) that
`full` sent back to E1: the GBDT alone separates out-of-distribution TLS less well.

**Train/serve skew: none.** `scripts/check_pipeline_skew.py` runs the real pipeline on 60 random
E2 test-fold pcaps and compares each member with the prediction the same model made in the
Phase-5 run: top label 60/60 for both members, median confidence difference 0.0000, on the host
and in Docker (`results/phase6/`). E2 accuracy on those 60 without fallback: 0.933.

**Found on the way:** CSTNET's 46,372 flows share only **601** distinct `flow_id`s (the id is
per-pcap, `"<n>#0"`), so anything keyed on `flow_id` across CSTNET silently conflates flows.
Phase-5 code maps by row and is unaffected; the first skew check, keyed on `flow_id`, was not
(it reported 1/60 until fixed).

Tests: `tests/test_pipeline.py` — quantum-safe skip, hint routing, fallback with E2's guess
kept, context override never second-guessed, redaction, artifact round trip, CLI CSV/JSONL, the
5-tuple byte join.

**Next:** 6.7 tables, 6.3/6.4 live capture, 6.8 packaging (refit-on-all model zoo, example pcaps from `data/_pqc_synth/` — ours to ship).

---

## 10. 6.5 PQC-triage statistics (2026-10-09)

```bash
python scripts/run_triage.py                 # research tables -> runs/triage/<id>/ (results/phase6/)
countdown triage capture.pcap [...]          # an operator's capture, from predicted classes
```

`eval/triage.py` puts every flow in one bucket by its verdict **and** the strength of the
evidence, and counts flows and bytes (harvest-now-decrypt-later is about what can be stored):

| bucket | meaning |
| --- | --- |
| `quantum_safe` | negotiated hybrid / PQC |
| `at_risk` | negotiated classical |
| `at_risk_ceiling` | only the ClientHello seen, and **no** post-quantum group even listed in `supported_groups` — no server choice can make it safe |
| `pqc_offered_undecided` | a post-quantum group offered or listed (a HelloRetryRequest could still pick it), outcome unseen |
| `not_observable` | tunnel or no handshake — a fact about the capture, never counted as exposed |
| `unknown` | a handshake that would not parse |

**Exposed** = `at_risk` + `at_risk_ceiling`. The exposed flows are then broken down by class, and
by readiness tier (`configs/readiness_tiers.yaml`) for traffic types.

### Results (ground-truth labels; `results/phase6/triage/`)

| dataset | flows | handshake observable | **exposed** flows / bytes | quantum-safe | TLS versions seen |
| --- | --- | --- | --- | --- | --- |
| CSTNET-TLS1.3 (2021 crawl) | 46,372 | 99.8 % | **99.8 % / 99.7 %** | 0 | 1.3 only |
| PostQuantumTLS (Android apps, 2023) | 15,759 | 82.6 % | **82.6 % / 86.4 %** | 0 | 1.3 (81 %), 1.2 |
| ISCX-VPN (2016) | 16,094 | 15.6 % | **15.6 % / 38.4 %** | 0 | 1.2, 1.0, 1.1 |
| ISCX-Tor (2016) | 54,429 | 10.7 % | **10.7 % / 9.9 %** | 0 | 1.2, 1.1, 1.0 |
| `_pqc_synth` (ours, via the pipeline) | 3 | 100 % | 2 of 3 | **1** (X25519MLKEM768) | 1.3 |

1. **No public dataset contains a single post-quantum handshake.** That includes
   PostQuantumTLS: its 90 captures (the complete Zenodo archive, re-parsed with the Phase-1
   reassembler on 2026-10-09, flows identical to the old cache) offer only classical groups —
   checked on raw codepoints, and the registry knows the draft Kyber codepoints too. Every
   observable flow in the corpus is exposed. The only PQC positives are our three synthetic
   captures, which behave correctly (hybrid → quantum-safe; the downgrade capture, where the
   client offered ML-KEM and the server chose x25519, → `at_risk`).
2. **Flows and bytes disagree, and bytes are what an adversary stores.** ISCX-VPN:
   `file_transfer` is 17 % of exposed flows but **89 % of exposed bytes**; by tier,
   `migrate_first` is 30 % of exposed flows and **89 % of exposed bytes**. ISCX-Tor: browsing is
   81 % of exposed flows, video 76 % of exposed bytes. Any priority list built on flow counts
   alone would rank the wrong traffic first.
3. **Tunnels hide most of ISCX** (84–89 % not observable): the outer VPN/Tor handshake is not
   the inner one. Those flows are reported, not guessed.
4. CSTNET's exposed groups: x25519 93 %, secp256r1 6 %, curveSM2 279 flows.

### Fidelity of the predicted breakdown

`countdown triage` reports *predicted* classes, so the mix it prints has an error. Measured on the
exposed flows of each expert's test fold (deployed models, Phase-5 cached test predictions):

| expert | exposed test flows | accuracy | class-mix total variation | worst class | tier-mix TV |
| --- | --- | --- | --- | --- | --- |
| E1 traffic type | 1,468 | 0.706 | **0.127** | `file_transfer` under-counted by 8.0 pts | 0.109 (`migrate_next` +10.9 pts) |
| E2 app | 9,316 | 0.915 | **0.028** | `nvidia.com` −0.3 pts | — |

E2's website mix is trustworthy. E1's is not yet: it under-counts `file_transfer`, a
`migrate_first` class, and over-counts `migrate_next` by 11 points — the error runs against the
traffic that matters most. `countdown triage` prints these error figures under its tables.

### What 6.5 changes next

- Real PQC positives are the gap. A capture of current browser traffic (Chrome and Firefox
  negotiate X25519MLKEM768 with large CDNs by default) would give the first non-synthetic
  quantum-safe flows — it has to be traffic we may capture.
- E1's `file_transfer` recall is the model work with the most triage value.

---

## 11. 6.6 Cross-dataset generalisation + the experiments runner (2026-10-09)

```bash
countdown experiments list
countdown experiments run --all          # -> results/experiments/{triage,cross_dataset,ablation}.{json,md}
```

`eval/experiments.py` regenerates every results table. Outputs carry no timestamps or run paths,
and **re-running reproduces them byte for byte** — checked on all three (`ablation` in
`tests/test_experiments.py`). `ablation` tabulates the finished Phase-5 runs rather than
re-training hours of GPU members; `triage` and `cross_dataset` recompute from the data (~3 min
for all three).

### Cross-dataset results (`results/experiments/cross_dataset.md`)

E1's deployed member (`flow_gbdt` on `flow_stats`), traffic type, seed 42, a fixed 600 rounds on
every side (early stopping on a grouped 10 % carve-out stopped at round 1 on ISCX-VPN and left a
near-constant model — in-domain 0.045 — so it is off for both columns). Macro-F1 over the classes
both sides contain; in-domain = the training side's own grouped 80/20 split on the same classes.

| shift | train → test | classes | in-domain | **cross** | drop |
| --- | --- | --- | --- | --- | --- |
| corpus | ISCX-VPN → ISCX-Tor | 7 | 0.576 | **0.102** | −0.47 |
| corpus | ISCX-Tor → ISCX-VPN | 7 | 0.578 | **0.113** | −0.47 |
| tunnel | ISCX-VPN plain → VPN-tunnelled | 6 | 0.598 | **0.335** | −0.26 |
| tunnel | ISCX-VPN tunnelled → plain | 6 | 0.681 | **0.182** | −0.50 |
| tunnel | ISCX-Tor plain → Tor-tunnelled | 8 | 0.777 | **0.138** | −0.64 |
| tunnel | ISCX-Tor tunnelled → plain | 8 | 0.601 | **0.078** | −0.52 |

**The traffic-type model does not transfer.** Across the two ISCX capture campaigns it falls from
~0.58 to ~0.11 — near the majority-class floor — on the same 7 classes. Plain → VPN-tunnelled
keeps the most (0.34; `email` 0.69, `voip` 0.52 survive the tunnel), plain → Tor collapses
(0.14). What the in-domain scores of Phases 3–5 measure is largely *this capture setup*, not
*this kind of traffic*.

Consequences:
- Every E1 number in this repo, and the E1 breakdown `countdown triage` prints, holds for traffic
  that resembles the training captures. On a new network, expect the cross column, not the
  in-domain one, until the model is retrained on local labelled data.
- This is the strongest argument yet for the deployment decision in
  `DEPLOYMENT-AND-PRIMARY-DATASET.md` (train on what the sensor will see) and for routing on a
  confidence fallback rather than trusting any one expert blindly.
- Not yet measured: the same question for E2 (app-ID). It needs a second labelled website corpus;
  CSTNET is the only one we have.

Caveats: one seed; GBDT on flow statistics only; the tunnelled sides are small (697 Tor flows,
2,365 VPN flows).

---

## 12. 6.7 Paper tables and figures (2026-10-09)

`countdown experiments run paper` (`eval/report_paper.py`) turns `results/experiments/*.json`
into `results/paper/tables/{triage,cross_dataset,ablation,fidelity}.tex` (booktabs, scaled to
the column; need `\usepackage{booktabs,graphicx}`) and
`results/paper/figures/{exposure,cross_dataset,ablation}.{pdf,png}`. They read only the
experiment JSONs, so the paper's numbers are the regenerated tables'. Byte-for-byte
reproducible (PDF/PNG dates stripped). All four tables and three figures compile with
`pdflatex` (TeX Live container) with no overfull boxes.

Figures follow the data-viz method: verdict buckets are states, so they take the status palette
(validated; *warning* and *serious* kept non-adjacent — ΔE 13.6 < 15 otherwise), with a legend,
direct labels and a hatch on the ClientHello-only bucket so nothing is told apart by colour alone;
comparisons take categorical slots 1–2. Two corrections after looking at the renders: an
invisible bucket (10 unparsed CSTNET flows) is dropped from the legend, and the ablation labels
use three decimals — two showed E1's tie (0.6654 vs 0.6648) as 0.67 vs 0.66.

## 13. 6.3 / 6.4 Live capture (2026-10-09)

```bash
docker compose -f docker/compose.yaml run --rm job countdown live eth0 --port 443 --duration 30
```

- `sources/live.py`: a Linux `AF_PACKET` socket (no libpcap, no new dependency), Ethernet frames,
  loopback de-duplicated, port filter in Python (kernel BPF needs libpcap: refused with a clear
  message), optional promiscuous mode, `--duration` / `--count` limits.
- **Idle wire:** the reassembler expires flows on the packet clock, so a quiet link would hold
  its last flows forever. The source yields an empty-frame **heartbeat** every `timeout_ms`
  without traffic and the reassembler runs its expiry sweep on it (`test_live.py`).
- `Pipeline.analyze_stream`: closed flows are batched and classified at most every
  `--batch-seconds`, the flush checked on every frame and heartbeat; JSON lines out. In `full`
  mode a streaming `ByteCollector` (`features/byte_prep.py`) keeps each 5-tuple's first 50
  payload packets for the graph member — byte-identical to the file extractor (tested).
  Streaming a replayed pcap gives exactly the records `analyze` gives (tested).
- **Privilege without touching the host:** the container runs as uid 1000, which does not
  inherit container capabilities, so the image has a copy of the interpreter, `python-netraw`,
  with only `cap_net_raw+ep`; the `countdown` wrapper uses it for `live` alone. Works with
  Docker's default capability set. A `live` compose service with host networking exists for
  real deployments (it sees all host traffic — only where monitoring is allowed).

**Live demo — the first real quantum-safe flows.** In a container on Docker's bridge network (so
the capture sees only the container's own traffic), fetching four public home pages over
Python's HTTPS (OpenSSL 3.5.7, which offers X25519MLKEM768 first):

| server | TLS | key exchange | verdict |
| --- | --- | --- | --- |
| www.cloudflare.com | 1.3 | **X25519MLKEM768** | quantum-safe — classification skipped |
| www.google.com | 1.3 | **X25519MLKEM768** | quantum-safe |
| en.wikipedia.org | 1.3 | **X25519MLKEM768** | quantum-safe |
| github.com | 1.3 | x25519 | exposed → routed, E2 → E1 fallback |

Three of four major sites negotiate a hybrid post-quantum exchange with a current client — in
contrast to every public dataset (§10), whose captures predate deployment. This is the
non-synthetic PQC positive §10 asked for; a larger capture of permitted traffic would turn it
into a statistic.

## 14. 6.8 Packaging (2026-10-09)

**Fresh clone → `analyze`:**
```bash
git clone https://github.com/JaeSK11/CountDown && cd CountDown/countdown
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,deep]"   # or the Docker image
.venv/bin/countdown models fetch                                  # model zoo, verified
.venv/bin/countdown analyze examples/*.pcap --mode fast
```

- **Model zoo** (`pipeline/zoo.py`): the deployed experts (E1 6 MB, E2 331 MB, E3 13 MB,
  joblib compress=3) are assets of the private GitHub release `models-v1`; `countdown models
  fetch` downloads them with `gh` (uses its auth). They are pickles, so the repo tracks
  `models_store.sha256` and `load_store` **refuses to unpickle any file whose hash is not in
  it** (`--allow-unverified` for one's own export). The hash comes from git, not the server.
  `scripts/publish_models.sh` writes the manifest and uploads a release. The models are the
  Phase-5 refit-on-train members (§9), so their reported test scores hold for these files.
- **Examples** (`examples/`): our three synthetic handshake captures — the only pcaps ours to
  ship — with `expected_fast.jsonl`. `tests/test_examples.py` always checks the crypto verdicts
  and, when the zoo verifies, the full routed output.

## 15. Phase 6 status

| # | task | state |
| --- | --- | --- |
| 6.1 | pipeline | ✅ §9 |
| 6.2 | `countdown analyze` | ✅ §9 |
| 6.3 | live source | ✅ §13 |
| 6.4 | `countdown live` (+ fast mode) | ✅ §13 — live TLS flows classified, incl. real hybrid PQC |
| 6.5 | PQC-triage + per-target | ✅ §10 |
| 6.6 | cross-dataset + ablation + runner | ✅ §11 |
| 6.7 | paper tables + figures | ✅ §12 |
| 6.8 | packaging | ✅ §14 — fresh-clone check below |

**Exit criterion** — `countdown analyze <pcap>` emits per-flow verdicts with five-tuple, l7, TLS
version, cipher, KEM group, PQC verdict, traffic class and confidence ✅; `countdown experiments
run --all` regenerates every results table deterministically ✅.

Known limits carried out of Phase 6: E1 does not transfer across capture setups (§11); E1's
predicted breakdown under-counts `file_transfer` (§10); E3's ensemble gain rests on 92 test flows;
`byte_net` is not in E2; the deployed models are refit-on-train.
