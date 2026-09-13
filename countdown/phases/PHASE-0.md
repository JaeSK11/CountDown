# Phase 0 — Project Setup & Data Loaders

**Roadmap ref:** `../PLAN.md` → Phase 0.
**Goal:** stand up the repo + a uniform data layer so that **any dataset → `(X, y)`** — a labeled
feature table (flow-stats) and a sequence tensor (packet size/direction) — ready for the Phase 3
ML baseline. No crypto parsing, no models yet.

**One-line exit criterion:** `countdown.data.load("iscxvpn", target="traffic_type")` returns a
`FlowDataset` whose `.features()` gives an `(N, F)` float matrix and `.labels()` gives `(N,)`
integer labels, reproducible via a fixed config.

---

## 1. Scope

**In scope**
- Repo scaffold, environment, dependencies, config system, logging, tests.
- A shared **Flow record schema** and **label taxonomy**.
- A minimal **flow extractor** (5-tuple bidirectional split) for multi-flow pcaps.
- **Dataset loaders** for the 5 core datasets (ISCXVPN, ISCXTor, CSTNET, PostQuantumTLS, MobileApp).
- Two **feature extractors**: `flow_stats` (tabular) and `packet_seq` (sequence).
- A caching layer (parsed flows/features cached to disk to avoid re-parsing 90 GB of pcaps).

**Out of scope (later phases)**
- TLS/QUIC handshake parsing + KEM extraction → Phase 2 (schema reserves slots for it).
- Any ML model → Phase 3+.
- Byte/image/graph feature extractors → Phase 4 (only flow_stats + packet_seq now).
- Live capture → Phase 6.
- WF datasets (WF-DF/Wang/ARES) — deferred; different task (website fingerprinting), and
  WF-DF/ARES aren't downloaded yet.

---

## 2. Dependencies & environment

- **Python** 3.10+, managed via `pyproject.toml` (or `requirements.txt` + venv).
- Core: `dpkt` (fast pcap parse), `scapy` (fallback/live later), `numpy`, `pandas`,
  `scikit-learn`, `lightgbm`, `pyyaml`, `tqdm`, `pyarrow` (parquet cache).
- Deferred (not installed in P0): `torch`, `dgl`/`torch-geometric`, `pyshark`/`tshark` (P2/P4).
- Deterministic: pin versions; global `SEED` in config.

**DoD:** `pip install -e .` succeeds; `python -c "import countdown"` works.

---

## 3. Directory layout to create

```
countdown/
├── pyproject.toml
├── README.md                     # architecture (from PLAN.md)
├── configs/
│   ├── datasets.yaml             # per-dataset paths + label-mapping rules
│   └── features.yaml             # flow_stats / packet_seq params (N packets, MTU cap, ...)
├── countdown/
│   ├── __init__.py
│   ├── config.py                 # load yaml, SEED, paths
│   ├── schema.py                 # Flow, Packet, FlowDataset dataclasses + label taxonomy
│   ├── flows/
│   │   └── extract.py            # pcap → flows (5-tuple bidirectional split)  [shared w/ P1]
│   ├── data/
│   │   ├── base.py               # DatasetLoader ABC + registry + cache
│   │   ├── iscxvpn.py
│   │   ├── iscxtor.py
│   │   ├── cstnet.py
│   │   ├── postquantumtls.py
│   │   └── mobileapp.py
│   ├── features/
│   │   ├── base.py               # FeatureExtractor ABC
│   │   ├── flow_stats.py
│   │   └── packet_seq.py
│   └── cache/                    # parquet/npy caches (gitignored)
├── tests/
│   ├── test_flow_extract.py
│   ├── test_loaders.py
│   └── test_features.py
└── data -> ../data               # symlink to existing datasets
```

---

## 4. Common schema (`schema.py`)

```
Packet:   ts: float, size: int, direction: int (+1 client→server, -1 server→client)
Flow:     flow_id, five_tuple(src_ip,src_port,dst_ip,dst_port,l4_proto),
          packets: list[Packet], start_ts, end_ts,
          dataset: str, label: str, label_fields: dict, meta: dict,
          # reserved for Phase 2 (left None in P0):
          tls_version, cipher_suite, kem_group, pqc_verdict
FlowDataset: flows/records + .features(extractor) -> (N,F) ndarray,
             .labels(target) -> (N,) int, .label_names, .split(...)
```

**Label taxonomy (unified `traffic_type`):**
`browsing, email, chat, audio_streaming, video_streaming, file_transfer, voip, p2p`
Orthogonal boolean fields kept in `label_fields`: `is_vpn` (ISCXVPN), `is_tor` (ISCXTor).
Other targets selected at load time: `app` (CSTNET/PostQuantumTLS), `activity` (MobileApp),
`tunnel_type` ∈ {none, vpn, tor}.

**DoD:** dataclasses + a `TRAFFIC_TYPES` enum; `target=` selects which label field becomes `y`.

---

## 5. Flow extractor (`flows/extract.py`)

- Input: a pcap path. Output: `list[Flow]` via 5-tuple **bidirectional** grouping
  (canonicalize direction by first-seen endpoint = client).
- Params (configurable): `flow_timeout` (e.g. 64 s idle split), `min_packets` (drop <4-pkt flows),
  `max_flows_per_pcap` (guard). Record per packet: ts, size (capped at MTU 1500 for features),
  direction.
- Fast path: `dpkt` for Eth/IP/TCP/UDP; handle VLAN, IPv6, and Linux cooked (SLL) if present.
- **Shared with Phase 1** (which adds the streaming `Source` wrapper + handshake bytes). Build the
  core here; refactor interface in P1.

**DoD:** on a known ISCXVPN pcap, produces a plausible flow count; unit test asserts direction
signs and 5-tuple grouping on a tiny synthetic pcap.

---

## 6. Per-dataset loader specs

All loaders subclass `DatasetLoader`, register by name, and **cache** parsed flows + features to
`cache/<dataset>/…parquet|npy` keyed by config hash.

### 6.1 `iscxvpn` — ISCXVPN2016  (multi-flow pcaps, label in filename)
- Path: `data/ISCXVPN2016/**/*.pcap` (102 files). Each file → many flows (5-tuple split).
- **Label from filename stem** (examples seen: `aim_chat_3a`, `email1a`, `facebook_audio1a`,
  `facebook_video1a`, `ftps_down_1a`, `hangouts_audio2a`, `netflix1`, `skype_file`, `vpn_*`):
  | pattern → traffic_type | examples |
  | --- | --- |
  | chat | aim_chat, icq_chat, *_chat, skype_chat |
  | email | email* |
  | voip | *_audio (facebook/hangouts/skype/voipbuster), voip* |
  | audio_streaming | spotify* |
  | video_streaming | *_video, netflix, youtube, vimeo, hangouts_video |
  | file_transfer | ftps*, sftp*, scp*, skype_file, *_transfer |
  | browsing | browsing* |
  | p2p | torrent*, bittorrent* |
  - `is_vpn = stem.startswith("vpn_")`. `tunnel_type = vpn if is_vpn else none`.
- **Task-0 subtask:** auto-list all 102 unique stems, finalize the regex map in
  `configs/datasets.yaml`, and log any **unmapped** stem (fail loudly, don't silently drop).

### 6.2 `iscxtor` — ISCXTor2016  (multi-flow pcaps; Tor/ vs NonTor/ dirs)
- Path: `data/ISCXTor2016/{Tor,NonTor}/**/*.pcap` (50 Tor + 44 NonTor). 5-tuple split.
- `is_tor = (dir == "Tor")`; `tunnel_type = tor if is_tor else none`.
- Label from filename (mixed case, normalize lower): `browsing*`, `AUDIO_*`→audio_streaming/voip,
  `*Chat*`→chat, `VIDEO_*`→video_streaming, `MAIL/EMAIL`→email, `FILE/FTP`→file_transfer,
  `VOIP/voip`→voip, `p2p/torrent`→p2p. Same "log unmapped stems" rule.

### 6.3 `cstnet` — CSTNET-TLS1.3  (one pcap ≈ one session; app = dir)
- Raw path: `data/CSTNET-TLS1.3/extracted/cstnet-tls 1.3/<domain>/*.pcap`
  (121 app dirs, ~491 pcaps each, 46,372 total). `label = app = <domain>` (e.g. `alipay.com`).
- **Two ingest modes** (config flag):
  1. `raw`: each pcap → 1 flow (already a session) → flow_stats/packet_seq. Good for cross-dataset
     consistency + later handshake/PQC parsing (Phase 2).
  2. `etbert`: reuse the preprocessed `extracted/flow_500` + `packet_5000` (ET-BERT format) for the
     byte/app-ID model in Phase 4 — no re-parsing.
- P0 implements `raw`; note `etbert` mode as a Phase-4 hook.
- 121 classes → allow `min_samples_per_class` filtering + `top_k_apps` cap in config.

### 6.4 `postquantumtls` — PostQuantumTLS  (one pcap per app; PQC validation set)
- Path: `data/PostQuantumTLS/pcaps/<package>.pcap` (90). `label = package` (Android app id).
- In P0: load as flows for completeness. **Primary use is Phase 2** (handshake/KEM ground truth) —
  P0 just needs it to load; feature/label plumbing minimal.

### 6.5 `mobileapp` — MobileAppActivity-Sensors2022  (frame-level CSVs, not pcap)
- Path: `data/MobileAppActivity-Sensors2022/D-OutCsv/<App>/<Activity>/*.csv` (368 files).
- CSV cols: `frame.number, frame.time_relative, wlan.sa, wlan.da, wlan.ta, wlan.ra,
  frame.time_delta_displayed, frame.len`. Build `Packet` list directly from rows
  (size=`frame.len` capped MTU; direction from `wlan.sa/da` vs the local client MAC; ts=`frame.time_relative`).
- Targets: `app` (8 classes) or `activity` (92 classes, label = `<App>/<Activity>`).
- Optional windowing (0.5/0.2 s, per the Sensors paper) deferred; P0 = one CSV → one sample.

---

## 7. Feature extractors

### 7.1 `flow_stats.py` (tabular → LightGBM in P3)
Fixed-length numeric vector per flow (documented, CICFlowMeter-inspired subset):
- Counts: total/fwd/bwd packet counts, total/fwd/bwd bytes.
- Packet size: min/max/mean/std overall + per direction; fwd/bwd byte ratio.
- IAT: mean/std/min/max of inter-arrival times overall + per direction.
- Flow: duration, packets/sec, bytes/sec, down/up ratio.
- First-N packet sizes (signed) as N extra columns (N from config, default 20).
- NaN/inf-safe (fillna 0). Emit a stable, versioned feature-name list.

### 7.2 `packet_seq.py` (sequence → CNN/SSM in P4)
- First `N` packets (default 32; config): array of **signed sizes** (`direction*min(size,1500)`)
  and array of IATs; zero-pad/truncate to N. Output `(N,)` or `(N,2)` tensors + a mask.

**DoD:** both extractors run on a `FlowDataset` and return correctly-shaped arrays; unit tests on
synthetic flows check determinism, padding, and NaN-safety.

---

## 8. Config system

- `configs/datasets.yaml`: per-dataset root path, glob, label-mapping rules (regex→traffic_type),
  ingest mode, filters (`min_packets`, `top_k_apps`, `min_samples_per_class`).
- `configs/features.yaml`: `flow_stats` params, `packet_seq` N, MTU cap, flow_timeout, SEED.
- One `Config` object; everything downstream reads from it (reproducibility).

---

## 9. Work breakdown (ordered) & Definition of Done

| # | Task | DoD |
| --- | --- | --- |
| 0.1 | Repo scaffold + `pyproject.toml` + `data` symlink + logging | `pip install -e .`; `import countdown` |
| 0.2 | `schema.py` (Flow/Packet/FlowDataset + taxonomy) | dataclasses + enum; typed |
| 0.3 | `flows/extract.py` (5-tuple split) | passes synthetic-pcap unit test |
| 0.4 | Config system + `datasets.yaml`/`features.yaml` | loaders read all paths/params from config |
| 0.5 | `iscxvpn` loader + finalize label map (log unmapped) | (X,y) for traffic_type; 0 unmapped stems |
| 0.6 | `iscxtor` loader | (X,y) for traffic_type + is_tor |
| 0.7 | `cstnet` loader (raw mode) | (X,y) for app; top_k filter works |
| 0.8 | `mobileapp` loader (CSV) | (X,y) for app/activity |
| 0.9 | `postquantumtls` loader (minimal) | flows load; ready for P2 |
| 0.10 | `flow_stats` + `packet_seq` extractors | shaped arrays; unit tests pass |
| 0.11 | Disk cache (parquet/npy, config-hash keyed) | 2nd load reads cache, no re-parse |
| 0.12 | `scripts/summarize_datasets.py` | prints per-dataset N flows, class counts, feature dims |

---

## 10. Acceptance test (end of Phase 0)

```
from countdown.data import load
ds = load("iscxvpn", target="traffic_type")     # parses (cached) → FlowDataset
X = ds.features("flow_stats")                    # (N, F) float32
y = ds.labels()                                  # (N,) int
assert X.shape[0] == y.shape[0] and X.shape[1] > 20
print(ds.class_counts())                         # dict per traffic_type
```
Repeat for `iscxtor` (traffic_type), `cstnet` (app), `mobileapp` (activity). `summarize_datasets.py`
prints a clean table for all five. **Then Phase 3 can train LightGBM immediately.**

---

## 11. Risks & open questions
- **Label-map completeness (ISCXVPN/Tor):** filename conventions are inconsistent → 0.5/0.6 must
  enumerate all stems and fail on unmapped ones (no silent drops).
- **Flow direction for MobileApp:** need the local client MAC per capture; infer from most-frequent
  `wlan.sa`/`ta` or from the Sensors metadata.
- **CSTNET scale (46k pcaps):** parsing all is heavy → cache aggressively; support `top_k_apps` /
  sampling for fast iteration.
- **pcap edge cases:** VLAN tags, IPv6, SLL headers, truncated packets → `dpkt` guards + skip-count.
- **Flow vs pcap granularity mismatch** across datasets is handled by the loader (split vs 1:1),
  but document per-dataset "what is one sample" clearly.

## 12. Not doing yet (explicit)
TLS/KEM parsing (P2), models (P3+), byte/image/graph features (P4), live capture (P6), WF datasets.
