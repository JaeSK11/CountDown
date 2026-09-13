# Phase 1 — Ingest & Flow Reassembly (Pipeline Stage 1)

**Roadmap ref:** `../PLAN.md` → Phase 1. **Depends on:** Phase 0 (`schema.py`, config, the batch
flow splitter).
**Goal:** a robust ingest layer — a pluggable **Source** (pcap now, live later) feeding a **stateful
flow reassembler** that emits `Flow` objects enriched with the metadata Phase 2 needs: the
**reassembled opening bytes of each direction** (the handshake window) and a **coarse L7 transport
hint** (tls / quic / tunnel / opaque). This is the runtime pipeline's Stage 1.

**One-line exit criterion:**
`list(Reassembler().run(PcapSource(pcap)))` yields `Flow` objects where, for a direct-TLS pcap
(CSTNET / PostQuantumTLS), `flow.handshake_server_bytes` starts with a TLS handshake record
(`0x16`) and `flow.l7_hint == "tls"`.

> **Deployment context (added after the passive-capture decision).** This layer runs on
> **passively captured real traffic**, not just clean dataset pcaps — so two things are
> **first-class, not edge cases** (both are already implemented in `flows/`):
> - **QUIC / UDP.** 2026 passive capture is QUIC-heavy. UDP flows are reassembled, QUIC long
>   headers detected (`quic_long_header`; v1/v2/draft/gQUIC/mvfst/GREASE), the Initial bytes
>   captured, and `l7_hint="quic"` set. Decrypting the QUIC Initial to reach the ClientHello is
>   the one remaining piece → **Phase 2**.
> - **Mid-stream / no-handshake flows.** Real capture is full of connections that predate the
>   capture window. `HeadBuffer.trusted_start` recognises a TLS/QUIC boundary without a SYN, and
>   `_handshake_incomplete` distinguishes "gap" from "no handshake present" and keys the flag on
>   the **server** window — so Phase 2 can tell "no PQC" from "could not see".

---

## 1. Why this is more than Phase 0's splitter

Phase 0 built a **batch, per-pcap, feature-oriented** 5-tuple split (enough to make `(X, y)`).
Phase 1 builds the **stateful, streaming, protocol-aware** version that:
- runs off a generic **Source** (so live capture drops in at Phase 6 with no rewrite),
- does **TCP head-of-stream reassembly** so a ClientHello/ServerHello spanning multiple segments is
  recovered intact (Phase 2 can't parse a split record),
- captures **handshake byte buffers** + a **coarse L7 hint**,
- manages a **flow table** with proper expiry (idle timeout, FIN/RST, max-duration).

**Refactor:** Phase 0's `flows/extract.py` is re-implemented as a thin batch wrapper over the new
reassembler (one code path). Phase 0's feature extractors must still produce identical `(X, y)`.

---

## 2. Scope

**In scope**
- `Source` interface + `PcapSource` + `DirectorySource` (iterate a dataset's pcaps). `LiveSource`
  interface defined but stubbed (impl in Phase 6).
- Link/IP/L4 decode (Eth, SLL/cooked, VLAN, IPv4/IPv6, TCP/UDP) → 5-tuple + payload.
- Stateful bidirectional **flow reassembler** with expiry; emits enriched `Flow` objects.
- **TCP head reassembly** of the first `handshake_window` bytes per direction (ordered, dedup
  retransmits, tolerate out-of-order & missing SYN).
- **Handshake capture** (raw opening bytes per direction) + **coarse L7 hint** (transport-level
  only — no TLS field parsing).
- Validation script + unit tests.

**Out of scope (later)**
- TLS/QUIC **field** parsing, cipher/KEM extraction, PQC verdict → **Phase 2**.
- QUIC Initial-packet **decryption** to reach the ClientHello → Phase 2 (P1 only detects QUIC +
  captures Initial bytes).
- Full bidirectional stream reassembly beyond the handshake window (not needed).
- ML / features beyond what Phase 0 already has.
- Live capture implementation → Phase 6.

---

## 3. Dependencies
No new hard deps beyond Phase 0 (`dpkt` for decode; TCP head-reassembly implemented in-house — it's
small since we only need the first few KB per direction). `scapy` remains the fallback/live path.

---

## 4. Directory additions

```
qsentinel/
├── sources/
│   ├── base.py          # Source ABC → yields RawPacket(ts, linktype, data)
│   ├── pcap.py          # PcapSource(path), DirectorySource(glob)
│   └── live.py          # LiveSource(iface) — interface only, NotImplemented in P1
├── flows/
│   ├── decode.py        # link/IP/L4 decode → (five_tuple, l4, payload, flags)
│   ├── reassembly.py    # Reassembler: flow table, TCP head-reassembly, expiry → Flow
│   ├── handshake.py     # capture opening bytes per dir + coarse L7 hint
│   └── extract.py       # (refactored) batch wrapper: pcap → list[Flow] via Reassembler
└── scripts/
    └── inspect_flows.py # pcap → table: 5-tuple, npkts, l7_hint, handshake?  (bytes)
```

---

## 5. Components

### 5.1 Source (`sources/`)
- `RawPacket(ts: float, linktype: int, data: bytes)`.
- `Source` ABC: `__iter__` yielding `RawPacket`; `close()`.
- `PcapSource(path)`: dpkt reader; expose `linktype` (Eth/SLL). `DirectorySource(glob, recursive)`:
  chain many pcaps, tagging each flow with its source file (for dataset labels).
- `LiveSource(iface, bpf=None)`: **interface only** in P1 (`raise NotImplementedError`), so Phase 6
  is a drop-in.

**DoD:** `PcapSource` iterates a dataset pcap; `DirectorySource` streams a whole dataset folder.

### 5.2 Decode (`flows/decode.py`)
- Parse link layer (Ethernet, Linux SLL/cooked, 802.1Q VLAN), IPv4/IPv6, TCP/UDP.
- Return `(five_tuple, l4_proto, payload, tcp_flags, seq, ack)` or `None` (non-IP / unparsable →
  counted, skipped). Robust to truncated captures.

**DoD:** unit test decodes Eth/IPv4/TCP, VLAN, and IPv6 sample frames; increments a skip counter on
garbage without crashing.

### 5.3 Reassembler (`flows/reassembly.py`)
- **Flow key:** canonical bidirectional 5-tuple; **client = endpoint of first packet** (or SYN
  sender if present).
- Per flow: `packets: list[Packet]` (ts, signed size, direction), byte/pkt counters, start/end ts,
  and a **head buffer per direction** (bounded to `handshake_window`, default 8 KB).
- **TCP head reassembly:** order early segments by seq into the per-direction head buffer; drop
  duplicates/retransmits; tolerate gaps (stop filling a buffer at the first unfilled gap). UDP: just
  append payloads until the window fills.
- **Expiry:** emit + evict a flow on FIN/RST (both dirs or timeout grace), `idle_timeout` (default
  64 s), or `max_duration`. Periodic sweep by packet clock (no wall-clock → deterministic on pcaps).
- Emit enriched `Flow` (schema from Phase 0) with: `client_ip`, `l7_hint`,
  `handshake_client_bytes`, `handshake_server_bytes`, `expiry_reason`.

**DoD:** on a CSTNET pcap, flow count is sane; a flow's head buffers are non-empty; retransmit
unit test yields no duplicate bytes.

### 5.4 Handshake capture + L7 hint (`flows/handshake.py`)
- From the reassembled head buffers, store the raw opening bytes (no field parsing here).
- **Coarse L7 hint** (cheap, transport-level only):
  - `tls`  — TCP payload starts with `0x16 0x03` (handshake, TLS record) — typically :443/:853.
  - `quic` — UDP long-header (first byte `0xC0`–`0xFF`) with a known QUIC version; capture Initial
    bytes (decode deferred to P2).
  - `tunnel_openvpn` / `tunnel_ike` — coarse port/first-byte heuristics (OpenVPN 1194, IKE 500/4500).
  - `tor` — hinted from dataset context (ISCXTor) or TLS-to-known-guard; kept coarse.
  - `tcp` / `udp` / `opaque` — otherwise.
- Hint is advisory; Phase 2 does authoritative parsing. Tunneled flows get flows + bytes but no PQC
  verdict (documented invariant).

**DoD:** on CSTNET/PostQuantumTLS, >90% of :443 TCP flows hint `tls` with a captured ServerHello
window; on ISCXVPN/Tor, flows are produced without errors (hint may be tunnel/opaque).

---

## 6. Flow metadata produced (fills Phase 0 schema)
`five_tuple, l4_proto, packets[], start/end_ts, client_ip, dataset, source_file, l7_hint,
handshake_client_bytes, handshake_server_bytes, expiry_reason`. Phase-2 fields
(`tls_version, cipher_suite, kem_group, pqc_verdict`) remain `None` here.

---

## 7. Work breakdown & Definition of Done

| # | Task | DoD |
| --- | --- | --- |
| 1.1 | `sources/base.py` + `RawPacket` | ABC + typed |
| 1.2 | `PcapSource` + `DirectorySource` | iterate one pcap / a dataset folder |
| 1.3 | `LiveSource` interface (stub) | importable, raises NotImplemented |
| 1.4 | `flows/decode.py` (Eth/SLL/VLAN/IPv4-6/TCP-UDP) | decode unit tests pass, skip-counter |
| 1.5 | `Reassembler` flow table + expiry | flows emitted; expiry reasons correct |
| 1.6 | TCP head reassembly (ordered, dedup, gap-safe) | retransmit/out-of-order unit tests |
| 1.7 | `handshake.py` capture + L7 hint | TLS hint + ServerHello window on CSTNET |
| 1.8 | Refactor `flows/extract.py` onto Reassembler | Phase-0 `(X,y)` unchanged (regression test) |
| 1.9 | `scripts/inspect_flows.py` | prints per-flow table for any pcap |
| 1.10 | Perf pass (streaming, bounded memory) | runs a large ISCXVPN pcap w/ capped RAM |

---

## 8. Acceptance test (end of Phase 1)

```
from qsentinel.sources import PcapSource
from qsentinel.flows import Reassembler

# direct-TLS pcap → TLS handshake recovered
flows = list(Reassembler().run(PcapSource(CSTNET_PCAP)))
f = next(x for x in flows if x.l7_hint == "tls")
assert f.handshake_server_bytes[:1] == b"\x16"          # TLS handshake record
assert f.client_ip and len(f.packets) >= 4

# tunneled pcap → still parses, no crash, hint != tls-with-KEM
tor = list(Reassembler().run(PcapSource(ISCXTOR_PCAP)))
assert len(tor) > 0

# regression: Phase-0 features unchanged
from qsentinel.data import load
assert load("iscxvpn", target="traffic_type").features("flow_stats").shape[1] > 20
```
`inspect_flows.py <pcap>` prints a per-flow table (5-tuple, npkts, l7_hint, handshake captured?).
**Then Phase 2 can parse `handshake_*_bytes` into TLS version / cipher / KEM.**

---

## 9. Risks & open questions
- **Missing SYN / mid-stream captures:** infer client by first-packet direction; note lower
  confidence. Some dataset pcaps start mid-flow.
- **TCP reassembly corner cases:** retransmits, overlapping segments, out-of-order, gaps → only need
  the head window, so stop at first gap and mark `handshake_incomplete`.
- **QUIC:** Initial packets are decryptable with version-derived keys, but that's Phase 2. P1 only
  detects + captures Initial bytes.
- **Tunneled flows (VPN/Tor):** handshake capture gets the **outer** layer only; inner crypto is
  invisible → no PQC verdict (by design). Document per l7_hint.
- **Scale/memory:** 46k CSTNET pcaps + large ISCXVPN captures → stream, cap head buffers, expire
  flows promptly, cache emitted flows (reuse Phase 0 cache).
- **Determinism:** drive expiry by packet timestamps, not wall clock, so pcap runs are reproducible.

## 10. Not doing yet (explicit)
TLS/QUIC field parsing & KEM extraction (P2), QUIC Initial decryption (P2), models/features beyond
Phase 0 (P3+), live capture impl (P6).

---

## 11. As built — implementation notes & deviations

Phase 1 is implemented and validated. This section records where the build departed from
the plan above and why, so Phase 2 inherits the reasoning and not just the code.

### 11.1 Two reassembler modes (affects task 1.8)
`Reassembler(phase0_compat=True)` reproduces Phase 0's batch splitter exactly; the default
`Reassembler()` is the streaming mode described in §5.3.

The split is not cosmetic and could not be avoided. §5.3 requires FIN/RST expiry, but
closing a flow on teardown *moves flow boundaries* — a post-FIN straggler starts a new flow
instead of joining the old one — which changes the packet-size/timing arrays and therefore
`(X, y)`. That is the correct behaviour for a live sensor and the wrong behaviour for
reproducing a cached feature matrix, so both exist and the choice is explicit at the call
site. `FlowExtractor` (the loader path) uses compat mode; `inspect_flows.py`, the
acceptance test and Phase 6 use streaming.

**Task 1.8 is proven, not asserted.** `tests/golden/phase0_flows.json` was captured from
the pre-refactor code over 29 captures (9 synthetic corner cases + 20 sampled from all four
datasets, ~7,000 flows), pinning a SHA-1 over each flow's `timestamps`/`sizes`/`directions`
plus its `ParseStats`. `tests/test_phase0_regression.py` re-runs all 29: **29/29
byte-identical**.

### 11.2 Reassembly parameters live in `reassembly:`, not `flow:`
`config.flow` is hashed into `Config.flow_key`, so adding `handshake_window` there would
have invalidated all five dataset caches and forced a re-parse of the 90 GB corpus for no
change in `(X, y)`. The new knobs live in a separate `reassembly:` block in
`configs/features.yaml`, which is not hashed. Verified: all five cache keys unchanged.

**Phase 2 action required.** The parquet cache gained `l7_hint`,
`handshake_client_bytes`, `handshake_server_bytes`, `handshake_incomplete` and
`expiry_reason` columns, but caches written before Phase 1 do not have them and their key
is still valid. `DatasetLoader._read_cache` warns loudly when it loads such a file. Phase 2
needs **one `load(..., refresh=True)` rebuild per dataset** before the handshake bytes are
available through the cached loader path. Reading a pcap directly through `Reassembler`
always has them.

A second, subtler constraint: `data/base.py` sorts cached flows by `json.dumps(meta)`, so
*any* new `meta` key silently reorders every dataset's rows. Compat mode therefore emits no
extra meta keys (`emit_handshake_meta` is off there), and
`test_compat_mode_emits_no_extra_meta_keys` pins it.

### 11.3 `handshake_incomplete` is narrower than §9 implied
§9 proposed marking the flag on any gap or missing SYN. Measured against the corpus, that
rule was wrong in both directions, so the flag now means precisely *"Phase 2 cannot reach a
verdict from this window"*:

* **Missing SYN is not disqualifying.** **Every one of CSTNET's 46,372 captures begins
  mid-connection with the SYN stripped**, yet starts exactly on a TLS record boundary. The
  §9 rule would have marked the entire dataset unparseable and gutted Phase 2's classical-
  TLS validation set. A well-formed protocol header at offset 0 now proves stream start
  just as well as a SYN does (`HeadBuffer.trusted_start`).
* **The test is applied to the server window.** The PQC verdict is read from the
  ServerHello's `key_share`. A flow carrying only a ClientHello can say what was *offered*
  but never what was negotiated. CSTNET truncates exactly this way: the client window opens
  on a ChangeCipherSpec while the server window still carries the ServerHello.
* **A record that is merely a handshake message is not a handshake start.** A server window
  opening on a NewSessionTicket looks like `0x16 0x03 ..` but means the handshake finished
  before capture (`opens_handshake` requires ClientHello or ServerHello).
* **Only a gap that truncates the handshake counts.** Holes further into the 8 KB window —
  common on CSTNET, where the certificate chain following the ServerHello is partly missing
  — still leave a whole, parseable ServerHello behind.

`handshake_client_bytes` is still populated when the flag is set; the flag says a verdict is
not derivable, not that the bytes are worthless. Per-direction detail is in
`meta["handshake_client_complete"]` / `["handshake_server_complete"]` (streaming mode).

### 11.4 DoD 1.7 restated (`scripts/validate_phase1.py`)
The original metric — ">90% of :443 TCP flows hint `tls` with a captured ServerHello" —
conflates something Phase 1 controls with something it cannot. **~12.6% of PostQuantumTLS
:443 flows have no handshake anywhere in the capture file**, because the TCP connection
predates the capture start. No reassembler can recover bytes that were never written.

The validator therefore measures the two separately, and gates on both:

| measure | cstnet | postquantumtls | iscxvpn (`vpn_`) | iscxtor |
| --- | --- | --- | --- | --- |
| :443 flows hinting `tls` | 100.0% | 98.9% | 90.9% | n/a (all `tor`) |
| handshake present in capture | 99.5% | 87.4% | 74.5% | 0% |
| → **whole ServerHello reassembled** | **100.0%** | **100.0%** | **100.0%** | n/a |
| unobservable flows correctly flagged | 100% | 100% | 100% | 100% |
| recoverable flows wrongly flagged | 0% | 0% | 0% | 0% |

Every handshake that is present in a capture is reassembled into a whole ServerHello
record, with no false alarms in either direction. Tunneled corpora produce flows with no
errors and no PQC claim, as designed.

### 11.5 Performance (task 1.10)
`scripts/bench_reassembly.py`, measuring peak RSS in a subprocess per mode:

| capture | packets | flows | mode | peak RSS | throughput |
| --- | --- | --- | --- | --- | --- |
| `ftps_down_1a.pcap` (5.5 GB) | 5.74 M | 224 | streaming | 258 MB | 95 k pkt/s |
| `vpn_netflix_A.pcap` (806 MB) | 870 k | 132 | streaming | 115 MB | 171 k pkt/s |
| `browsing2.pcap` (24 MB) | 55 k | 1820 | streaming | +3 MB | 92 k pkt/s |
| `browsing2.pcap` (24 MB) | 55 k | 1820 | batch | +13 MB | 92 k pkt/s |

Streaming heap growth is 5.1x smaller than batch on a many-flow capture: memory scales with
*concurrently open* flows, not total flows. **Known characteristic:** it does still scale
with packets held in open flows (~38 B/packet), which is what the 5.5 GB run's 219 MB of
growth is — 5.7 M packets across only 224 long-lived flows. `max_duration` (default 3600 s
of packet clock) is the backstop for a pathologically long-lived flow.

A full flow table no longer aborts the read in streaming mode: the stalest flow is evicted
with `expiry_reason="table_full"`, because a live sensor must not stop capturing. Compat
mode keeps Phase 0's `break`.

### 11.6 Schema additions
`Flow` gained `l7_hint`, `handshake_client_bytes`, `handshake_server_bytes`,
`handshake_incomplete`, `expiry_reason`, and derived properties `client_ip`, `client_port`,
`server_ip`, `server_port`, `l4_proto`, `source_file`, `has_handshake`. `SCHEMA_VERSION` was
**not** bumped: the additions are observationally inert for Phase 0 features (pinned by
`test_handshake_capture_does_not_alter_the_feature_arrays`).

### 11.7 Test coverage
**69 → 181 tests**, all passing (`pytest -m "not slow"` runs 180 in ~7 s):

| file | n | covers |
| --- | --- | --- |
| `test_decode.py` | 10 | Eth/VLAN/IPv6/UDP decode, payload+seq, skip counters, pcapng sniffing |
| `test_sources.py` | 11 | PcapSource, DirectorySource chaining/tagging/corrupt-file skip, LiveSource stub |
| `test_handshake.py` | 31 | ordering, retransmits, overlap, gaps, seq wraparound, window cap, every L7 hint |
| `test_reassembly.py` | 24 | flow table, all five expiry reasons, packet-clock determinism, table-full eviction |
| `test_phase0_regression.py` | 32 | the 29-capture golden + meta/array inertness |
| `test_phase1_acceptance.py` | 4 | §8 verbatim (1 marked `slow`) |

### 11.8 Not addressed
`scapy` remains an unused fallback (§3) — the dpkt path handled every capture in the corpus,
so no fallback was wired in. Bringing it in is a Phase 6 decision, alongside `LiveSource`.
