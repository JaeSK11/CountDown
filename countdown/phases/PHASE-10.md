# Phase 10 — models-v2, scan-aware pipeline, census etiquette

**Pipeline:** stage 1 + stage 4 + reporting · **Status:** in progress (planned 2026-11-11,
started 2026-11-12); §5 tracks it. Picks up the Phase-5 seed checks (`PHASE-5.md` §10.7) and three points raised by
K. Webster, *There is no such thing as a benign internet scanner* (2025-01-29,
kenwebster.com): scanners are 35–70 % of connection attempts on an exposed host; no scanner
should be assumed benign; scan traffic is recognisable from its TCP signature.

---

## 1. models-v2: refit on all data

Refit each expert's deployed configuration on all of its rows (no held-out test), hash-pin as
`models-v2` like `models-v1`.

- E1: `flow_gbdt` alone (10 seeds: no ensemble gain).
- E2: `flow_gbdt` + `graph_gnn`, soft vote, temperature (5 seeds +0.024; 5 disjoint outer folds
  +0.021, 5/5 — `results/ensemble_seeds/E2-folds.md`).
- E3: three-member vote, temperature (10 seeds: +0.080, 10/10).
- Calibration temperatures can't be fit on test any more: fit them on out-of-fold predictions.
- **Exit:** `countdown models` lists v2 with hashes; `examples/` and the experiment tables
  regenerate unchanged in shape.

## 2. Scan-aware stage 1

Two gaps the article points at; both are rules, not models.

**2a. Count what we drop.** `flows/extract.py` and `flows/reassembly.py` drop flows under
`min_packets` (4). A SYN scan is 1–3 packets per target, so on an internet-facing capture most
connection attempts vanish uncounted. Add a per-capture tally of dropped short flows —
SYN-only, SYN/RST, SYN/SYN-ACK/RST — with distinct sources and destination ports, in the
`triage` summary and the `live` heartbeat.

**2b. Flag handshake-only flows.** A TLS scanner (our own census included) completes the
handshake and closes. It passes `min_packets`, gets a correct PQC verdict, and then a
meaningless E1 traffic class — there is no "scan" class. Mark flows with no application data
after the handshake as `traffic_class = "handshake_only"` (E1/E2/E3 not consulted), keep the
PQC verdict, and give them their own row in the triage tables.

- Test material: the Phase-7/8 census captures are labelled handshake-only traffic already.
- Not in scope: TCP window/option signatures as features (the article's SYN-ACK fingerprints).
  None of our features use them and no dataset labels scans; that would need new data and a
  new class.
- **Exit:** a synthetic SYN scan is counted, not classified; every census flow is
  `handshake_only` with its PQC verdict unchanged; ISCX/CSTNET/MobileApp results unchanged.

## 3. Census etiquette

Our weekly census is itself an internet scanner. It is low-rate (a few `openssl` handshakes
per site per week, ordinary kernel TCP stack, so not a SYN-signature scanner), but it should
still be identifiable and avoidable before results are published or a second vantage point
is added.

- A static page describing the probes (who, what, how often, contact) and a reverse-DNS name
  for the probing address pointing at it.
- An exclusion list (`scripts/census_exclude.txt`) honoured by `pqc_census.py`; excluded sites
  are reported as excluded, never as unreachable.
- Keep the client honest: plain OpenSSL ClientHellos, no faked browser fingerprint — the
  census must not pretend to be a browser.
- Note in the census README: sites that block scanners show up as unreachable, which shrinks
  the denominator, not the quantum-safe share's correctness.
- **Exit:** page + exclusion list live; the scheduler logs the exclusion count per run.

## 4. Order

1 → 2a → 2b → 3 before any public census result.

---

## 5. Status

| § | task | state |
| --- | --- | --- |
| 1 | `models-v2` trained (`runs/models-v2/`) | E3 ✅, E1 ✅, E2 running |
| 1 | published + manifest | after E2 |
| 2a | short-flow tally | ✅ (§6) |
| 2b | handshake-only flows | ✅ (§7) |
| 3 | census etiquette | not started |

## 6. Built: the short-flow tally (2a)

`countdown/flows/short_flows.py`. The reassembler now also records, per flow, whether it saw a
SYN, a SYN-ACK, an RST and how many packets carried payload; every flow it drops for being
under `min_packets` is tallied by shape — `syn_only`, `syn_rst`, `half_open` (the three
*scan-like* shapes), `tcp_payload`, `tcp_other`, `udp` — with the number of distinct scan
initiators, targets and destination ports. Only counts leave the tally, never an address.

- `countdown analyze` / `live`: a `short flows` line in the stderr summary; `live` also prints
  it every 60 s, so a capture that is nothing but a scan still reports something.
- `countdown triage`: a "Flows dropped before classification" table, and `short_flows` in the
  JSON.
- Checked on a synthetic scan (4 filtered, 3 closed, 3 open ports + one real connection): 10
  scan-like, 1 stray ACK, the real connection classified — `tests/test_short_flows.py`.
- Emitted flows are unchanged: the tally only looks at what was already dropped.

## 7. Built: handshake-only flows (2b)

`countdown/crypto/handshake_only.py`. A TCP TLS flow is `handshake_only` when the whole
client stream is in the reassembled head (`handshake_bytes_seen` = head length) and, ignoring
19-byte records, the client sent no content-type-23 record beyond its TLS 1.3 Finished (none
at all on TLS 1.2). The pipeline then sets `traffic_class = "handshake_only"`,
`route_reason = "handshake_only"`, consults no expert and keeps the PQC verdict; `triage`
counts them separately and leaves them out of the class mixes.

**Measured, not assumed** (golden capture `tests/golden/handshake_only/`, this machine,
2026-11-12): `openssl s_client` over TLS 1.3 sends ClientHello, CCS, Finished (69 B) and an
encrypted close_notify (**19 B**); over TLS 1.2 no type-23 record at all; a real HTTPS GET adds
its request (73–80 B) after the Finished. All five probes are flagged, the session is not.

**Effect on existing results:**
- The labelled datasets' cached flows carry no `handshake_bytes_seen`, so nothing in them is
  flagged: training data, experiment tables and the paper are unchanged.
- `examples/` and the synthetic PQC captures in `results/experiments/triage.json` are bare
  handshakes: all three are now `handshake_only`. Before, E1 had named the two exposed ones
  `voip` (0.40) and `chat` (0.65). Verdicts, HNDL scores and mitigation codes are unchanged.
- The census's own flows are handshake-only too, as intended.

**Limits:** TCP only — QUIC encrypts everything after the Initial, so a QUIC probe still gets a
class. A server-first protocol whose client never speaks would also be flagged.
