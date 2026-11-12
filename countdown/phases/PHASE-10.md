# Phase 10 — models-v2, scan-aware pipeline, census etiquette (planned)

**Pipeline:** stage 1 + stage 4 + reporting · **Status:** planned, not started (written
2026-11-11). Picks up the Phase-5 seed checks (`PHASE-5.md` §10.7) and three points raised by
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
