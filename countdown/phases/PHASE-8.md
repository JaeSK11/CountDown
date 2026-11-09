# Phase 8 — QUIC/HTTP3 census, and two QUIC parser fixes it turned up

**Pipeline:** stages 2–3 + reporting · **Deliverable:** the first of the Phase-7 next candidates
(`PHASE-7.md`, end): a QUIC profile for the PQC census, measured the same day as the TCP one.

Started and finished 2026-11-09. The model zoo is unchanged (`models-v1`). No new dependencies:
the image's OpenSSL 3.5.7 already has a QUIC client (`s_client -quic`).

---

## 1. What was added and changed

```
scripts/pqc_census.py        # profile `quic`, --profiles, --quic-timeout, QUIC-vs-TCP section,
                             #   transport-aware join, kernel drop count, "server reply not read"
countdown/crypto/quic_parser.py   # resync past short-header packets; Initial keys follow a Retry
countdown/sources/live.py    # kernel_drops / kernel_seen from PACKET_STATISTICS
tests/test_census.py         # new: trace parsing, join, comparison, metric
tests/test_quic_parser.py    # + resync and Retry tests
tests/golden/quic/rfc9001_a4_retry.hex   # RFC 9001 A.4 Retry, tag checked under the s5.8 key
results/phase8/census/       # census.{json,md}, sites.csv
```

## 2. How the QUIC profile works

```bash
docker compose -f docker/compose.yaml run --rm job python-netraw scripts/pqc_census.py \
    --sites ../data/_census/top-1m.csv --tranco-list 746WX --top 1000 \
    --profiles tls13,tls12,quic --out results/phase8/census
```

- **Probe:** `openssl s_client -quic -alpn h3` per site, the same default groups as `tls13`
  (X25519MLKEM768 first), over QUIC v1 on UDP 443. A bare handshake, no HTTP/3 request. A host
  without HTTP/3 never answers (the client keeps retransmitting its Initial), so a QUIC probe
  times out after `--quic-timeout` (5 s); apex then `www.` as before.
- **Ground truth:** `s_client` does not print the negotiated group for QUIC. With `-trace` it
  dumps the decrypted ServerHello, and the census reads the key share from that
  (`_trace_server_group`: the last ServerHello, so a HelloRetryRequest is handled; the trace
  spells x25519 `ecdh_x25519`).
- **Passive side:** the same capture as the TCP profiles; port 443 now matches UDP too. The
  pipeline already decrypted QUIC Initials (Phase 2, task 2.5); nothing new was needed to get
  records.
- **Join:** a probe is joined only to a record of its own transport (`proto` tcp / udp), so a
  host's QUIC and TCP connections can never be swapped.
- **All three profiles in one run,** so the TCP/QUIC comparison is same-day, same network.

## 3. Results (2026-11-09, one run, one vantage point)

| | `tls13` (TCP) | `tls12` (TCP) | `quic` (UDP) |
| --- | --- | --- | --- |
| handshakes completed / captured | 760 / 760 | 761 / 761 | **306 / 306** |
| **quantum-safe (X25519MLKEM768)** | **461 (60.7 %)** | 0 (none possible) | **294 (96.1 %)** |
| other groups | x25519 212, secp256r1 62, … | x25519 627, secp256r1 87, … | x25519 12 |
| **passive group vs client's report** | 747 match, 5 not captured, **0 wrong** | 730 match, 24 not captured, **0 wrong** | **306 match, 0 wrong** |

Capture: 43,789 frames, **0 dropped by the kernel**.

**QUIC vs TCP on the 306 sites that answered both:**

| | quantum-safe over QUIC | classical over QUIC |
| --- | --- | --- |
| **quantum-safe over TCP** | 250 | **0** |
| **classical over TCP** | **44** | 12 |

1. **40 % of the sites that speak TLS answer HTTP/3 directly** (306 of 760). No site answered
   QUIC but not TCP.
2. **Over QUIC, 96 % negotiate a hybrid post-quantum exchange; the same sites manage 82 % over
   TCP.** The difference is one-sided: 44 sites are post-quantum on QUIC and classical on
   TCP, and none the other way. The 44 are mostly sites behind a large CDN or edge stack whose
   QUIC termination already does ML-KEM while the TCP one does not: the whole Amazon retail
   family (13 domains), Bing, AOL, Dropbox, Dell, Intel, Samsung, Etsy, IMDb, Wix, usda.gov and
   others (`census.json` → `quic_vs_tcp.group_differs`). Their TCP groups were x25519 (30),
   secp256r1 (11), secp521r1 (1), unseen (2).
3. **What that does to the headline:** a client that uses HTTP/3 where the server offers it is
   quantum-safe with **505 of 760 sites (66.4 %)**: the 461 TCP ones plus the 44. The Phase-7
   60 % is the TCP-only figure. Caveat: a browser uses h3 only after it learns of it (Alt-Svc
   or an HTTPS DNS record); the census dials QUIC directly, so this is an upper bound on what
   a browser gets on a first visit.
4. **The 12 QUIC servers that stayed classical** (x25519) are Yandex (3 domains), Alibaba/
   Taobao, Tencent, Shopee (2), Slack, Box, Xiaohongshu, adgrx.com.
5. **TCP, five days after Phase 7** (2026-11-04 → 11-09, same list): 455 → 461 quantum-safe.
   Of the 755 sites captured both times, 6 became quantum-safe (airbnb.com, aliyun.com,
   cnbc.com, inmobi.com, latimes.com, netflix.net) and 2 went back to classical (clarity.ms,
   comcast.com); livejournal.com and visa.cn are quantum-safe now but did not complete a
   handshake in Phase 7. The two full runs on 11-09 (before and after the Retry fix, which does
   not touch TCP) differed by 1 site on `tls13`, so a site moving either way can be a different
   backend answering, not a rollout; with two dated points it is not yet a trend
   (`results/census/trend.md` tracks it from here).
6. **Large-ClientHello intolerance, again:** 4 sites completed the TLS 1.2 handshake but not
   the default ML-KEM one (aliyuncs.com, manulife-sinochem.com, netcraze.io, netease.com). Only
   visa.cn overlaps the Phase-7 list of 6, and visa.cn is now quantum-safe, so this looks
   per-run, not per-site.

## 4. How it was checked: two QUIC parser bugs

The first 30-site and 1,000-site runs were not clean: in the 1,000-site run 4 QUIC records had
basis `offered` (server reply not read) and the census counted 3 of them as **wrong**. Both
causes were in `crypto/quic_parser.py`, and both were found by capturing the failing hosts to a
pcap and dissecting their datagrams.

- **A Retry changes the key root.** yandex.ru, yandex.com, ya.ru and wordpress.org answer the
  first Initial with a **Retry**. The client then re-sends its Initial to the Retry's SCID, and
  RFC 9001 §5.2 roots **both** directions' Initial keys at that SCID from then on. The parser
  knew only the original DCID, so it read the first ClientHello and nothing after the Retry.
  Fix: collect every Retry's SCID from the server head and try each root per Initial (AEAD
  authentication picks the right one). A Retry packet has no Length field, so the walk also
  now steps over it instead of mis-reading its token as a length. Test: RFC 9001 A.4's Retry
  vector (tag verified under the §5.8 key) in front of the A.2/A.3 plaintexts re-sealed under
  the Retry's SCID.
- **A short-header packet ended the walk.** A direction's head is its datagrams concatenated,
  and the walk stopped at the first packet that is not a long header. A 1-RTT packet has no
  length, so one 1-RTT datagram arriving before the second of two Initials (an ML-KEM
  ServerHello is ~1.2 kB, two packets) truncated the ServerHello. Fix: resynchronise on the
  next `0b11xxxxxx` + known-version header; a false match inside ciphertext fails AEAD and is
  skipped. Intermittent live (it depends on arrival order), reproduced exactly with the RFC
  vectors.
- **The census metric** counted an `offered` record as a match or as wrong by comparing the
  client's first offer to the server's choice. It now reports those as "server reply not
  read". After the two fixes there were none.
- **Kernel drops** were the first suspect and were ruled out: `LiveSource` now reads the
  socket's `PACKET_STATISTICS` and the census reports it (0 in every run).

**Effect on existing results.** `countdown experiments run triage paper` changes 2 flows:
PostQuantumTLS `at_risk` 12,997 → 12,999, `at_risk_ceiling` 21 → 19 (their server reply is now
read and is classical). No percentage and no paper table changes; the exposure figure
re-renders. `PHASE-7.md` §6.3 keeps its numbers as they were at the end of that phase.

## 5. Limits

- One run, from one network, through Docker's NAT, with one client (OpenSSL). Middleboxes on
  other networks drop or rate-limit UDP 443 differently.
- QUIC v1 only (OpenSSL's client offers nothing else); no HTTP/3 request, no Alt-Svc / HTTPS
  RR discovery (point 3).
- Tranco top 1,000 only: popular sites, so mostly large operators. The long tail is less
  likely to run QUIC at all.

## 6. Status

| # | task | status |
| --- | --- | --- |
| 8.1 | QUIC/HTTP3 profile in the census, same-day TCP comparison | ✅ §3: 306 QUIC sites, 96.1 % quantum-safe, 0 wrong |
| 8.2 | QUIC Retry key re-rooting | ✅ §4 |
| 8.3 | QUIC walk resync past short headers | ✅ §4 |
| 8.4 | kernel drop count in live capture | ✅ |

Tests: 623 passed.

**Next candidates** (carried from Phase 7): repeat the census on a schedule (done in
`PHASE-9.md`) or from other networks so the dated points become a trend; flow-level relabelling of the ISCX file-transfer
captures; ECH, and streaming/eBPF prefiltering.
