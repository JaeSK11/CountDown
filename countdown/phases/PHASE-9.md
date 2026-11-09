# Phase 9 — Scheduled census and the adoption trend

**Pipeline:** reporting · **Deliverable:** the Phase-8 census as a time series. Started
2026-11-09. No new dependencies; the model zoo is unchanged (`models-v1`).

---

## 1. What was added

```
scripts/census_scheduler.py   # weekly daemon: due check, run into <date>.partial, rename, trend
scripts/census_trend.py       # every dated run -> results/census/trend.{json,md,pdf,png}
docker/compose.yaml           # service `census` (countdown-census, unless-stopped, no GPU);
                              #   shared settings split into x-base (no GPU) and x-common
tests/test_census_trend.py
results/census/               # trend.*; scheduled runs land in runs/<date>/
```

## 2. How it runs

```bash
docker compose -f docker/compose.yaml up -d census
docker compose -f docker/compose.yaml run --rm census python scripts/census_scheduler.py --status
```

- **When:** every 7 days at 03:00 UTC after the latest run (`CENSUS_EVERY_DAYS`,
  `CENSUS_HOUR_UTC`). The first scheduled run is due **2026-11-16 03:00 UTC** (7 days after
  the Phase-8 run). A run missed while the machine is off happens at the next 15-minute check.
- **What:** `pqc_census.py --profiles tls13,tls12,quic` on Tranco list `746WX`, top 1,000, the
  same panel as Phases 7–8, so a change is a server's, not the list's. Takes ~7 minutes.
- **Safety:** results go to `<date>.partial` and are renamed only on success, so an
  interrupted run is redone, never counted. A failure is retried after 6 h. The container
  uses the default bridge network (it sees only its own probes) and no GPU, so it runs beside
  the GPU queue.
- **It commits nothing.** New runs appear in `results/census/runs/` for a person to review
  and commit.
- **Log:** `runs/census/scheduler.log` (START / DONE / FAIL / TREND) and `runs/census/<date>.log`.

## 3. The trend

`scripts/census_trend.py` reads `results/phase7/census`, `results/phase8/census` and every
finished `results/census/runs/<date>/`, keeps one Tranco panel, and writes:

- per run: quantum-safe share over TCP (`tls13`), over QUIC, and "HTTP/3 where offered" (safe
  over either, out of the TCP sites); QUIC reachability; wrong passive verdicts (a point is
  trusted only at 0); kernel drops;
- per pair of consecutive runs: the sites captured both times that became quantum-safe or
  went back to classical, per profile;
- `trend.pdf`/`.png`: one line per measure, direct labels, the paper palette's categorical
  slots 1–3 (validated with the dataviz validator; aqua's low contrast is offset by the direct
  labels, markers and the table).

As of 2026-11-09 it has two points:

| date | TCP | QUIC | HTTP/3 where offered |
| --- | --- | --- | --- |
| 2026-11-04 | 60.0 % (455/758) | — | — |
| 2026-11-09 | 60.7 % (461/760) | 96.1 % (294/306) | 66.4 % (505/760) |

## 4. Limits

- One vantage point (this machine, through Docker's NAT). A second network is the next step
  for the "one network" limit; the scheduler would run unchanged on any host with the image.
- A fixed panel ages: the Tranco top 1,000 of 2026-11-03 drifts from the live ranking. Changing
  the list starts a new series (the trend compares one list only).
- Weekly points will show rollout over months, not days. A site flipping between two runs
  can be a different backend answering.

## 5. Status

| # | task | status |
| --- | --- | --- |
| 9.1 | scheduler daemon (compose `census`) | ✅ |
| 9.2 | trend table, churn and figure | ✅ |
| 9.3 | first scheduled run | due 2026-11-16 03:00 UTC |

Tests: 628 passed.
