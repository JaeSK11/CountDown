# Post-quantum adoption over time (passive census)

Tranco list `746WX`, top 1000, the same panel every run. Quantum-safe = negotiated X25519MLKEM768 (or another hybrid/PQC group), from the captured packets. "HTTP/3 where offered" = safe over TCP or over QUIC, out of the sites captured over TCP. A point is trusted only with 0 wrong passive verdicts.

| date | TCP (TLS 1.3 default) | QUIC (HTTP/3) | HTTP/3 where offered | QUIC sites | wrong | kernel drops | run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-04 | 60.0% (455/758) | — | — | — | 0 | — | `results/phase7/census` |
| 2026-11-09 | 60.7% (461/760) | 96.1% (294/306) | 66.4% (505/760) | 306 | 0 | 0 | `results/phase8/census` |

## Site changes between consecutive runs

| from → to | profile | sites in both | became quantum-safe | became classical |
| --- | --- | --- | --- | --- |
| 2026-11-04 → 2026-11-09 | tls13 | 755 | 6: airbnb.com, aliyun.com, cnbc.com, inmobi.com, latimes.com, netflix.net | 2: clarity.ms, comcast.com |

A site flipping between two runs can be a different backend answering, not a rollout; read a trend from several points, not from one pair.
