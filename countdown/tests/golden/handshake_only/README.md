# Handshake-only golden capture

`sclient_and_sessions.pcap` — captured 2026-11-12 on this machine (Docker bridge network, the
census's own vantage point) by `countdown`'s live source, port 443 only:

| flows | what |
| --- | --- |
| 3 | `openssl s_client` over TLS 1.3 to example.com, www.cloudflare.com, www.google.com |
| 2 | `openssl s_client -tls1_2` to example.org, www.wikipedia.org |
| 1 | a real HTTPS GET to example.com (request + response) |

The probes are exactly what `scripts/pqc_census.py` sends. A sixth flow (a full page download
from www.cloudflare.com, 300 packets) was cut to keep the file small. Used by
`tests/test_handshake_only.py` (Phase 10 §2b).
