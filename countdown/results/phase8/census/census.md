# Post-quantum census of popular HTTPS servers (passive)

1000 sites (Tranco list 746WX, top 1000), probed 2026-11-09 from one client (OpenSSL 3.5.7); verdicts are the pipeline's, from the captured packets; `s_client`'s own report is the ground truth.

## Profile `tls13`

1000 sites probed, 760 handshakes completed, 760 captured and joined to a pipeline record.

- **quantum-safe (negotiated hybrid/PQC): 461 (60.7%)**
- verdicts: hybrid 461, classical 299
- key-exchange groups: X25519MLKEM768 461, x25519 214, secp256r1 59, None 13, secp521r1 8, secp384r1 5
- protocols negotiated: 1.3 641, 1.2 119
- passive group vs the client's report (752 handshakes): 747 match, 5 curve outside the captured window (verdict still classical/negotiated), **0 wrong**

## Profile `tls12`

1000 sites probed, 761 handshakes completed, 761 captured and joined to a pipeline record.

- **quantum-safe (negotiated hybrid/PQC): 0 (0.0%)**
- verdicts: classical 761
- key-exchange groups: x25519 627, secp256r1 86, None 31, secp521r1 10, secp384r1 7
- protocols negotiated: 1.2 761
- passive group vs the client's report (754 handshakes): 730 match, 24 curve outside the captured window (verdict still classical/negotiated), **0 wrong**
- certificate verdicts: classical 761
- leaf keys: RSA-2048 463, secp256r1 266, RSA-4096 14, secp384r1 9, RSA-3072 3
- passive leaf key == client's parse: 755/755
- RSA key transport negotiated: 7
- completed with this TLS 1.2 ClientHello but **not** with the default (ML-KEM-offering) one: 4

## Profile `quic`

1000 sites probed, 306 handshakes completed, 306 captured and joined to a pipeline record.

- **quantum-safe (negotiated hybrid/PQC): 294 (96.1%)**
- verdicts: hybrid 294, classical 12
- key-exchange groups: X25519MLKEM768 294, x25519 12
- protocols negotiated: 1.3 306
- passive group vs the client's report (306 handshakes): 306 match, 0 curve outside the captured window (verdict still classical/negotiated), **0 wrong**

## QUIC vs TCP (`quic` vs `tls13`, same sites, same day)

- answered TLS 1.3-default over TCP: 760; answered QUIC v1 (h3): 306 (40.3% of the TCP sites)
- both captured: 306. Quantum-safe on both 250, TCP only 0, QUIC only 44, neither 12
- quantum-safe share on those 306 sites: TCP 81.7%, QUIC 96.1% (TCP over all 760 captured TCP sites: 60.7%)
- key-exchange group differs between the two: 44
  - aliexpress.com: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.ca: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.co.jp: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.co.uk: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.co.za: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.com: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.com.au: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.com.br: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.de: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.es: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.fr: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.in: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - amazon.it: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - aol.com: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - appsflyer.com: TCP secp256r1, QUIC X25519MLKEM768 (client saw secp256r1 / X25519MLKEM768)
  - autodesk.com: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - bing.com: TCP secp521r1, QUIC X25519MLKEM768 (client saw secp521r1 / X25519MLKEM768)
  - britannica.com: TCP secp256r1, QUIC X25519MLKEM768 (client saw secp256r1 / X25519MLKEM768)
  - cpanel.net: TCP x25519, QUIC X25519MLKEM768 (client saw x25519 / X25519MLKEM768)
  - dailymotion.com: TCP secp256r1, QUIC X25519MLKEM768 (client saw secp256r1 / X25519MLKEM768)

