# Post-quantum census of popular HTTPS servers (passive)

1000 sites (Tranco list 746WX, top 1000), probed 2026-11-04 from one client (OpenSSL 3.5.7); verdicts are the pipeline's, from the captured packets; `s_client`'s own report is the ground truth.

## Profile `tls13`

1000 sites probed, 759 handshakes completed, 758 captured and joined to a pipeline record.

- **quantum-safe (negotiated hybrid/PQC): 455 (60.0%)**
- verdicts: hybrid 455, classical 303
- key-exchange groups: X25519MLKEM768 455, x25519 216, secp256r1 61, None 13, secp521r1 8, secp384r1 5
- protocols negotiated: 1.3 638, 1.2 121
- passive group vs the client's report (751 handshakes): 745 match, 6 curve outside the captured window (verdict still classical/negotiated), **0 wrong**

## Profile `tls12`

1000 sites probed, 760 handshakes completed, 759 captured and joined to a pipeline record.

- **quantum-safe (negotiated hybrid/PQC): 0 (0.0%)**
- verdicts: classical 759
- key-exchange groups: x25519 623, secp256r1 88, None 32, secp521r1 9, secp384r1 7
- protocols negotiated: 1.2 760
- passive group vs the client's report (752 handshakes): 727 match, 25 curve outside the captured window (verdict still classical/negotiated), **0 wrong**
- certificate verdicts: classical 759
- leaf keys: RSA-2048 465, secp256r1 264, RSA-4096 15, secp384r1 9, RSA-3072 3
- passive leaf key == client's parse: 756/756
- RSA key transport negotiated: 7
- completed with this TLS 1.2 ClientHello but **not** with the default (ML-KEM-offering) one: 6

