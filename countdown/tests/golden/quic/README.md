# QUIC golden vectors

Published test vectors from **RFC 9001, Appendix A** ("Sample Packet Protection"), copied
verbatim from https://www.rfc-editor.org/rfc/rfc9001.txt.

| file | RFC section | contents |
| --- | --- | --- |
| `rfc9001_a2_client_initial.hex` | A.2 | the complete protected client Initial (1200 B), carrying a real ClientHello for `example.com` |
| `rfc9001_a3_server_initial.hex` | A.3 | the complete protected server Initial (135 B), carrying the matching ServerHello |
| `rfc9001_a4_retry.hex` | A.4 | the server's Retry (SCID `f067a5502a4262b5`, token `token`); its integrity tag verifies under the RFC 9001 §5.8 key |

Both derive from the client-chosen Destination Connection ID `8394c8f03e515708`.

These exist so the QUIC key schedule is checked against an **independent** implementation
of RFC 9001 rather than against our own encryptor: a round-trip test (encrypt and decrypt
with the same derived keys) passes even when every secret is wrong, and would tell us
nothing about whether we can read real traffic.
