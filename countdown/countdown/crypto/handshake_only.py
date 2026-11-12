"""Handshake-only TLS flows: the client completed (or tried) a handshake and sent nothing else
(Phase 10, section 2b).

A TLS scanner -- our own census, Censys-style certificate grabbers -- opens a connection,
finishes the handshake and closes.  The PQC verdict of such a flow is real (it comes from the
handshake), but a traffic class is not: there is no traffic, and no "scan" class in any of
our label spaces, so E1 would still name one.  These flows get
``traffic_class = "handshake_only"`` instead of a model's guess.

The test reads the client's TLS records, which are readable only while the whole client
stream sits in the reassembled head window (``meta["handshake_bytes_seen"][0]`` equals the
head's length).  Measured on ``openssl s_client`` and on real HTTPS sessions (2026-11-12):

* TLS 1.3 ``s_client``: ClientHello, ChangeCipherSpec, then two content-type-23 records --
  the encrypted Finished (53 or 69 bytes) and the encrypted close_notify alert (**19** bytes:
  2 alert bytes + 1 inner type + 16 tag);
* TLS 1.2 ``s_client``: handshake records, ChangeCipherSpec, an alert (type 21) -- no type 23;
* a real session adds at least one more type-23 record (the request) after the Finished.

So: handshake-only when, ignoring 19-byte records, the client sent at most one type-23 record
(TLS 1.3: its Finished) or none (TLS 1.2 and earlier, whose Finished is type 22).

Limits: TCP only (QUIC encrypts everything after the Initial); a server-first protocol whose
client never speaks would also match.
"""

from __future__ import annotations

from typing import Any

APPLICATION_DATA = 23
HANDSHAKE = 22
ENCRYPTED_ALERT_LEN = 19


def tls_records(data: bytes) -> list[tuple[int, int]] | None:
    """``[(content_type, length), ...]`` if ``data`` is a whole number of TLS records."""
    out, i = [], 0
    while i < len(data):
        if i + 5 > len(data) or data[i] not in (20, 21, 22, 23) or data[i + 1] != 3:
            return None
        n = int.from_bytes(data[i + 3:i + 5], "big")
        out.append((data[i], n))
        i += 5 + n
    return out if i == len(data) else None


def is_handshake_only(flow: Any) -> bool:
    if flow.five_tuple.l4_proto != "tcp":
        return False
    cb = flow.handshake_client_bytes or b""
    if not cb or cb[0] != HANDSHAKE:
        return False
    meta = flow.meta or {}
    seen = (meta.get("handshake_bytes_seen") or [None])[0]
    if seen is None or int(seen) != len(cb) or not meta.get("handshake_client_complete", False):
        return False                       # the client sent more than we hold, or a gap
    recs = tls_records(cb)
    if recs is None:
        return False
    app = sum(1 for t, n in recs if t == APPLICATION_DATA and n != ENCRYPTED_ALERT_LEN)
    allowed = 1 if (flow.tls_version or "").endswith("1.3") else 0
    return app <= allowed
