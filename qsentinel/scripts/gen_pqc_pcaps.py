#!/usr/bin/env python3
"""Generate guaranteed PQC / classical handshake captures (Phase 2, task 2.8).

Why this exists: PostQuantumTLS was captured in 2023, when PQC adoption was ~0, so it
cannot be relied on for positives.  Without guaranteed positives the PQC-detection recall
figure would be computed over an empty set.  This script manufactures them against real
servers, so the *negotiated group* -- the field the whole verdict keys off -- is genuine
production output, not a fixture we wrote to match our own parser.

How it works, and the one caveat to state plainly:

    The **ServerHello is real.**  It is produced by Cloudflare's (or the target's) TLS
    stack, and it is the authoritative field: `key_share.group` is the server's selection.

    The **ClientHello is ours.**  Local OpenSSL is 3.0.x and has no ML-KEM, so no local
    TLS client can offer X25519MLKEM768.  Instead we hand-build an RFC 8446-conformant
    ClientHello that offers it.  The ML-KEM encapsulation key is syntactically valid --
    768 coefficients below q packed 12-bit per FIPS 203, which passes the modulus check
    servers apply -- but is not a real keypair.  That is sound because we never complete
    the handshake: we send one ClientHello, read the ServerHello, and stop.  Nothing about
    the server's group selection depends on us being able to decapsulate.

Three captures are produced, and the third is the interesting one:

    hybrid      offer X25519MLKEM768 to a PQC-enabled host  -> negotiated hybrid
    classical   offer only x25519                            -> negotiated classical
    downgrade   offer X25519MLKEM768 to a non-PQC host       -> negotiated *classical*

The downgrade case is the direct test of offered-vs-negotiated: a client that offered PQC
on a connection that did not get it.  A verdict engine keying off the ClientHello would
call it `hybrid`; keying off the ServerHello, as ours does, it is `classical`.

Usage:
    python scripts/gen_pqc_pcaps.py [--out data/_pqc_synth] [--host-hybrid HOST] ...
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import struct
import sys
from pathlib import Path

import dpkt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsentinel.config import get_logger, setup_logging  # noqa: E402
from qsentinel.crypto.tls_parser import parse_tls_handshake  # noqa: E402

log = get_logger("gen_pqc_pcaps")

# Named groups we offer.
G_X25519MLKEM768 = 0x11EC
G_X25519 = 0x001D
G_SECP256R1 = 0x0017

TLS13_SUITES = (0x1301, 0x1302, 0x1303)
SIG_ALGS = (0x0403, 0x0804, 0x0401, 0x0503, 0x0805, 0x0501, 0x0806, 0x0601)

MSS = 1400  # segment the flight, so Phase 1's head reassembly is exercised too


# -- ML-KEM encapsulation key --------------------------------------------------------------

def mlkem768_encapsulation_key(rng: random.Random) -> bytes:
    """1184 bytes: ByteEncode_12 of 768 coefficients in [0, q) followed by a 32-byte rho.

    FIPS 203 requires ``ByteEncode(ByteDecode(ek)) == ek``, so servers reject an ek made of
    uniform random bytes (each 12-bit coefficient must be < q = 3329, which random bytes
    fail with overwhelming probability).  Sampling the coefficients first and encoding them
    produces a key that passes that check without being a real keypair.
    """
    coeffs = [rng.randrange(3329) for _ in range(768)]
    out = bytearray()
    for i in range(0, len(coeffs), 2):
        a, b = coeffs[i], coeffs[i + 1]
        out += bytes([a & 0xFF, ((a >> 8) | (b << 4)) & 0xFF, (b >> 4) & 0xFF])
    assert len(out) == 1152, len(out)
    return bytes(out) + os.urandom(32)


def x25519mlkem768_share(rng: random.Random) -> bytes:
    """The client share for X25519MLKEM768: ML-KEM-768 ek ‖ X25519 public key (1216 B).

    The order is load-bearing and is *not* the same for every hybrid -- RFC 10024 puts the
    ML-KEM key first for X25519MLKEM768 and the ECDH key first for SecP256r1MLKEM768.
    Reversing it draws a TLS alert instead of a ServerHello (verified against Cloudflare).
    """
    return mlkem768_encapsulation_key(rng) + os.urandom(32)


# -- ClientHello construction --------------------------------------------------------------

def _ext(ext_type: int, body: bytes) -> bytes:
    return struct.pack(">HH", ext_type, len(body)) + body


def _vec(body: bytes, len_bytes: int) -> bytes:
    return len(body).to_bytes(len_bytes, "big") + body


def build_client_hello(host: str, groups: list[int],
                       shares: list[tuple[int, bytes]]) -> bytes:
    """An RFC 8446 TLS 1.3 ClientHello record offering ``groups``."""
    exts = b""
    exts += _ext(0, _vec(b"\x00" + _vec(host.encode(), 2), 2))              # server_name
    exts += _ext(10, _vec(b"".join(struct.pack(">H", g) for g in groups), 2))  # supported_groups
    exts += _ext(13, _vec(b"".join(struct.pack(">H", s) for s in SIG_ALGS), 2))
    exts += _ext(43, _vec(struct.pack(">H", 0x0304), 1))                    # supported_versions
    exts += _ext(45, _vec(b"\x01", 1))                                      # psk_dhe_ke
    key_shares = b"".join(struct.pack(">H", g) + _vec(k, 2) for g, k in shares)
    exts += _ext(51, _vec(key_shares, 2))                                   # key_share
    exts += _ext(16, _vec(_vec(b"http/1.1", 1), 2))                         # ALPN

    body = (
        struct.pack(">H", 0x0303)          # legacy_version
        + os.urandom(32)                   # random
        + _vec(os.urandom(32), 1)          # legacy_session_id
        + _vec(b"".join(struct.pack(">H", c) for c in TLS13_SUITES), 2)
        + _vec(b"\x00", 1)                 # compression: null
        + _vec(exts, 2)
    )
    msg = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + len(msg).to_bytes(2, "big") + msg


# -- network probe ---------------------------------------------------------------------------

def probe(host: str, client_hello: bytes, port: int = 443,
          timeout: float = 15.0, want: int = 6000) -> tuple[str, int, str, int, bytes]:
    """Send one ClientHello, read the server flight, return the 5-tuple and response.

    We never complete the handshake -- one flight in, one flight out.  That is all the
    ServerHello needs, and it keeps the probe from being a real client session.
    """
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        local_ip, local_port = sock.getsockname()[:2]
        remote_ip, remote_port = sock.getpeername()[:2]
        sock.sendall(client_hello)
        buf = b""
        try:
            while len(buf) < want:
                chunk = sock.recv(16384)
                if not chunk:
                    break
                buf += chunk
        except socket.timeout:
            pass
    finally:
        sock.close()
    return local_ip, local_port, remote_ip, remote_port, buf


# -- pcap writing ------------------------------------------------------------------------------

def _frame(src_ip: str, dst_ip: str, sport: int, dport: int, payload: bytes,
           flags: int, seq: int, ack: int) -> bytes:
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, seq=seq, ack=ack, data=payload)
    ip = dpkt.ip.IP(src=socket.inet_aton(src_ip), dst=socket.inet_aton(dst_ip),
                    p=dpkt.ip.IP_PROTO_TCP, data=tcp)
    ip.len = len(bytes(ip))
    return bytes(dpkt.ethernet.Ethernet(src=b"\xaa" * 6, dst=b"\xbb" * 6,
                                        type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def write_pcap(path: Path, client_ip: str, client_port: int, server_ip: str,
               server_port: int, client_bytes: bytes, server_bytes: bytes) -> None:
    """Wrap the real handshake bytes in a well-formed TCP conversation.

    The handshake bytes are exactly what crossed the wire; the TCP/IP framing is
    reconstructed (we cannot capture raw packets without CAP_NET_RAW).  A full three-way
    handshake is emitted so Phase 1 sees a SYN and treats offset 0 as a trusted stream
    start, and the flight is segmented at MSS so head reassembly is genuinely exercised.
    """
    SYN, ACK, PSH, FIN = 0x02, 0x10, 0x08, 0x01
    pkts: list[tuple[float, bytes]] = []
    t = 0.0
    cseq, sseq = 1000, 5000

    pkts.append((t, _frame(client_ip, server_ip, client_port, server_port, b"", SYN, cseq, 0)))
    t += 0.01
    pkts.append((t, _frame(server_ip, client_ip, server_port, client_port, b"",
                           SYN | ACK, sseq, cseq + 1)))
    t += 0.01
    cseq += 1
    sseq += 1
    pkts.append((t, _frame(client_ip, server_ip, client_port, server_port, b"", ACK, cseq, sseq)))

    for i in range(0, len(client_bytes), MSS):
        chunk = client_bytes[i:i + MSS]
        t += 0.001
        last = i + MSS >= len(client_bytes)
        pkts.append((t, _frame(client_ip, server_ip, client_port, server_port, chunk,
                               (PSH | ACK) if last else ACK, cseq, sseq)))
        cseq += len(chunk)

    for i in range(0, len(server_bytes), MSS):
        chunk = server_bytes[i:i + MSS]
        t += 0.002
        last = i + MSS >= len(server_bytes)
        pkts.append((t, _frame(server_ip, client_ip, server_port, client_port, chunk,
                               (PSH | ACK) if last else ACK, sseq, cseq)))
        sseq += len(chunk)

    t += 0.01
    pkts.append((t, _frame(client_ip, server_ip, client_port, server_port, b"",
                           FIN | ACK, cseq, sseq)))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        for ts, buf in pkts:
            writer.writepkt(buf, ts=ts)


# -- cases ---------------------------------------------------------------------------------------

def run_case(name: str, host: str, groups: list[int], shares: list[tuple[int, bytes]],
             out_dir: Path, expect: str) -> dict[str, object] | None:
    log.info("[%s] offering %s to %s", name, [f"0x{g:04X}" for g in groups], host)
    # Build once: the ClientHello carries fresh randomness, so rebuilding it would put
    # different bytes in the pcap than the ones the server actually replied to.
    client_hello = build_client_hello(host, groups, shares)
    try:
        cip, cport, sip, sport, resp = probe(host, client_hello)
    except OSError as exc:
        log.error("[%s] probe failed: %s", name, exc)
        return None
    if not resp:
        log.error("[%s] no response from %s", name, host)
        return None

    path = out_dir / f"{name}.pcap"
    write_pcap(path, cip, cport, sip, sport, client_hello, resp)

    hs = parse_tls_handshake(client_hello, resp)
    got = hs.negotiated_group_name
    ok = got is not None
    log.info("[%s] -> %s  negotiated=%s suite=%s  (%d B response)",
             name, "ok" if ok else "NO GROUP", got, hs.cipher_suite_name, len(resp))
    if resp[:1] == b"\x15":
        log.warning("[%s] server sent a TLS alert, not a ServerHello", name)
    return {"name": name, "host": host, "path": str(path), "negotiated": got,
            "expect": expect, "response_bytes": len(resp)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/_pqc_synth", type=Path)
    ap.add_argument("--host-hybrid", default="pq.cloudflareresearch.com",
                    help="a host known to support X25519MLKEM768")
    ap.add_argument("--host-classical", default="pq.cloudflareresearch.com",
                    help="host for the classical-only capture")
    ap.add_argument("--host-downgrade", default="www.debian.org",
                    help="a host that does NOT support PQC, for the offered!=negotiated case")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()

    rng = random.Random(args.seed)
    out = Path(args.out)
    results = []

    r = run_case("hybrid_x25519mlkem768", args.host_hybrid,
                 [G_X25519MLKEM768, G_X25519, G_SECP256R1],
                 [(G_X25519MLKEM768, x25519mlkem768_share(rng))], out, expect="hybrid")
    results.append(r)

    r = run_case("classical_x25519", args.host_classical,
                 [G_X25519, G_SECP256R1],
                 [(G_X25519, os.urandom(32))], out, expect="classical")
    results.append(r)

    r = run_case("downgrade_offered_pqc", args.host_downgrade,
                 [G_X25519MLKEM768, G_X25519, G_SECP256R1],
                 [(G_X25519MLKEM768, x25519mlkem768_share(rng)), (G_X25519, os.urandom(32))],
                 out, expect="classical")
    results.append(r)

    ok = [r for r in results if r]
    print()
    print(f"{'case':28s} {'negotiated':24s} {'expect':10s} path")
    for r in ok:
        print(f"{r['name']:28s} {str(r['negotiated']):24s} {r['expect']:10s} {r['path']}")
    hybrids = [r for r in ok if r["negotiated"] in ("X25519MLKEM768", "X25519Kyber768Draft00")]
    classicals = [r for r in ok if r["negotiated"] in ("x25519", "secp256r1")]
    print(f"\n{len(hybrids)} hybrid capture(s), {len(classicals)} classical capture(s)")
    # Task 2.8 DoD: at least one hybrid and at least one classical capture.
    return 0 if (hybrids and classicals) else 1


if __name__ == "__main__":
    raise SystemExit(main())
