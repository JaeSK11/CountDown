#!/usr/bin/env python
"""Post-quantum census of popular HTTPS servers, measured passively (Phase 7, task 7.4).

    docker compose -f docker/compose.yaml run --rm job \\
        python-netraw scripts/pqc_census.py --sites ../data/_census/top-1m.csv --top 1000

One process does both halves, so the capture ends exactly when the probes do:

* **capture** -- ``countdown live`` on the container's own interface (Docker's bridge network:
  it sees only this container's traffic), port 443, ``fast`` mode, through the full pipeline;
* **probe** -- for each site one ``openssl s_client`` handshake per profile (OpenSSL 3.5:
  ``tls13`` = its defaults, which offer X25519MLKEM768 first; ``tls12`` = ``-tls1_2``, so
  the certificate chain crosses the wire in cleartext).  A bare handshake and close -- no
  HTTP request.  A site whose apex does not answer is retried once as ``www.<site>``.

What the client negotiated (``s_client``'s own report) is the ground truth the passive
verdicts are checked against.  Writes ``results/phase7/census/{census.json,census.md,
sites.csv}`` -- per-site public facts only (SNI, protocol, group, certificate key), no
addresses.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILES = {"tls13": [], "tls12": ["-tls1_2"]}

#: OpenSSL's group names -> the KEM registry's.
_OPENSSL_GROUPS = {"x25519": "x25519", "x448": "x448", "prime256v1": "secp256r1",
                   "secp256r1": "secp256r1", "secp384r1": "secp384r1", "secp521r1": "secp521r1",
                   "x25519mlkem768": "X25519MLKEM768", "secp256r1mlkem768": "SecP256r1MLKEM768",
                   "secp384r1mlkem1024": "SecP384r1MLKEM1024", "mlkem768": "MLKEM768",
                   "mlkem1024": "MLKEM1024", "mlkem512": "MLKEM512"}


def read_sites(path: Path, top: int) -> list[str]:
    out = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                out.append(row[1].strip())
            if len(out) >= top:
                break
    return out


def _group(text: str) -> str | None:
    m = re.search(r"Negotiated TLS1\.3 group: (\S+)", text)
    if not m:
        m = re.search(r"(?:Peer|Server) Temp Key: (?:ECDH, )?([A-Za-z0-9_]+)", text)
    if not m or m.group(1) == "<NULL>":
        return None
    return _OPENSSL_GROUPS.get(m.group(1).lower(), m.group(1))


def _leaf_key(text: str) -> str | None:
    """Key of the first PEM certificate in ``s_client`` output, as ``x509.py`` names it."""
    m = re.search(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", text, re.S)
    if not m:
        return None
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

    try:
        key = x509.load_pem_x509_certificate(m.group(0).encode()).public_key()
    except Exception:
        return "unparsed"
    if isinstance(key, rsa.RSAPublicKey):
        return f"RSA-{key.key_size}"
    if isinstance(key, ec.EllipticCurvePublicKey):
        return key.curve.name
    if isinstance(key, ed25519.Ed25519PublicKey):
        return "Ed25519"
    return type(key).__name__


def probe(host: str, profile: str, timeout: float) -> dict:
    t0 = time.time()
    cmd = ["openssl", "s_client", "-connect", f"{host}:443", "-servername", host, *PROFILES[profile]]
    try:
        p = subprocess.run(cmd, input=b"", capture_output=True, timeout=timeout)
        text = (p.stdout + p.stderr).decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        text = "TIMEOUT"
    proto = re.search(r"Protocol\s*(?:version)?\s*:\s*(TLSv[\d.]+)", text)
    cipher = re.search(r"(?:Ciphersuite|Cipher is|Cipher\s+:)\s*:?\s*(\S+)", text)
    # s_client prints a Protocol line even when the handshake failed; "Cipher is (NONE)" /
    # "Cipher : 0000" is how it reports that no session was established.
    ok = (proto is not None and "CONNECTED" in text and cipher is not None
          and cipher.group(1) not in ("(NONE)", "0000"))
    return {"host": host, "profile": profile, "ok": ok,
            "protocol": proto.group(1).replace("TLSv", "") if proto else None,
            "cipher": cipher.group(1) if cipher else None,
            "group": _group(text) if ok else None,
            "leaf_key": _leaf_key(text) if ok else None,
            "t_start": t0, "t_end": time.time()}


def probe_site(site: str, profile: str, timeout: float) -> dict:
    r = probe(site, profile, timeout)
    if not r["ok"] and not site.startswith("www."):
        r2 = probe(f"www.{site}", profile, timeout)
        if r2["ok"]:
            r = r2
    r["site"] = site
    return r


def join(probes: list[dict], records: list[dict]) -> None:
    """Attach to each probe the pipeline record of its connection: same SNI, first packet
    inside the probe's own time window.  A host is probed once per profile and the profiles
    run seconds apart, so the match is unique; a record is never given to two probes."""
    by_sni: dict[str, list[dict]] = {}
    for r in records:
        if r.get("sni") and r.get("start") is not None:
            by_sni.setdefault(r["sni"], []).append(r)
    used: set[int] = set()
    for p in sorted(probes, key=lambda p: p["t_start"]):
        cands = [r for r in by_sni.get(p["host"], [])
                 if id(r) not in used and p["t_start"] - 0.05 <= r["start"] <= p["t_end"]]
        p["record"] = min(cands, key=lambda r: r["start"]) if cands else None
        if p["record"] is not None:
            used.add(id(p["record"]))


def summarise(probes: list[dict], meta: dict) -> tuple[dict, str]:
    out: dict = {"meta": meta, "profiles": {}}
    md = ["# Post-quantum census of popular HTTPS servers (passive)", "",
          f"{meta['sites']} sites (Tranco list {meta['tranco_list']}, top {meta['sites']}), probed "
          f"{meta['date']} from one client (OpenSSL {meta['openssl']}); verdicts are the pipeline's, "
          "from the captured packets; `s_client`'s own report is the ground truth.", ""]
    for prof in PROFILES:
        ps = [p for p in probes if p["profile"] == prof]
        ok = [p for p in ps if p["ok"]]
        seen = [p for p in ok if p.get("record")]
        verdict = Counter(p["record"]["pqc_verdict"] for p in seen)
        qs = [p for p in seen if p["record"]["quantum_safe"]]
        groups = Counter(p["record"]["kem_group"] for p in seen)
        truth = [p for p in seen if p["group"]]
        agree_group = sum(1 for p in truth if p["record"]["kem_group"] == p["group"])
        unseen = sum(1 for p in truth if p["record"]["kem_group"] is None)
        wrong = len(truth) - agree_group - unseen
        res = {"probed": len(ps), "handshake_ok": len(ok), "captured": len(seen),
               "verdicts": dict(verdict.most_common()), "quantum_safe": len(qs),
               "quantum_safe_share": round(len(qs) / len(seen), 4) if seen else 0.0,
               "kem_groups": dict(groups.most_common(8)),
               "protocols": dict(Counter(p["protocol"] for p in ok).most_common()),
               "passive_group_vs_client": {"match": agree_group, "curve_not_captured": unseen,
                                           "wrong": wrong, "n": len(truth)}}
        if prof == "tls12":
            auth = Counter(p["record"].get("auth_verdict") for p in seen)
            keys = Counter(p["record"].get("cert_key") for p in seen if p["record"].get("cert_key"))
            k_ok = sum(1 for p in seen if p["leaf_key"] and p["record"].get("cert_key") == p["leaf_key"])
            k_n = sum(1 for p in seen if p["leaf_key"] and p["record"].get("cert_key"))
            res.update(auth_verdicts=dict(auth.most_common()), cert_keys=dict(keys.most_common(8)),
                       passive_cert_key_matches_client=[k_ok, k_n],
                       rsa_key_transport=sum(1 for p in seen if "disable_rsa_kex" in (p["record"].get("mitigations") or [])))
        if prof == "tls12":
            ok13 = {p["site"] for p in probes if p["profile"] == "tls13" and p["ok"]}
            only12 = sorted(p["site"] for p in ok if p["site"] not in ok13)
            res["completed_only_without_pq_client_hello"] = only12
        out["profiles"][prof] = res
        md += [f"## Profile `{prof}`", "",
               f"{res['probed']} sites probed, {res['handshake_ok']} handshakes completed, "
               f"{res['captured']} captured and joined to a pipeline record.", "",
               f"- **quantum-safe (negotiated hybrid/PQC): {res['quantum_safe']} "
               f"({res['quantum_safe_share']:.1%})**",
               f"- verdicts: " + ", ".join(f"{k} {v}" for k, v in res["verdicts"].items()),
               f"- key-exchange groups: " + ", ".join(f"{k} {v}" for k, v in res["kem_groups"].items()),
               f"- protocols negotiated: " + ", ".join(f"{k} {v}" for k, v in res["protocols"].items()),
               f"- passive group vs the client's report ({len(truth)} handshakes): {agree_group} match, "
               f"{unseen} curve outside the captured window (verdict still classical/negotiated), "
               f"**{wrong} wrong**"]
        if prof == "tls12":
            md += [f"- certificate verdicts: " + ", ".join(f"{k} {v}" for k, v in res["auth_verdicts"].items()),
                   f"- leaf keys: " + ", ".join(f"{k} {v}" for k, v in res["cert_keys"].items()),
                   f"- passive leaf key == client's parse: {k_ok}/{k_n}",
                   f"- RSA key transport negotiated: {res['rsa_key_transport']}",
                   f"- completed with this TLS 1.2 ClientHello but **not** with the default "
                   f"(ML-KEM-offering) one: {len(res['completed_only_without_pq_client_hello'])}"]
        md.append("")
    return out, "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help="Tranco CSV (rank,domain)")
    ap.add_argument("--tranco-list", default="unknown", help="Tranco list id, for the record")
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "results/phase7/census"))
    args = ap.parse_args()

    from countdown.pipeline import Pipeline, PipelineConfig
    from countdown.sources.live import LiveSource

    sites = read_sites(Path(args.sites), args.top)
    pipe = Pipeline(PipelineConfig(mode="fast"))
    records: list[dict] = []
    stop = threading.Event()
    src = LiveSource(args.iface, ports=[443], stop=stop)
    cap = threading.Thread(target=pipe.analyze_stream, args=(src, records.extend),
                           kwargs={"batch_seconds": 5.0}, daemon=True)
    cap.start()
    time.sleep(2.0)
    probes: list[dict] = []
    with ThreadPoolExecutor(args.workers) as pool:
        for prof in PROFILES:
            t = time.time()
            probes += list(pool.map(lambda s: probe_site(s, prof, args.timeout), sites))
            print(f"[census] {prof}: {sum(p['ok'] for p in probes if p['profile'] == prof)}/"
                  f"{len(sites)} handshakes in {time.time() - t:.0f}s", file=sys.stderr)
            time.sleep(3.0)          # keeps the profiles' time windows apart
    time.sleep(3.0)
    stop.set()
    cap.join(timeout=120)
    join(probes, records)

    ver = subprocess.run(["openssl", "version"], capture_output=True, text=True).stdout.split()
    meta = {"sites": len(sites), "tranco_list": args.tranco_list, "date": time.strftime("%Y-%m-%d"),
            "openssl": ver[1] if len(ver) > 1 else "?", "records_captured": len(records)}
    res, md = summarise(probes, meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(json.dumps(res, indent=2) + "\n")
    (out / "census.md").write_text(md)
    keep = ("pqc_verdict", "pqc_basis", "kem_group", "quantum_safe", "tls_version", "cipher",
            "auth_verdict", "cert_key", "hndl_score", "hndl_level")
    with open(out / "sites.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "host", "profile", "client_ok", "client_protocol", "client_group",
                    "client_leaf_key", *keep, "mitigations"])
        for p in probes:
            r = p.get("record") or {}
            w.writerow([p["site"], p["host"], p["profile"], p["ok"], p["protocol"], p["group"],
                        p["leaf_key"], *[r.get(k) for k in keep], " ".join(r.get("mitigations") or [])])
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
