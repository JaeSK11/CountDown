#!/usr/bin/env python
"""Post-quantum adoption over time, from every dated census run (Phase 9).

    python scripts/census_trend.py            # -> results/census/trend.{json,md,pdf,png}

A *run* is any directory holding a ``census.json`` written by ``scripts/pqc_census.py``:
``results/phase7/census`` (2026-11-04, TCP only), ``results/phase8/census`` (2026-11-09,
TCP + QUIC) and the scheduled runs in ``results/census/runs/<date>/``.  Runs are ordered by
the date in their own metadata, and only runs on the same Tranco list are compared, so the
site panel is fixed and a change is a server's, not the list's.

Per run: the quantum-safe share over TCP (profile ``tls13``), over QUIC, and for a client
that uses HTTP/3 where the server answers it ("either": safe over TCP or QUIC, over the TCP
sites); the passive-vs-client error count, which must stay 0 for a point to be trusted; and
the per-site churn against the previous run.  Deterministic for the same runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_DIRS = [ROOT / "results/phase7/census", ROOT / "results/phase8/census"]
SCHEDULED = ROOT / "results/census/runs"
OUT = ROOT / "results/census"

#: categorical slots 1-3 of the paper palette (validated: dataviz validate_palette.js, light)
SERIES = {"tcp": ("TCP (TLS 1.3 default)", "#2a78d6", "o"),
          "quic": ("QUIC (HTTP/3)", "#eb6834", "s"),
          "either": ("HTTP/3 where offered", "#1baf7a", "D")}


def find_runs(extra: list[Path] = ()) -> list[Path]:
    dirs = [d for d in RUN_DIRS if (d / "census.json").exists()]
    if SCHEDULED.exists():
        dirs += sorted(d for d in SCHEDULED.iterdir()
                       if not d.name.endswith(".partial") and (d / "census.json").exists())
    dirs += [d for d in extra if (d / "census.json").exists()]
    return dirs


def _sites(run: Path) -> dict[tuple[str, str], dict]:
    out = {}
    path = run / "sites.csv"
    if path.exists():
        with open(path) as fh:
            for r in csv.DictReader(fh):
                out[(r["site"], r["profile"])] = r
    return out


def _captured(r: dict | None) -> bool:
    return bool(r) and r["client_ok"] == "True" and r["pqc_verdict"] not in ("", None)


def summarise_run(run: Path) -> dict:
    j = json.loads((run / "census.json").read_text())
    meta, prof = j["meta"], j["profiles"]
    sites = _sites(run)
    point = {"date": meta["date"], "tranco_list": meta.get("tranco_list"), "sites": meta["sites"],
             "path": str(run.relative_to(ROOT)), "profiles": sorted(prof),
             "kernel_drops": meta.get("capture_kernel_drops")}
    for key, name in (("tcp", "tls13"), ("quic", "quic")):
        p = prof.get(name)
        point[key] = None if p is None else {
            "ok": p["handshake_ok"], "captured": p["captured"], "safe": p["quantum_safe"],
            "share": p["quantum_safe_share"]}
    point["wrong"] = sum(p["passive_group_vs_client"]["wrong"] for p in prof.values())
    point["either"] = None
    if point["quic"] is not None:
        names = sorted({s for s, _ in sites})
        tcp = [s for s in names if _captured(sites.get((s, "tls13")))]
        safe = [s for s in tcp if sites[(s, "tls13")]["quantum_safe"] == "True"
                or (_captured(sites.get((s, "quic"))) and sites[(s, "quic")]["quantum_safe"] == "True")]
        point["either"] = {"sites": len(tcp), "safe": len(safe),
                           "share": round(len(safe) / len(tcp), 6) if tcp else 0.0}
    return point


def churn(prev: Path, cur: Path, profile: str) -> dict | None:
    """Sites captured in both runs on ``profile``, and which flipped quantum-safe."""
    a, b = _sites(prev), _sites(cur)
    both = sorted(s for (s, p) in b if p == profile and _captured(b[(s, p)]) and _captured(a.get((s, p))))
    if not both:
        return None
    safe = lambda d, s: d[(s, profile)]["quantum_safe"] == "True"
    return {"profile": profile, "common": len(both),
            "became_safe": [s for s in both if safe(b, s) and not safe(a, s)],
            "became_classical": [s for s in both if safe(a, s) and not safe(b, s)]}


def build(runs: list[Path]) -> dict:
    pts = sorted(((summarise_run(r), r) for r in runs), key=lambda x: (x[0]["date"], x[0]["path"]))
    lists = {p["tranco_list"] for p, _ in pts}
    if len(lists) > 1:
        # Different panels are not comparable; keep the list most runs used.
        main = max(lists, key=lambda l: sum(p["tranco_list"] == l for p, _ in pts))
        pts = [(p, r) for p, r in pts if p["tranco_list"] == main]
    out = {"tranco_list": pts[0][0]["tranco_list"] if pts else None,
           "points": [p for p, _ in pts], "churn": []}
    for (pa, ra), (pb, rb) in zip(pts, pts[1:]):
        for prof in ("tls13", "quic"):
            c = churn(ra, rb, prof)
            if c:
                out["churn"].append({"from": pa["date"], "to": pb["date"], **c})
    return out


def _pct(x: dict | None) -> str:
    return "—" if x is None else f"{x['share']:.1%} ({x['safe']}/{x.get('captured', x.get('sites'))})"


def markdown(res: dict) -> str:
    md = ["# Post-quantum adoption over time (passive census)", "",
          f"Tranco list `{res['tranco_list']}`, top {res['points'][0]['sites'] if res['points'] else '?'}, "
          "the same panel every run. Quantum-safe = negotiated X25519MLKEM768 (or another hybrid/PQC "
          "group), from the captured packets. \"HTTP/3 where offered\" = safe over TCP or over QUIC, "
          "out of the sites captured over TCP. A point is trusted only with 0 wrong passive verdicts.",
          "", "| date | TCP (TLS 1.3 default) | QUIC (HTTP/3) | HTTP/3 where offered | QUIC sites | wrong | kernel drops | run |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in res["points"]:
        md.append(f"| {p['date']} | {_pct(p['tcp'])} | {_pct(p['quic'])} | {_pct(p['either'])} | "
                  f"{p['quic']['ok'] if p['quic'] else '—'} | {p['wrong']} | "
                  f"{'—' if p['kernel_drops'] is None else p['kernel_drops']} | `{p['path']}` |")
    if res["churn"]:
        md += ["", "## Site changes between consecutive runs", "",
               "| from → to | profile | sites in both | became quantum-safe | became classical |",
               "| --- | --- | --- | --- | --- |"]
        for c in res["churn"]:
            fmt = lambda xs: (f"{len(xs)}: " + ", ".join(xs[:12]) + (" …" if len(xs) > 12 else "")) if xs else "0"
            md.append(f"| {c['from']} → {c['to']} | {c['profile']} | {c['common']} | "
                      f"{fmt(c['became_safe'])} | {fmt(c['became_classical'])} |")
        md += ["", "A site flipping between two runs can be a different backend answering, not a "
               "rollout; read a trend from several points, not from one pair."]
    return "\n".join(md) + "\n"


def figure(res: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from datetime import date

    from countdown.eval.report_paper import AXIS, GRID, INK, INK2, SURFACE

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    days = [date.fromisoformat(p["date"]) for p in res["points"]]
    for key, (label, colour, marker) in SERIES.items():
        xs = [d for d, p in zip(days, res["points"]) if p[key] is not None]
        ys = [100 * p[key]["share"] for p in res["points"] if p[key] is not None]
        if not xs:
            continue
        ax.plot(xs, ys, color=colour, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    lo, hi = min(days), max(days)
    pad = max((hi - lo).days * 0.08, 2)
    ax.set_xlim(mdates.date2num(lo) - pad, mdates.date2num(hi) + pad * 2.5)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(f"Quantum-safe share of the Tranco top {res['points'][0]['sites']} "
                 f"(list {res['tranco_list']})", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left", ncol=3)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "trend.pdf", out / "trend.png"]
    fig.savefig(paths[0], facecolor=SURFACE, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=200, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    res = build(find_runs())
    if not res["points"]:
        print("no census runs found", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trend.json").write_text(json.dumps(res, indent=2) + "\n")
    md = markdown(res)
    (out / "trend.md").write_text(md)
    if not args.no_figure:
        figure(res, out)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
